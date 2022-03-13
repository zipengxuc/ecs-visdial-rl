import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import random


class Decoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 startToken,
                 endToken,
                 beta,
                 dropout=0,
                 **kwargs):
        super(Decoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        self.startToken = startToken
        self.endToken = endToken
        self.dropout = dropout
        self.beta = beta

        # Modules
        self.rnn = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=self.dropout)
        self.outNet = nn.Linear(self.rnnHiddenSize, self.vocabSize)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.all_log_probs = []
        self.all_mask = []

    def forward(self, encStates, inputSeq, seq=True):
        '''
        Given encoder states, forward pass an input sequence 'inputSeq' to
        compute its log likelihood under the current decoder RNN state.

        Arguments:
            encStates: (H, C) Tuple of hidden and cell encoder states
            inputSeq: Input sequence for computing log probabilities
            reconstruct_past_ques: Option whether there is a coherrence term
                                   for predicting question on past round

        Output:
            A (batchSize, length, vocabSize) sized tensor of log-probabilities
            obtained from feeding 'inputSeq' to decoder RNN at evert time step

        Note:
            Maximizing the NLL of an input sequence involves feeding as input
            tokens from the GT (ground truth) sequence at every time step and
            maximizing the probability of the next token ("teacher forcing").
            See 'maskedNll' in utils/utilities.py where the log probability of
            the next time step token is indexed out for computing NLL loss.
        '''
        if inputSeq is not None:
            if seq:
                inputSeq = self.wordEmbed(inputSeq)
            outputs, hidden = self.rnn(inputSeq, encStates)
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputSize = outputs.size()
            flatOutputs = outputs.contiguous().view(-1, outputSize[2])
            flatScores = self.outNet(flatOutputs)
            flatLogProbs = self.logSoftmax(flatScores*self.beta)
            logProbs = flatLogProbs.contiguous().view(outputSize[0], outputSize[1], -1)

        return logProbs, hidden

    def forwardDecode(self,
                      encStates,
                      maxSeqLen=20,
                      inference='sample',
                      beamSize=1, futureReward=True, run_mcts=False, num_beams_mcts = 10):
        '''
        Decode a sequence of tokens given an encoder state, using either
        sampling or greedy inference.

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            maxSeqLen : Maximum length of token sequence to generate
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width

        Notes:
            * This function is not called during SL pre-training
            * Greedy inference is used for evaluation
            * Sampling is used in RL fine-tuning
        '''
        if inference == 'greedy' and beamSize > 1:
            # Use beam search inference when beam size is > 1
            return self.beamSearchDecoder(encStates, beamSize, maxSeqLen)

        # Determine if cuda tensors are being used
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        self.samples = []
        self.mcts_samples = []
        self.mcts_samples_lens = []

        maxLen = maxSeqLen + 1  # Extra <END> token
        batchSize = encStates[0].size(1)
        # Placeholder for filling in tokens at evert time step
        seq = th.LongTensor(batchSize, maxLen + 1)
        seq.fill_(self.endToken)
        seq[:, 0] = self.startToken
        seq = Variable(seq, requires_grad=False)

        # Initial state linked from encStates
        hid = encStates

        sampleLens = th.LongTensor(batchSize).fill_(0)
        # Tensors needed for tracking sampleLens
        unitColumn = th.LongTensor(batchSize).fill_(1)
        mask = th.ByteTensor(seq.size()).fill_(0)

        self.saved_log_probs = []

        # Generating tokens sequentially
        for t in range(maxLen - 1):
            emb = self.wordEmbed(seq[:, t:t + 1])
            # emb has shape  (batch, 1, embedSize)
            output, hid = self.rnn(emb, hid)
            # output has shape (batch, 1, rnnHiddenSize)
            scores = self.outNet(output.squeeze(1))
            logProb = self.logSoftmax(scores*self.beta)

            # Explicitly removing padding token (index 0) and <START> token
            # (index -2) from logProbs so that they are never sampled.
            # This is allows us to keep <START> and padding token in
            # the decoder vocab without any problems in RL sampling.
            if t > 0:
                logProb = torch.cat([logProb[:, 1:-2], logProb[:, -1:]], 1)
            elif t == 0:
                # Additionally, remove <END> token from the first sample
                # to prevent the sampling of an empty sequence.
                logProb = logProb[:, 1:-2]
            # This also shifts end token index back by 1
            END_TOKEN_IDX = self.endToken - 1

            probs = torch.exp(logProb)

            #MCTS stuff
            if inference == 'sample' and run_mcts:
                # create samples for the hidden state
                startPrefix = seq[:, :t + 1].unsqueeze(1)
                if t != maxLen-1:
                    mcts_samples, mcts_seqLens = self.get_mcts_samples(hid, num_beams_mcts, maxLen - 1 - t, seq[:, t])
                    mcts_samples = torch.cat([startPrefix.repeat(1, num_beams_mcts, 1), mcts_samples], dim=2)
                    # updating lengths as well
                    mcts_seqLens = mcts_seqLens + t + 1
                    self.mcts_samples.append(mcts_samples)
                    self.mcts_samples_lens.append(mcts_seqLens)
                    # randomly select one of the MCTS sample to be the value on the next iteration
                    sample_index = random.randint(0,num_beams_mcts-1)
                    sample = mcts_samples[:,sample_index,t+1]
                    # log probs
                    # sample has ids wrt to the vocab. It has the following guarantees:
                    # start token/padding token is never sampled sample lens deal with the end token
                    # since we have remove the padding token in logProb ,we need to offset the sample by -1
                    sample = sample - 1
                    sample = sample.contiguous()
                    self.saved_log_probs.append(logProb.gather(1, sample.view(-1, 1)))
                    sample = sample.unsqueeze(-1)

            elif inference == 'sample' and not run_mcts:
                categorical_dist = Categorical(probs)
                sample = categorical_dist.sample()
                # Saving log probs for a subsequent reinforce call
                self.saved_log_probs.append(categorical_dist.log_prob(sample))
                sample = sample.unsqueeze(-1)

            elif inference == 'greedy':
                _, sample = torch.max(probs, dim=1, keepdim=True)
            else:
                raise ValueError("Invalid inference type: '{}'".format(inference))

            # Compensating for removed padding token prediction earlier
            sample = sample + 1  # Incrementing all token indices by 1

            self.samples.append(sample)
            seq.data[:, t + 1] = sample.squeeze().data
            # Marking spots where <END> token is generated
            mask[:, t] = sample.squeeze().data.eq(END_TOKEN_IDX)

            # Compensating for shift in <END> token index
            sample.data.masked_fill_(mask[:, t].unsqueeze(1), self.endToken)

        mask[:, maxLen - 1].fill_(1)

        # Computing lengths of generated sequences
        for t in range(maxLen):
            # Zero out the spots where end token is reached
            unitColumn.masked_fill_(mask[:, t], 0)
            # Update mask
            mask[:, t] = unitColumn
            # Add +1 length to all un-ended sequences
            sampleLens = sampleLens + unitColumn

        # Keep mask for later use in RL reward masking
        self.mask = Variable(mask, requires_grad=False)
        if inference == 'sample' and run_mcts:
            # we now have sequence lengths of all the samples. We cant generate MCTS samples for the
            # the last token. For that case, we will just duplicate the sample.
            self.mcts_samples.append(seq[:, :maxLen].unsqueeze(1).repeat(1, num_beams_mcts, 1))
            mcts_seqLens = Variable(sampleLens.unsqueeze(1).repeat(1,1,num_beams_mcts).squeeze(0))
            self.mcts_samples_lens.append(mcts_seqLens)

        # Adding <START> to generated answer lengths for consistency
        sampleLens = sampleLens + 1
        sampleLens = Variable(sampleLens, requires_grad=False)

        startColumn = sample.data.new(sample.size()).fill_(self.startToken)
        startColumn = Variable(startColumn, requires_grad=False)

        # Note that we do not add startColumn to self.samples itself
        # as reinforce is called on self.samples (which needs to be
        # the output of a stochastic function)
        gen_samples = [startColumn] + self.samples
        if futureReward:
            self.all_log_probs.append(self.saved_log_probs)
            self.all_mask.append(self.mask)

        samples = torch.cat(gen_samples, 1)
        return samples, sampleLens

    def evalOptions(self, encStates, options, optionLens, scoringFunction):
        '''
        Forward pass a set of candidate options to get log probabilities

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            options   : (batchSize, numOptions, maxSequenceLength) sized
                        tensor with <START> and <END> tokens

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.

        Output:
            A (batchSize, numOptions) tensor containing the score
            of each option sentence given by the generator
        '''
        batchSize, numOptions, maxLen = options.size()
        optionsFlat = options.contiguous().view(-1, maxLen)

        # Reshaping H, C for each option
        encStates = [x.unsqueeze(2).repeat(1,1,numOptions,1).\
                        view(self.numLayers, -1, self.rnnHiddenSize)
                        for x in encStates]

        logProbs, _ = self.forward(encStates, inputSeq=optionsFlat)
        scores = scoringFunction(logProbs, optionsFlat, returnScores=True)
        return scores.view(batchSize, numOptions)


    def reinforce(self,reward,futureReward=False,mcts=False):
        '''
        Compute loss using REINFORCE on log probabilities of tokens
        sampled from decoder RNN, scaled by input 'reward'.

        Note that an earlier call to forwardDecode must have been
        made in order to have samples for which REINFORCE can be
        applied. These samples are stored in 'self.saved_log_probs'.
        '''
        loss = 0
        # samples = torch.stack(self.samples, 1)
        # sampleLens = self.sampleLens - 1
        if len(self.saved_log_probs) == 0:
            raise RuntimeError("Reinforce called without sampling in Decoder")

        if not futureReward:
            for t, log_prob in enumerate(self.saved_log_probs):

                if not mcts:
                    # reward is a single vector
                    loss += -1 * log_prob * (reward.detach() * (self.mask[:, t].float()))
                else:
                    # reward is a matrix, different token have different returns from the environment
                    loss += -1 * log_prob * (reward[:, t].detach() * (self.mask[:, t].float()))
        else:
            print("Using future reward")
            # print(reward)

            for ind,log_probs in enumerate(self.all_log_probs):

                mask = self.all_mask[ind]
                cur_reward = reward[:,ind]
                # print(cur_reward)

                for t,log_prob in enumerate(log_probs):
                    loss += -1 * log_prob * (cur_reward.detach() * (mask[:, t].float()))

            # resetting the logs
            self.all_mask = []
            self.all_log_probs = []

        return loss

    def beamSearchDecoder(self, initStates, beamSize, maxSeqLen):
        '''
        Beam search for sequence generation
        Arguments:
            initStates - Initial encoder states tuple
            beamSize - Beam Size
            maxSeqLen - Maximum length of sequence to decode
        '''

        # For now, use beam search for evaluation only
        assert self.training == False

        # Determine if cuda tensors are being used
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        LENGTH_NORM = True
        maxLen = maxSeqLen + 1  # Extra <END> token
        batchSize = initStates[0].size(1)

        startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
        backVector = th.LongTensor(beamSize)
        torch.arange(0, beamSize, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batchSize, 1)

        tokenArange = th.LongTensor(self.vocabSize)
        torch.arange(0, self.vocabSize, out=tokenArange)
        tokenArange = Variable(tokenArange)

        startTokenArray = Variable(startTokenArray)
        backVector = Variable(backVector)
        hiddenStates = initStates

        # Inits
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(
            self.endToken)
        beamTokensTable = Variable(beamTokensTable)
        backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
        backIndices = Variable(backIndices)

        aliveVector = beamTokensTable[:, :, 0].eq(self.endToken).unsqueeze(2)

        for t in range(maxLen - 1):  # Beam expansion till maxLen]
            if t == 0:
                # First column of beamTokensTable is generated from <START> token
                emb = self.wordEmbed(startTokenArray)
                # emb has shape (batchSize, 1, embedSize)
                output, hiddenStates = self.rnn(emb, hiddenStates)
                # output has shape (batchSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze(1))
                logProbs = self.logSoftmax(scores*self.beta)
                # scores & logProbs has shape (batchSize, vocabSize)

                # Find top beamSize logProbs
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                logProbSums = topLogProbs

                # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                hiddenStates = [
                    x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                    for x in hiddenStates
                ]
                hiddenStates = [
                    x.view(self.numLayers, -1, self.rnnHiddenSize)
                    for x in hiddenStates
                ]
                # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
            else:
                # Subsequent columns are generated from previous tokens
                emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                # emb has shape (batchSize, beamSize, embedSize)
                output, hiddenStates = self.rnn(
                    emb.view(-1, 1, self.embedSize), hiddenStates)
                # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores*self.beta)
                # logProbs has shape (batchSize*beamSize, vocabSize)
                # NOTE: Padding token has been removed from generator output during
                # sampling (RL fine-tuning). However, the padding token is still
                # present in the generator vocab and needs to be handled in this
                # beam search function. This will be supported in a future release.
                logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                                                       self.vocabSize)

                if LENGTH_NORM:
                    # Add (current log probs / (t+1))
                    logProbs = logProbsCurrent * (aliveVector.float() /
                                                  (t + 1))
                    # Add (previous log probs * (t/t+1) ) <- Mean update
                    coeff_ = aliveVector.eq(0).float() + (
                        aliveVector.float() * t / (t + 1))
                    logProbs += logProbSums.unsqueeze(2) * coeff_
                else:
                    # Add currrent token logProbs for alive beams only
                    logProbs = logProbsCurrent * (aliveVector.float())
                    # Add previous logProbSums upto t-1
                    logProbs += logProbSums.unsqueeze(2)

                # Masking out along |V| dimension those sequence logProbs
                # which correspond to ended beams so as to only compare
                # one copy when sorting logProbs
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                mask_[:, :,
                      0] = 0  # Zeroing all except first row for ended beams
                minus_infinity_ = torch.min(logProbs).item()
                logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                logProbs = logProbs.view(batchSize, -1)
                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).\
                                repeat(batchSize,beamSize,1)
                tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                tokensArray = tokensArray.view(batchSize, -1)
                backIndexArray = backVector.unsqueeze(2).\
                                repeat(1,1,self.vocabSize).view(batchSize,-1)

                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                logProbSums = topLogProbs
                beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                backIndices[:, :, t] = backIndexArray.gather(1, topIdx)

                # Update corresponding hidden and cell states for next time step
                hiddenCurrent, cellCurrent = hiddenStates

                # Reshape to get explicit beamSize dim
                original_state_size = hiddenCurrent.size()
                num_layers, _, rnnHiddenSize = original_state_size
                hiddenCurrent = hiddenCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)
                cellCurrent = cellCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)

                # Update states according to the next top beams
                backIndexVector = backIndices[:, :, t].unsqueeze(0)\
                    .unsqueeze(-1).repeat(num_layers, 1, 1, rnnHiddenSize)
                hiddenCurrent = hiddenCurrent.gather(2, backIndexVector)
                cellCurrent = cellCurrent.gather(2, backIndexVector)

                # Restore original shape for next rnn forward
                hiddenCurrent = hiddenCurrent.view(*original_state_size)
                cellCurrent = cellCurrent.view(*original_state_size)
                hiddenStates = (hiddenCurrent, cellCurrent)

            # Detecting endToken to end beams
            aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = t
            if aliveBeams == 0:
                break

        # Backtracking to get final beams
        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        # Keep this on when returning the top beam
        RECOVER_TOP_BEAM_ONLY = True

        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while (tokenIdx >= 0):
            tokens.append(beamTokensTable[:,:,tokenIdx].\
                        gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].\
                        gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLens = tokens.ne(self.endToken).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            # 'tokens' has shape (batchSize, beamSize, maxLen)
            # 'seqLens' has shape (batchSize, beamSize)
            tokens = tokens[:, 0]  # Keep only top beam
            seqLens = seqLens[:, 0]

        return Variable(tokens), Variable(seqLens)

    def get_mcts_samples(self, initStates, beamSize, maxLen, startTokens):

        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        batchSize = initStates[0].size(1)

        startTokenArray = startTokens

        # startTokenArray = Variable(startTokenArray)
        hiddenStates = initStates

        # Inits
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(self.endToken)
        seqLenTable = th.LongTensor(batchSize, beamSize).fill_(0)
        unitMat = th.LongTensor(batchSize, beamSize).fill_(1)
        END_TOKEN_IDX = self.endToken - 1

        for t in range(maxLen):  # Beam expansion till maxLen
            if t == 0:
                # First column of beamTokensTable is generated from <START> token
                emb = self.wordEmbed(startTokenArray)
                emb = emb.unsqueeze(1)
                # emb has shape (batchSize, 1, embedSize)
                output, hiddenStates = self.rnn(emb, hiddenStates)
                # output has shape (batchSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze(1))
                logProbs = self.logSoftmax(scores*self.beta)

                # Additionally, remove <END> token in addition to removing the padding and start token from the first sample
                # to prevent the sampling of an empty sequence.
                logProbs = logProbs[:, 1:-2]

                # scores & logProbs has shape (batchSize, vocabSize)
                probs = torch.exp(logProbs)
                categorical_dist = Categorical(probs)
                samples = [categorical_dist.sample().unsqueeze(1) for _ in range(beamSize)]
                samples = torch.cat(samples, 1)
                samples = samples + 1

                beamTokensTable[:, :, 0] = samples.data

                # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                hiddenStates = [
                    x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                    for x in hiddenStates
                ]
                hiddenStates = [
                    x.view(self.numLayers, -1, self.rnnHiddenSize)
                    for x in hiddenStates
                ]

                # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
            else:
                # Subsequent columns are generated from previous tokens
                emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                # emb has shape (batchSize, beamSize, embedSize)
                output, hiddenStates = self.rnn(emb.view(-1, 1, self.embedSize), hiddenStates)
                # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores*self.beta)
                # Explicitly removing padding token (index 0) and <START> token
                # (index -2) from logProbs so that they are never sampled.
                # This is allows us to keep <START> and padding token in
                # the decoder vocab without any problems in RL sampling.
                logProbsCurrent = torch.cat([logProbsCurrent[:, 1:-2], logProbsCurrent[:, -1:]], 1)
                probs = torch.exp(logProbsCurrent)
                categorical_dist = Categorical(probs)
                samples = [categorical_dist.sample().unsqueeze(1)]
                samples = torch.cat(samples, 1)
                samples = samples.view(batchSize, beamSize)
                samples = samples + 1 # adding one offset as we deleted the padding token.
                # computational graph deliberately broken at this point
                beamTokensTable[:, :, t] = samples.data

            # update lengths here; check if an end token was generated
            unitMat.masked_fill_(beamTokensTable[:,:,t].eq(END_TOKEN_IDX), 0)
            # Add +1 length to all un-ended sequences
            seqLenTable = seqLenTable + unitMat

        return Variable(beamTokensTable), Variable(seqLenTable)


class Decoder_orig(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 startToken,
                 endToken,
                 dropout=0,
                 **kwargs):
        super(Decoder_orig, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        self.startToken = startToken
        self.endToken = endToken
        self.dropout = dropout

        # Modules
        self.rnn = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=self.dropout)
        self.outNet = nn.Linear(self.rnnHiddenSize, self.vocabSize)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.all_log_probs = []
        self.all_mask = []

    def forward(self, encStates, inputSeq):
        '''
        Given encoder states, forward pass an input sequence 'inputSeq' to
        compute its log likelihood under the current decoder RNN state.

        Arguments:
            encStates: (H, C) Tuple of hidden and cell encoder states
            inputSeq: Input sequence for computing log probabilities
            reconstruct_past_ques: Option whether there is a coherrence term
                                   for predicting question on past round

        Output:
            A (batchSize, length, vocabSize) sized tensor of log-probabilities
            obtained from feeding 'inputSeq' to decoder RNN at evert time step

        Note:
            Maximizing the NLL of an input sequence involves feeding as input
            tokens from the GT (ground truth) sequence at every time step and
            maximizing the probability of the next token ("teacher forcing").
            See 'maskedNll' in utils/utilities.py where the log probability of
            the next time step token is indexed out for computing NLL loss.
        '''
        if inputSeq is not None:
            inputSeq = self.wordEmbed(inputSeq)
            outputs, hidden = self.rnn(inputSeq, encStates)
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputSize = outputs.size()
            flatOutputs = outputs.contiguous().view(-1, outputSize[2])
            flatScores = self.outNet(flatOutputs)
            flatLogProbs = self.logSoftmax(flatScores)
            logProbs = flatLogProbs.contiguous().view(outputSize[0], outputSize[1], -1)

        return logProbs, hidden

    def forwardDecode(self,
                      encStates,
                      maxSeqLen=20,
                      inference='sample',
                      beamSize=1, futureReward=True, run_mcts=False, num_beams_mcts = 10):
        '''
        Decode a sequence of tokens given an encoder state, using either
        sampling or greedy inference.

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            maxSeqLen : Maximum length of token sequence to generate
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width

        Notes:
            * This function is not called during SL pre-training
            * Greedy inference is used for evaluation
            * Sampling is used in RL fine-tuning
        '''
        if inference == 'greedy' and beamSize > 1:
            # Use beam search inference when beam size is > 1
            return self.beamSearchDecoder(encStates, beamSize, maxSeqLen)

        # Determine if cuda tensors are being used
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        self.samples = []
        self.mcts_samples = []
        self.mcts_samples_lens = []

        maxLen = maxSeqLen + 1  # Extra <END> token
        batchSize = encStates[0].size(1)
        # Placeholder for filling in tokens at evert time step
        seq = th.LongTensor(batchSize, maxLen + 1)
        seq.fill_(self.endToken)
        seq[:, 0] = self.startToken
        seq = Variable(seq, requires_grad=False)

        # Initial state linked from encStates
        hid = encStates

        sampleLens = th.LongTensor(batchSize).fill_(0)
        # Tensors needed for tracking sampleLens
        unitColumn = th.LongTensor(batchSize).fill_(1)
        mask = th.ByteTensor(seq.size()).fill_(0)

        self.saved_log_probs = []

        # Generating tokens sequentially
        for t in range(maxLen - 1):
            emb = self.wordEmbed(seq[:, t:t + 1])
            # emb has shape  (batch, 1, embedSize)
            output, hid = self.rnn(emb, hid)
            # output has shape (batch, 1, rnnHiddenSize)
            scores = self.outNet(output.squeeze(1))
            logProb = self.logSoftmax(scores)

            # Explicitly removing padding token (index 0) and <START> token
            # (index -2) from logProbs so that they are never sampled.
            # This is allows us to keep <START> and padding token in
            # the decoder vocab without any problems in RL sampling.
            if t > 0:
                logProb = torch.cat([logProb[:, 1:-2], logProb[:, -1:]], 1)
            elif t == 0:
                # Additionally, remove <END> token from the first sample
                # to prevent the sampling of an empty sequence.
                logProb = logProb[:, 1:-2]
            # This also shifts end token index back by 1
            END_TOKEN_IDX = self.endToken - 1

            probs = torch.exp(logProb)

            #MCTS stuff
            if inference == 'sample' and run_mcts:
                # create samples for the hidden state
                startPrefix = seq[:, :t + 1].unsqueeze(1)
                if t != maxLen-1:
                    mcts_samples, mcts_seqLens = self.get_mcts_samples(hid, num_beams_mcts, maxLen - 1 - t, seq[:, t])
                    mcts_samples = torch.cat([startPrefix.repeat(1, num_beams_mcts, 1), mcts_samples], dim=2)
                    # updating lengths as well
                    mcts_seqLens = mcts_seqLens + t + 1
                    self.mcts_samples.append(mcts_samples)
                    self.mcts_samples_lens.append(mcts_seqLens)
                    # randomly select one of the MCTS sample to be the value on the next iteration
                    sample_index = random.randint(0,num_beams_mcts-1)
                    sample = mcts_samples[:,sample_index,t+1]
                    # log probs
                    # sample has ids wrt to the vocab. It has the following guarantees:
                    # start token/padding token is never sampled sample lens deal with the end token
                    # since we have remove the padding token in logProb ,we need to offset the sample by -1
                    sample = sample - 1
                    sample = sample.contiguous()
                    self.saved_log_probs.append(logProb.gather(1, sample.view(-1, 1)))
                    sample = sample.unsqueeze(-1)

            elif inference == 'sample' and not run_mcts:
                categorical_dist = Categorical(probs)
                sample = categorical_dist.sample()
                # Saving log probs for a subsequent reinforce call
                self.saved_log_probs.append(categorical_dist.log_prob(sample))
                sample = sample.unsqueeze(-1)

            elif inference == 'greedy':
                _, sample = torch.max(probs, dim=1, keepdim=True)
            else:
                raise ValueError("Invalid inference type: '{}'".format(inference))

            # Compensating for removed padding token prediction earlier
            sample = sample + 1  # Incrementing all token indices by 1

            self.samples.append(sample)

            seq.data[:, t + 1] = sample.squeeze().data
            # Marking spots where <END> token is generated
            mask[:, t] = sample.squeeze().data.eq(END_TOKEN_IDX)

            # Compensating for shift in <END> token index
            sample.data.masked_fill_(mask[:, t].unsqueeze(1), self.endToken)

        mask[:, maxLen - 1].fill_(1)

        # Computing lengths of generated sequences
        for t in range(maxLen):
            # Zero out the spots where end token is reached
            unitColumn.masked_fill_(mask[:, t], 0)
            # Update mask
            mask[:, t] = unitColumn
            # Add +1 length to all un-ended sequences
            sampleLens = sampleLens + unitColumn

        # Keep mask for later use in RL reward masking
        self.mask = Variable(mask, requires_grad=False)
        if inference == 'sample' and run_mcts:
            # we now have sequence lengths of all the samples. We cant generate MCTS samples for the
            # the last token. For that case, we will just duplicate the sample.
            self.mcts_samples.append(seq[:, :maxLen].unsqueeze(1).repeat(1, num_beams_mcts, 1))
            mcts_seqLens = Variable(sampleLens.unsqueeze(1).repeat(1,1,num_beams_mcts).squeeze(0))
            self.mcts_samples_lens.append(mcts_seqLens)

        # Adding <START> to generated answer lengths for consistency
        sampleLens = sampleLens + 1
        sampleLens = Variable(sampleLens, requires_grad=False)

        startColumn = sample.data.new(sample.size()).fill_(self.startToken)
        startColumn = Variable(startColumn, requires_grad=False)

        # Note that we do not add startColumn to self.samples itself
        # as reinforce is called on self.samples (which needs to be
        # the output of a stochastic function)
        gen_samples = [startColumn] + self.samples
        if futureReward:
            self.all_log_probs.append(self.saved_log_probs)
            self.all_mask.append(self.mask)

        samples = torch.cat(gen_samples, 1)
        return samples, sampleLens

    def evalOptions(self, encStates, options, optionLens, scoringFunction):
        '''
        Forward pass a set of candidate options to get log probabilities

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            options   : (batchSize, numOptions, maxSequenceLength) sized
                        tensor with <START> and <END> tokens

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.

        Output:
            A (batchSize, numOptions) tensor containing the score
            of each option sentence given by the generator
        '''
        batchSize, numOptions, maxLen = options.size()
        optionsFlat = options.contiguous().view(-1, maxLen)

        # Reshaping H, C for each option
        encStates = [x.unsqueeze(2).repeat(1,1,numOptions,1).\
                        view(self.numLayers, -1, self.rnnHiddenSize)
                        for x in encStates]

        logProbs, _ = self.forward(encStates, inputSeq=optionsFlat)
        scores = scoringFunction(logProbs, optionsFlat, returnScores=True)
        return scores.view(batchSize, numOptions)


    def reinforce(self,reward,futureReward=False,mcts=False):
        '''
        Compute loss using REINFORCE on log probabilities of tokens
        sampled from decoder RNN, scaled by input 'reward'.

        Note that an earlier call to forwardDecode must have been
        made in order to have samples for which REINFORCE can be
        applied. These samples are stored in 'self.saved_log_probs'.
        '''
        loss = 0
        # samples = torch.stack(self.samples, 1)
        # sampleLens = self.sampleLens - 1
        if len(self.saved_log_probs) == 0:
            raise RuntimeError("Reinforce called without sampling in Decoder")

        if not futureReward:
            for t, log_prob in enumerate(self.saved_log_probs):

                if not mcts:
                    # reward is a single vector
                    loss += -1 * log_prob * (reward.detach() * (self.mask[:, t].float()))
                else:
                    # reward is a matrix, different token have different returns from the environment
                    loss += -1 * log_prob * (reward[:, t].detach() * (self.mask[:, t].float()))
        else:
            print("Using future reward")
            # print(reward)

            for ind,log_probs in enumerate(self.all_log_probs):

                mask = self.all_mask[ind]
                cur_reward = reward[:,ind]
                # print(cur_reward)

                for t,log_prob in enumerate(log_probs):
                    loss += -1 * log_prob * (cur_reward.detach() * (mask[:, t].float()))

            # resetting the logs
            self.all_mask = []
            self.all_log_probs = []

        return loss

    def beamSearchDecoder(self, initStates, beamSize, maxSeqLen):
        '''
        Beam search for sequence generation
        Arguments:
            initStates - Initial encoder states tuple
            beamSize - Beam Size
            maxSeqLen - Maximum length of sequence to decode
        '''

        # For now, use beam search for evaluation only
        assert self.training == False

        # Determine if cuda tensors are being used
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        LENGTH_NORM = True
        maxLen = maxSeqLen + 1  # Extra <END> token
        batchSize = initStates[0].size(1)

        startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
        backVector = th.LongTensor(beamSize)
        torch.arange(0, beamSize, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batchSize, 1)

        tokenArange = th.LongTensor(self.vocabSize)
        torch.arange(0, self.vocabSize, out=tokenArange)
        tokenArange = Variable(tokenArange)

        startTokenArray = Variable(startTokenArray)
        backVector = Variable(backVector)
        hiddenStates = initStates

        # Inits
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(
            self.endToken)
        beamTokensTable = Variable(beamTokensTable)
        backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
        backIndices = Variable(backIndices)

        aliveVector = beamTokensTable[:, :, 0].eq(self.endToken).unsqueeze(2)

        for t in range(maxLen - 1):  # Beam expansion till maxLen]
            if t == 0:
                # First column of beamTokensTable is generated from <START> token
                emb = self.wordEmbed(startTokenArray)
                # emb has shape (batchSize, 1, embedSize)
                output, hiddenStates = self.rnn(emb, hiddenStates)
                # output has shape (batchSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze(1))
                logProbs = self.logSoftmax(scores)
                # scores & logProbs has shape (batchSize, vocabSize)

                # Find top beamSize logProbs
                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                logProbSums = topLogProbs

                # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                hiddenStates = [
                    x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                    for x in hiddenStates
                ]
                hiddenStates = [
                    x.view(self.numLayers, -1, self.rnnHiddenSize)
                    for x in hiddenStates
                ]
                # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
            else:
                # Subsequent columns are generated from previous tokens
                emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                # emb has shape (batchSize, beamSize, embedSize)
                output, hiddenStates = self.rnn(
                    emb.view(-1, 1, self.embedSize), hiddenStates)
                # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores)
                # logProbs has shape (batchSize*beamSize, vocabSize)
                # NOTE: Padding token has been removed from generator output during
                # sampling (RL fine-tuning). However, the padding token is still
                # present in the generator vocab and needs to be handled in this
                # beam search function. This will be supported in a future release.
                logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                                                       self.vocabSize)

                if LENGTH_NORM:
                    # Add (current log probs / (t+1))
                    logProbs = logProbsCurrent * (aliveVector.float() /
                                                  (t + 1))
                    # Add (previous log probs * (t/t+1) ) <- Mean update
                    coeff_ = aliveVector.eq(0).float() + (
                        aliveVector.float() * t / (t + 1))
                    logProbs += logProbSums.unsqueeze(2) * coeff_
                else:
                    # Add currrent token logProbs for alive beams only
                    logProbs = logProbsCurrent * (aliveVector.float())
                    # Add previous logProbSums upto t-1
                    logProbs += logProbSums.unsqueeze(2)

                # Masking out along |V| dimension those sequence logProbs
                # which correspond to ended beams so as to only compare
                # one copy when sorting logProbs
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                mask_[:, :,
                      0] = 0  # Zeroing all except first row for ended beams
                minus_infinity_ = torch.min(logProbs).item()
                logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                logProbs = logProbs.view(batchSize, -1)
                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).\
                                repeat(batchSize,beamSize,1)
                tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                tokensArray = tokensArray.view(batchSize, -1)
                backIndexArray = backVector.unsqueeze(2).\
                                repeat(1,1,self.vocabSize).view(batchSize,-1)

                topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                logProbSums = topLogProbs
                beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                backIndices[:, :, t] = backIndexArray.gather(1, topIdx)

                # Update corresponding hidden and cell states for next time step
                hiddenCurrent, cellCurrent = hiddenStates

                # Reshape to get explicit beamSize dim
                original_state_size = hiddenCurrent.size()
                num_layers, _, rnnHiddenSize = original_state_size
                hiddenCurrent = hiddenCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)
                cellCurrent = cellCurrent.view(
                    num_layers, batchSize, beamSize, rnnHiddenSize)

                # Update states according to the next top beams
                backIndexVector = backIndices[:, :, t].unsqueeze(0)\
                    .unsqueeze(-1).repeat(num_layers, 1, 1, rnnHiddenSize)
                hiddenCurrent = hiddenCurrent.gather(2, backIndexVector)
                cellCurrent = cellCurrent.gather(2, backIndexVector)

                # Restore original shape for next rnn forward
                hiddenCurrent = hiddenCurrent.view(*original_state_size)
                cellCurrent = cellCurrent.view(*original_state_size)
                hiddenStates = (hiddenCurrent, cellCurrent)

            # Detecting endToken to end beams
            aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = t
            if aliveBeams == 0:
                break

        # Backtracking to get final beams
        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        # Keep this on when returning the top beam
        RECOVER_TOP_BEAM_ONLY = True

        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while (tokenIdx >= 0):
            tokens.append(beamTokensTable[:,:,tokenIdx].\
                        gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].\
                        gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLens = tokens.ne(self.endToken).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            # 'tokens' has shape (batchSize, beamSize, maxLen)
            # 'seqLens' has shape (batchSize, beamSize)
            tokens = tokens[:, 0]  # Keep only top beam
            seqLens = seqLens[:, 0]

        return Variable(tokens), Variable(seqLens)

    def get_mcts_samples(self, initStates, beamSize, maxLen, startTokens):

        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        batchSize = initStates[0].size(1)

        startTokenArray = startTokens

        # startTokenArray = Variable(startTokenArray)
        hiddenStates = initStates

        # Inits
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(self.endToken)
        seqLenTable = th.LongTensor(batchSize, beamSize).fill_(0)
        unitMat = th.LongTensor(batchSize, beamSize).fill_(1)
        END_TOKEN_IDX = self.endToken - 1

        for t in range(maxLen):  # Beam expansion till maxLen
            if t == 0:
                # First column of beamTokensTable is generated from <START> token
                emb = self.wordEmbed(startTokenArray)
                emb = emb.unsqueeze(1)
                # emb has shape (batchSize, 1, embedSize)
                output, hiddenStates = self.rnn(emb, hiddenStates)
                # output has shape (batchSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze(1))
                logProbs = self.logSoftmax(scores)

                # Additionally, remove <END> token in addition to removing the padding and start token from the first sample
                # to prevent the sampling of an empty sequence.
                logProbs = logProbs[:, 1:-2]

                # scores & logProbs has shape (batchSize, vocabSize)
                probs = torch.exp(logProbs)
                categorical_dist = Categorical(probs)
                samples = [categorical_dist.sample().unsqueeze(1) for _ in range(beamSize)]
                samples = torch.cat(samples, 1)
                samples = samples + 1

                beamTokensTable[:, :, 0] = samples.data

                # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                hiddenStates = [
                    x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                    for x in hiddenStates
                ]
                hiddenStates = [
                    x.view(self.numLayers, -1, self.rnnHiddenSize)
                    for x in hiddenStates
                ]

                # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
            else:
                # Subsequent columns are generated from previous tokens
                emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                # emb has shape (batchSize, beamSize, embedSize)
                output, hiddenStates = self.rnn(emb.view(-1, 1, self.embedSize), hiddenStates)
                # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                scores = self.outNet(output.squeeze())
                logProbsCurrent = self.logSoftmax(scores)
                # Explicitly removing padding token (index 0) and <START> token
                # (index -2) from logProbs so that they are never sampled.
                # This is allows us to keep <START> and padding token in
                # the decoder vocab without any problems in RL sampling.
                logProbsCurrent = torch.cat([logProbsCurrent[:, 1:-2], logProbsCurrent[:, -1:]], 1)
                probs = torch.exp(logProbsCurrent)
                categorical_dist = Categorical(probs)
                samples = [categorical_dist.sample().unsqueeze(1)]
                samples = torch.cat(samples, 1)
                samples = samples.view(batchSize, beamSize)
                samples = samples + 1 # adding one offset as we deleted the padding token.
                # computational graph deliberately broken at this point
                beamTokensTable[:, :, t] = samples.data

            # update lengths here; check if an end token was generated
            unitMat.masked_fill_(beamTokensTable[:,:,t].eq(END_TOKEN_IDX), 0)
            # Add +1 length to all un-ended sequences
            seqLenTable = seqLenTable + unitMat

        return Variable(beamTokensTable), Variable(seqLenTable)