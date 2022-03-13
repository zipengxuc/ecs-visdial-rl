import torch
import torch.nn as nn
from visdial.models.agent import Agent
import visdial.models.encoders.hre_att as hre_att
import visdial.models.decoders.gen as gen_dec
from utils import utilities as utils


class Answerer(Agent):
    # initialize
    def __init__(self, encoderParam, decoderParam, verbose=1,multiGPU=False):
        '''
            A-Bot Model

            Uses an encoder network for input sequences (questions, answers and
            history) and a decoder network for generating a response (answer).
        '''
        super(Answerer, self).__init__()
        self.encType = encoderParam['type']
        self.decType = decoderParam['type']

        # Encoder
        if verbose:
            print('Encoder: ' + self.encType)
            print('Decoder: ' + self.decType)
        if '100g' in self.encType:
            self.encoder = hre_att.Encoder(**encoderParam)
            if multiGPU:
                self.encoder = nn.DataParallel(self.encoder)
        else:
            raise Exception('Unknown encoder {}'.format(self.encType))

        # Decoder
        if 'gen' == self.decType:
            self.decoder = gen_dec.Decoder(**decoderParam)
            if multiGPU:
                self.decoder = nn.DataParallel(self.decoder)
        else:
            raise Exception('Unkown decoder {}'.format(self.decType))

        # Share word embedding parameters between encoder and decoder
        if multiGPU:
            self.decoder.module.wordEmbed = self.encoder.module.wordEmbed
        else:
            self.decoder.wordEmbed = self.encoder.wordEmbed

        # Initialize weights
        utils.initializeWeights(self.encoder)
        utils.initializeWeights(self.decoder)
        self.multiGPU = multiGPU
        self.reset()

    def reset(self):
        '''Delete dialog history.'''
        self.caption = None
        self.answers = []
        self.answer_mask = []
        if self.multiGPU:
            self.encoder.module.reset()
        else:
            self.encoder.reset()

    def observe(self, round, ans=None, caption=None, ansMask=None, **kwargs):
        '''
        Update Q-Bot percepts. See self.encoder.observe() in the corresponding
        encoder class definition (hre).
        '''
        if caption is not None:
            assert round == -1, "Round number should be -1 when observing"\
                                " caption, got %d instead"
            self.caption = caption
        if ans is not None:
            assert round == len(self.answers),\
                "Round number does not match number of answers observed"
            self.answers.append(ans)
            if ansMask is not None:
                self.answer_mask.append(ansMask)
        if self.multiGPU:
            self.encoder.module.observe(round, ans=ans, caption=caption, **kwargs)
        else:
            self.encoder.observe(round, ans=ans, caption=caption, **kwargs)

    def forward(self):
        '''
        Forward pass the last observed answer to compute its log
        likelihood under the current decoder RNN state.
        '''
        if 'com' in self.encType or '3' in self.encType:
            att, img, feat, encStates = self.encoder()
            _, encStates = self.decoder(encStates, inputSeq=feat.unsqueeze(1), seq=False)

        else:
            encStates = self.encoder()
        if len(self.answers) > 0:
            decIn = self.answers[-1]
            if len(self.answer_mask):
                decMask = self.answer_mask[-1]
        elif self.caption is not None:
            decIn = self.caption
        else:
            raise Exception('Must provide an input sequence')

        if 'gen_att' in self.decType:
            if len(self.answer_mask):
                logProbs, _ = self.decoder(encStates, inputSeq=decIn, attV=att, img=img, ansMask=decMask)
            else:
                logProbs, _ = self.decoder(encStates, inputSeq=decIn, attV=att, img=img)
        else:
            logProbs, _ = self.decoder(encStates, inputSeq=decIn)
        return logProbs

    def forwardDecode(self, inference='sample',futureReward=False, beamSize=1, maxSeqLen=20,run_mcts=False):
        '''
        Decode a sequence (answer) using either sampling or greedy inference.
        An answer is decoded given the current state (dialog history). This
        can be called at every round after a question is observed.

        Arguments:
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
            maxSeqLen : Maximum length of token sequence to generate
        '''
        if 'com' in self.encType or '3' in self.encType:
            att, img, feat, encStates = self.encoder()
            _, encStates = self.decoder(encStates, inputSeq=feat.unsqueeze(1), seq=False)

        else:
            encStates = self.encoder()
        if 'gen_att' in self.decType:
            if self.multiGPU:
                answers, ansLens = self.decoder.module.forwardDecode(
                    encStates, attV=att, img=img,
                    maxSeqLen=maxSeqLen,
                    inference=inference,futureReward=futureReward,
                    beamSize=beamSize,run_mcts=run_mcts)
            else:
                answers, ansLens = self.decoder.forwardDecode(
                    encStates, attV=att, img=img,
                    maxSeqLen=maxSeqLen,
                    inference=inference,futureReward=futureReward,
                    beamSize=beamSize,run_mcts=run_mcts)
        else:
            answers, ansLens = self.decoder.forwardDecode(
                encStates,
                maxSeqLen=maxSeqLen,
                inference=inference,futureReward=futureReward,
                beamSize=beamSize,run_mcts=run_mcts)
        return answers, ansLens

    def evalOptions(self, options, optionLens, scoringFunction):
        '''
        Given the current state (question and conversation history), evaluate
        a set of candidate answers to the question.

        Output:
            Log probabilities of candidate options.
        '''
        if 'com' in self.encType or '3' in self.encType:
            att, img, feat, states = self.encoder()
            _, states = self.decoder(states, inputSeq=feat.unsqueeze(1), seq=False)

        else:
            states = self.encoder()
        if 'gen_att' in self.decType:
            if self.multiGPU:
                return self.decoder.module.evalOptions(states, options, optionLens,
                                                scoringFunction, attV=att, img=img)
            else:
                return self.decoder.evalOptions(states, options, optionLens,
                                                scoringFunction, attV=att, img=img)
        else:
            return self.decoder.evalOptions(states, options, optionLens,
                                            scoringFunction)

    def reinforce(self, reward, futureReward=False, mcts=False):
        # Propogate reinforce function call to decoder
        if self.multiGPU:
            return self.decoder.module.reinforce(reward,futureReward=futureReward,mcts=mcts)
        else:
            return self.decoder.reinforce(reward,futureReward=futureReward,mcts=mcts)
