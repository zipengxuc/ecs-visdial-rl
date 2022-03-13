import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def initializeWeights(root, itype='xavier'):
    assert itype == 'xavier', 'Only Xavier initialization supported'

    for module in root.modules():
        # Initialize weights
        name = type(module).__name__
        # If linear or embedding
        if name in ['Embedding', 'Linear']:
            fanIn = module.weight.data.size(0)
            fanOut = module.weight.data.size(1)

            factor = math.sqrt(2.0 / (fanIn + fanOut))
            weight = torch.randn(fanIn, fanOut) * factor
            module.weight.data.copy_(weight)
        elif 'LSTM' in name:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    param.data.fill_(0.0)
                else:
                    fanIn = param.size(0)
                    fanOut = param.size(1)

                    factor = math.sqrt(2.0 / (fanIn + fanOut))
                    weight = torch.randn(fanIn, fanOut) * factor
                    param.data.copy_(weight)
        else:
            pass

        # Check for bias and reset
        if hasattr(module, 'bias') and type(module.bias) != bool:
            module.bias.data.fill_(0.0)


def saveModel(model, optimizer, saveFile, params):
    if params['multiGPU']:
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }, saveFile)
    else:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
        }, saveFile)


def loadModelB(params, agent='abot', overwrite=False, multiGPU=False):
    if overwrite is False:
        params = params.copy()
    loadedParams = {}
    # should be everything used in encoderParam, decoderParam below
    if agent == 'abot':
        encoderOptions = [
            'encoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'a_numLayers',
            'useHistory', 'useIm', 'imgEmbedSize', 'imgFeatureSize', 'numRounds',
            'dropout'
        ]
        decoderOptions = [
            'decoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'a_numLayers', 'beta',
            'dropout'
        ]
    elif agent == 'qbot':
        encoderOptions = [
            'encoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'q_numLayers',
            'useHistory', 'useIm', 'imgEmbedSize', 'imgFeatureSize', 'numRounds',
            'dropout'
        ]
        decoderOptions = [
            'decoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'q_numLayers',
            'dropout'
        ]
    modelOptions = encoderOptions + decoderOptions

    mdict = None
    gpuFlag = params['useGPU']
    continueFlag = params['continue']
    numEpochs = params['numEpochs']
    startArg = 'startFrom' if agent == 'abot' else 'qstartFrom'
    if continueFlag:
        assert params[startArg], "Can't continue training without a \
                                    checkpoint"

    # load a model from disk if it is given
    if params[startArg]:
        print('Loading model (weights and config) from {}'.format(
            params[startArg]))

        if gpuFlag:
            mdict = torch.load(params[startArg])
        else:
            mdict = torch.load(params[startArg],
                               map_location=lambda storage, location: storage)

        # Model options is a union of standard model options defined
        # above and parameters loaded from checkpoint
        modelOptions = list(set(modelOptions).union(set(mdict['params'])))
        exps = ['q_numLayers', 'a_numLayers']
        for opt in modelOptions:
            if opt in exps:
                continue
            elif opt not in params:
                # Loading options from a checkpoint which are
                # necessary for continuing training, but are
                # not present in original parameter list.
                if continueFlag:
                    print("Loaded option '%s' from checkpoint" % opt)
                    params[opt] = mdict['params'][opt]
                    loadedParams[opt] = mdict['params'][opt]
            elif params[opt] != mdict['params'][opt]:
                # When continuing training from a checkpoint, overwriting
                # parameters loaded from checkpoint is okay.
                if continueFlag:
                    print("Overwriting param '%s'" % str(opt))
                    params[opt] = mdict['params'][opt]

        params['continue'] = continueFlag
        params['numEpochs'] = numEpochs
        params['useGPU'] = gpuFlag

        # if params['continue']:
        #     assert 'ckpt_lRate' in params, "Checkpoint does not have\
        #         info for restoring learning rate and optimizer."

    # assert False, "STOP right there, criminal scum!"

    # Initialize model class
    encoderParam = {k: params[k] for k in encoderOptions}
    decoderParam = {k: params[k] for k in decoderOptions}

    encoderParam['startToken'] = encoderParam['vocabSize'] - 2
    encoderParam['endToken'] = encoderParam['vocabSize'] - 1
    decoderParam['startToken'] = decoderParam['vocabSize'] - 2
    decoderParam['endToken'] = decoderParam['vocabSize'] - 1

    if agent == 'abot':
        encoderParam['type'] = params['encoder']
        decoderParam['type'] = params['decoder']
        encoderParam['numLayers'] = params['a_numLayers']
        decoderParam['numLayers'] = params['a_numLayers']
        encoderParam['isAnswerer'] = True
        from visdial.models.answerer import Answerer
        model = Answerer(encoderParam, decoderParam, multiGPU=multiGPU)

    elif agent == 'qbot':
        encoderParam['type'] = params['qencoder']
        decoderParam['type'] = params['qdecoder']
        encoderParam['numLayers'] = params['q_numLayers']
        decoderParam['numLayers'] = params['q_numLayers']
        encoderParam['isAnswerer'] = False
        encoderParam['useIm'] = False
        from visdial.models.questioner import Questioner
        model = Questioner(
            encoderParam,
            decoderParam,
            imgFeatureSize=4096, multiGPU=multiGPU)

    if params['useGPU']:
        model.cuda()

    for p in model.encoder.parameters():
        p.register_hook(clampGrad)
    for p in model.decoder.parameters():
        p.register_hook(clampGrad)
    # NOTE: model.parameters() should be used here, otherwise immediate
    # child modules in model will not have gradient clamping

    # copy parameters if specified
    if mdict:
        model.load_state_dict(mdict['model'])
        optim_state = mdict['optimizer']
    else:
        optim_state = None
    return model, loadedParams, optim_state


def loadModel(params, agent='abot', overwrite=False, multiGPU=False):
    if overwrite is False:
        params = params.copy()
    loadedParams = {}
    # should be everything used in encoderParam, decoderParam below
    if agent == 'abot':
        encoderOptions = [
            'encoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'a_numLayers',
            'useHistory', 'useIm', 'imgEmbedSize', 'imgFeatureSize', 'numRounds',
            'dropout'
        ]
        decoderOptions = [
            'decoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'a_numLayers', 'beta',
            'dropout'
        ]
    elif agent == 'qbot':
        encoderOptions = [
            'encoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'q_numLayers',
            'useHistory', 'useIm', 'imgEmbedSize', 'imgFeatureSize', 'numRounds',
            'dropout'
        ]
        decoderOptions = [
            'decoder', 'vocabSize', 'embedSize', 'rnnHiddenSize', 'q_numLayers',
            'dropout'
        ]
    modelOptions = encoderOptions + decoderOptions

    mdict = None
    gpuFlag = params['useGPU']
    continueFlag = params['continue']
    numEpochs = params['numEpochs']
    startArg = 'startFrom' if agent == 'abot' else 'qstartFrom'
    if continueFlag:
        assert params[startArg], "Can't continue training without a \
                                    checkpoint"

    # load a model from disk if it is given
    if params[startArg]:
        print('Loading model (weights and config) from {}'.format(
            params[startArg]))

        if gpuFlag:
            mdict = torch.load(params[startArg])
        else:
            mdict = torch.load(params[startArg],
                map_location=lambda storage, location: storage)

        # Model options is a union of standard model options defined
        # above and parameters loaded from checkpoint
        modelOptions = list(set(modelOptions).union(set(mdict['params'])))
        modelOptions.append('ckpt_iterid')
        modelOptions.append('a_ckpt_lRate')
        modelOptions.append('q_ckpt_lRate')
        exps = ['q_numLayers', 'a_numLayers']
        for opt in modelOptions:
            if opt in exps:
                continue
            elif opt not in mdict['params']:
                print(opt)
                print("not in!")
                continue
            elif opt not in params:
                # Loading options from a checkpoint which are
                # necessary for continuing training, but are
                # not present in original parameter list.
                if continueFlag:
                    print("Loaded option '%s' from checkpoint" % opt)
                    params[opt] = mdict['params'][opt]
                    loadedParams[opt] = mdict['params'][opt]
            elif params[opt] != mdict['params'][opt]:
                # When continuing training from a checkpoint, overwriting
                # parameters loaded from checkpoint is okay.
                if continueFlag:
                    print("Overwriting param '%s'" % str(opt))
                    params[opt] = mdict['params'][opt]

        params['continue'] = continueFlag
        params['numEpochs'] = numEpochs
        params['useGPU'] = gpuFlag

        if params['continue']:
            assert 'ckpt_iterid' in params, "Checkpoint does not have\
                info for restoring learning rate and optimizer."

    # assert False, "STOP right there, criminal scum!"

    # Initialize model class
    encoderParam = {k: params[k] for k in encoderOptions}
    decoderParam = {k: params[k] for k in decoderOptions}

    encoderParam['startToken'] = encoderParam['vocabSize'] - 2
    encoderParam['endToken'] = encoderParam['vocabSize'] - 1
    decoderParam['startToken'] = decoderParam['vocabSize'] - 2
    decoderParam['endToken'] = decoderParam['vocabSize'] - 1

    if agent == 'abot':
        encoderParam['type'] = params['encoder']
        decoderParam['type'] = params['decoder']
        encoderParam['numLayers'] = params['a_numLayers']
        decoderParam['numLayers'] = params['a_numLayers']
        encoderParam['isAnswerer'] = True
        from visdial.models.answerer import Answerer
        model = Answerer(encoderParam, decoderParam,multiGPU=multiGPU)

    elif agent == 'qbot':
        encoderParam['type'] = params['qencoder']
        decoderParam['type'] = params['qdecoder']
        encoderParam['numLayers'] = params['q_numLayers']
        decoderParam['numLayers'] = params['q_numLayers']
        encoderParam['isAnswerer'] = False
        encoderParam['useIm'] = False
        from visdial.models.questioner import Questioner
        model = Questioner(
            encoderParam,
            decoderParam,
            imgFeatureSize=encoderParam['imgFeatureSize'], multiGPU=multiGPU)

    if params['useGPU']:
        model.cuda()

    for p in model.encoder.parameters():
        p.register_hook(clampGrad)
    for p in model.decoder.parameters():
        p.register_hook(clampGrad)
    # NOTE: model.parameters() should be used here, otherwise immediate
    # child modules in model will not have gradient clamping

    # copy parameters if specified
    if mdict:
        model.load_state_dict(mdict['model'])
        optim_state = mdict['optimizer']
    else:
        optim_state = None
    return model, loadedParams, optim_state


def clampGrad(grad, limit=5.0):
    '''
    Gradient clip by value
    '''
    grad.data.clamp_(min=-limit, max=limit)
    return grad


def getSortedOrder(lens):
    sortedLen, fwdOrder = torch.sort(
        lens.contiguous().view(-1), dim=0, descending=True)
    _, bwdOrder = torch.sort(fwdOrder)
    if isinstance(sortedLen, Variable):
        sortedLen = sortedLen.data
    sortedLen = sortedLen.cpu().numpy().tolist()
    return sortedLen, fwdOrder, bwdOrder


def dynamicRNN(rnnModel,
               seqInput,
               seqLens,
               initialState=None,
               returnStates=False):
    '''
    Inputs:
        rnnModel     : Any torch.nn RNN model
        seqInput     : (batchSize, maxSequenceLength, embedSize)
                        Input sequence tensor (padded) for RNN model
        seqLens      : batchSize length torch.LongTensor or numpy array
        initialState : Initial (hidden, cell) states of RNN

    Output:
        A single tensor of shape (batchSize, rnnHiddenSize) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence. If returnStates is True, also return a tuple of hidden
        and cell states at every layer of size (num_layers, batchSize,
        rnnHiddenSize)
    '''
    sortedLen, fwdOrder, bwdOrder = getSortedOrder(seqLens)
    sortedSeqInput = seqInput.index_select(dim=0, index=fwdOrder)
    packedSeqInput = pack_padded_sequence(
        sortedSeqInput, lengths=sortedLen, batch_first=True)

    if initialState is not None:
        hx = initialState
        sortedHx = [x.index_select(dim=1, index=fwdOrder) for x in hx]
        assert hx[0].size(0) == rnnModel.num_layers  # Matching num_layers
    else:
        hx = None

    rnnModel.flatten_parameters()
    _, (h_n, c_n) = rnnModel(packedSeqInput, hx)

    rnn_output = h_n[-1].index_select(dim=0, index=bwdOrder)

    if returnStates:
        h_n = h_n.index_select(dim=1, index=bwdOrder)
        c_n = c_n.index_select(dim=1, index=bwdOrder)
        return rnn_output, (h_n, c_n)
    else:
        return rnn_output


def maskedNll(seq, gtSeq, returnScores=False):
    '''
    Compute the NLL loss of ground truth (target) sentence given the
    model. Assumes that gtSeq has <START> and <END> token surrounding
    every sequence and gtSeq is left aligned (i.e. right padded)

    S: <START>, E: <END>, W: word token, 0: padding token, P(*): logProb

        gtSeq:
            [ S     W1    W2  E   0   0]
        Teacher forced logProbs (seq):
            [P(W1) P(W2) P(E) -   -   -]
        Required gtSeq (target):
            [  W1    W2    E  0   0   0]
        Mask (non-zero tokens in target):
            [  1     1     1  0   0   0]
    '''
    # Shifting gtSeq 1 token left to remove <START>
    padColumn = gtSeq.data.new(gtSeq.size(0), 1).fill_(0)
    padColumn = Variable(padColumn)
    target = torch.cat([gtSeq, padColumn], dim=1)[:, 1:]

    # Generate a mask of non-padding (non-zero) tokens
    mask = target.data.gt(0)
    loss = 0
    if isinstance(gtSeq, Variable):
        mask = Variable(mask, volatile=gtSeq.volatile)
    assert isinstance(target, Variable)
    gtLogProbs = torch.gather(seq, 2, target.unsqueeze(2)).squeeze(2)
    # Mean sentence probs:
    # gtLogProbs = gtLogProbs/(mask.float().sum(1).view(-1,1))
    if returnScores:
        return (gtLogProbs * (mask.float())).sum(1)
    maskedLL = torch.masked_select(gtLogProbs, mask)
    nll_loss = -torch.sum(maskedLL) / torch.sum(mask)
    return nll_loss


def length_to_mask(lengths, max_len):
    batchSize = lengths.size(0)
    ref_q = torch.arange(max_len).unsqueeze(0).repeat(batchSize, 1).cuda()
    mask = ref_q < lengths.unsqueeze(1)
    return mask.cuda()


def concatPaddedSequences(seq1, seqLens1, seq2, seqLens2, padding='right'):
    '''
    Concates two input sequences of shape (batchSize, seqLength). The
    corresponding lengths tensor is of shape (batchSize). Padding sense
    of input sequences needs to be specified as 'right' or 'left'

    Args:
        seq1, seqLens1 : First sequence tokens and length
        seq2, seqLens2 : Second sequence tokens and length
        padding        : Padding sense of input sequences - either
                         'right' or 'left'

    ref_q = torch.arange(questions.size(-1)).unsqueeze(0).repeat(questions.size(0), 1)
    mask = ref_q <= ques_len.unsqueeze(1)
    merged[mask] = questions[mask]
    '''

    # concat_list = []
    # cat_seq = torch.cat([seq1, seq2], dim=1)
    maxLen1 = seq1.size(1)
    maxLen2 = seq2.size(1)
    maxCatLen = maxLen2 + maxLen1
    batchSize = seq1.size(0)
    newSize = torch.Size([batchSize, maxCatLen])
    merged = torch.LongTensor(newSize).fill_(0).cuda()

    if padding == 'right':
        r1 = length_to_mask(seqLens1 + seqLens2, merged.size(-1))
        r2 = length_to_mask(seqLens1, merged.size(-1))
        mask1 = r1 ^ r2
        mask2 = length_to_mask(seqLens2, maxLen2)
        merged[:, :maxLen1] = seq1

        merged[mask1] = seq2[mask2]
    else:
        raise RuntimeError("pad only in right")

    return merged
