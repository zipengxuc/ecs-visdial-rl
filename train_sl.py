import os
import gc
import random
import pprint
from six.moves import range
import time
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.rank_answerer import rankABot
from eval_utils.rank_questioner import rankQBot
from utils import utilities as utils
# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------
torch.backends.cudnn.enabled = False
# Read the command line options
params = options.readCommandLine()

params["cocoDir"] = './data/coco'
params["cocoInfo"] = './data/visdial/chat_processed_params.json'

# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
splits = ['train', 'val']

dataset = VisDialDataset(params, splits)

# Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
# os.makedirs('checkpoints', exist_ok=True)
# os.mkdir(params['savePath'])

# Loading Modules
parameters = []
aBot = None
qBot = None

print("building model... %f" % time.time())

# Loading Q-Bot
if params['trainMode'] is 'sl-qbot':
    qBot, loadedParams, optim_state = utils.loadModel(params, 'qbot', multiGPU=params['multiGPU'])
    for key in loadedParams:
        params[key] = loadedParams[key]
    # Filtering parameters which require a gradient update
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))

# Loading A-Bot
if params['trainMode'] is 'sl-abot':
    aBot, loadedParams, optim_state = utils.loadModel(params, 'abot', multiGPU=params['multiGPU'])
    for key in loadedParams:
        params[key] = loadedParams[key]
    parameters.extend(aBot.parameters())

print("finished building model! %f" % time.time())

# Setup pytorch dataloader
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=False,
    num_workers=params['numWorkers'],
    drop_last=True,
    collate_fn=dataset.collate_fn,
    pin_memory=False)

print("finished loading data! %f" % time.time())

pprint.pprint(params)

# Setup optimizer
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['learningRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0

optimizer = optim.Adam(parameters, lr=lRate, betas=(0.8, 0.999))
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    optimizer.load_state_dict(optim_state)
runningLoss = None

mse_criterion = nn.MSELoss(reduce=False)

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------

def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch

start_t = timer()

outNet = nn.Linear(params["rnnHiddenSize"], params["vocabSize"])
logSoftmax = nn.LogSoftmax(dim=1)

if params["useGPU"]:
    outNet.cuda()
    logSoftmax.cuda()

baseline = 0

print("start training now!")

for epochId, idx, batch in batch_iter(dataloader):
    # Keeping track of iterId and epoch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epoch = iterId // numIterPerEpoch
    gc.collect()
    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
            else v for key, v in batch.items()}

    image = Variable(batch['img_feat'], requires_grad=False)
    caption = Variable(batch['cap'], requires_grad=False)
    captionLens = Variable(batch['cap_len'], requires_grad=False)
    gtQuestions = Variable(batch['ques'], requires_grad=False)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
    gtAnswers = Variable(batch['ans'], requires_grad=False)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)
    options = Variable(batch['opt'], requires_grad=False)
    optionLens = Variable(batch['opt_len'], requires_grad=False)
    gtAnsId = Variable(batch['ans_id'], requires_grad=False)

    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    featLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']

    # Setting training modes for both bots and observing captions, images where needed
    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)

    # Q-Bot image feature regression ('guessing') only occurs if Q-Bot is present
    if params['trainMode'] is 'sl-qbot':
        initialGuess = qBot.predictImage()
        prevFeatDist = mse_criterion(initialGuess, image)
        featLoss += torch.mean(prevFeatDist)
        prevFeatDist = torch.mean(prevFeatDist,1)

    past_dialog_hidden = None
    cur_dialog_hidden = None

    # calculate the mean reward value for this batch. This will be used to update baseline.
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:
            sl-abot :
                Supervised pre-training of A-Bot model using cross
                entropy loss with ground truth answers
            sl-qbot :
                Supervised pre-training of Q-Bot model using cross
                entropy loss with ground truth questions for the
                dialog model and mean squared error loss for image
                feature regression (i.e. image prediction)
            rl-full-QAf :
                RL-finetuning of A-Bot and Q-Bot in a cooperative
                setting where the common reward is the difference
                in mean squared error between the current and
                previous round of Q-Bot's image prediction.
                Annealing: In order to ease in the RL objective,
                fine-tuning starts with first N-1 rounds of SL
                objective and last round of RL objective - the
                number of RL rounds are increased by 1 after
                every epoch until only RL objective is used for
                all rounds of dialog.
        '''
        factRNN = None
        dialogRNN = None
        dialogState = None

        # Tracking components which require a forward pass
        # A-Bot dialog model
        forwardABot = params['trainMode'] == 'sl-abot'
        # Q-Bot dialog model
        forwardQBot = params['trainMode'] == 'sl-qbot'
        # Q-Bot feature regression network
        forwardFeatNet = forwardQBot

        # Answerer Forward Pass
        if forwardABot:
            # Observe Ground Truth (GT) question
            aBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            # Observe GT answer for teacher forcing
            aBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])
            ansLogProbs = aBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Answers
            aBotLoss += utils.maskedNll(ansLogProbs,
                                        gtAnswers[:, round].contiguous())

        # Questioner Forward Pass (dialog model)
        if forwardQBot:
            # Observe GT question for teacher forcing
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            quesLogProbs = qBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Questions
            qBotLoss += utils.maskedNll(quesLogProbs,
                                        gtQuestions[:, round].contiguous())
            # Observe GT answer for updating dialog history
            qBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog. This can be set to 10 if needed.
        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an image prediction after each round
            predFeatures = qBot.predictImage()
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist)
            featLoss += featDist

    # Loss coefficients
    featLoss = featLoss * params['featLossCoeff']
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds
    aBotLoss = (params['CELossCoeff'] * aBotLoss)  # / numRounds
    featLoss = featLoss / numRounds  # / (numRounds+1)


    loss = qBotLoss + aBotLoss + featLoss
    loss.backward()
    # _ = nn.utils.clip_grad_norm_(parameters, 5)
    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.item()
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.item()

    # Decay learning rate for Q-Bot
    if forwardQBot:
        if lRate > params['minLRate'] and iterId % params['decayCir'] == 0:
            for gId, group in enumerate(optimizer.param_groups):
                optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
            lRate *= params['lrDecayRate']

    # Print every now and then
    if iterId % 100 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.item(), lRate
        ]
        start_t = end_t
        print(printFormat % tuple(printInfo))

    # Evaluate every epoch
    if iterId % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID
        curEpoch = float(iterId) / numIterPerEpoch

        # lr decay per epoch
        if forwardABot:
            if lRate > params['minLRate']:
                if int(curEpoch) != 0 and int(curEpoch) % 10 == 0:
                    lRate = lRate * params["lrDecayRate"]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lRate
                epochId = (1.0 * iterId / numIterPerEpoch) + 1

        # Set eval mode
        if aBot:
            aBot.eval()
        if qBot:
            qBot.eval()

        print('Performing validation...')
        if aBot and 'ques' in batch:
            print("aBot Validation:")

            # NOTE: A-Bot validation is slow, so adjust exampleLimit as needed
            with torch.no_grad():
                rankMetrics = rankABot(
                    aBot,
                    dataset,
                    'val',
                    scoringFunction=utils.maskedNll,
                    exampleLimit=None,useNDCG=params["useNDCG"])

            for metric, value in rankMetrics.items():
                print(metric, value)

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                print("val CE", logProbsMean)

                if params['trainMode'] == 'sl-abot':
                    valLoss = logProbsMean

        if qBot:
            print("qBot Validation:")
            with torch.no_grad():
                rankMetrics, roundMetrics = rankQBot(qBot, dataset, 'val')

            for metric, value in rankMetrics.items():
                print(metric, value)

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                print("val CE", logProbsMean)

            if 'featLossMean' in rankMetrics:
                featLossMean = params['featLossCoeff'] * (
                    rankMetrics['featLossMean'])

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean
                    print("valLoss", valLoss)

    # Save the model after every epoch
    if iterId % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate

        if aBot:
            saveFile = os.path.join(params['savePath'],
                                    'abot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, optimizer, saveFile, params)
        if qBot:
            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)

        print("Saving visdom env to disk: {}".format(params["visdomEnv"]))
