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
from eval_utils.rank_questioner import rankQABots
from utils import utilities as utils
# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------
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
q_parameters = []
a_parameters = []
aBot = None
qBot = None

print("building model... %f" % time.time())

# Loading Q-Bot
qBot, q_loadedParams, q_optim_state = utils.loadModel(params, 'qbot')
for key in q_loadedParams:
    params[key] = q_loadedParams[key]

if params['trainMode'] == 'rl-full-QAf' and params['freezeQFeatNet']:
    qBot.freezeFeatNet()
# Filtering parameters which require a gradient update
q_parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))

# Loading A-Bot
aBot, a_loadedParams, a_optim_state = utils.loadModel(params, 'abot')
for key in a_loadedParams:
    params[key] = a_loadedParams[key]
a_parameters.extend(aBot.parameters())

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
    startIterID = q_loadedParams['ckpt_iterid'] + 1  # Iteration ID
    a_lRate = params['a_learningRate']
    q_lRate = params['q_learningRate']
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    a_lRate = params['a_learningRate']
    q_lRate = params['q_learningRate']
    startIterID = 0

q_optimizer = optim.Adam(q_parameters, lr=q_lRate)
a_optimizer = optim.Adam(a_parameters, lr=a_lRate)
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    q_optimizer.load_state_dict(q_optim_state)
    a_optimizer.load_state_dict(a_optim_state)
runningLoss = None

mse_criterion = nn.MSELoss(reduce=False)
kl_loss_q = nn.KLDivLoss(reduce=False)
kl_loss_a = nn.KLDivLoss(reduce=False)

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
batch_num = params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)

if params['useCurriculum']:
    if params['continue']:
        rlRound = params['startR']
    else:
        rlRound = params['numRounds'] - params['curriStartmR']
else:
    rlRound = 0

# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------


def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


def polar_att(y, mask, gate=0.7):
    y_hard = torch.zeros(y.size()).cuda()
    for i in range(y.size(0)):
        y_s = torch.masked_select(y[i], mask)
        y_max = y_s.max()
        y_min = y_s.min()
        y_dif = y_max - y_min + 1e-8
        y_norm = (y[i] - y_min) / y_dif
        y_hard[i] = y_norm.ge(gate)
    return y_hard.long()


start_t = timer()
baseline = 0
baseline_a = 0
baseline_q = 0

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
    att_ini = Variable(batch['att_ini'], requires_grad=False)
    image_f = Variable(batch['img_feat_all'], requires_grad=False)
    caption = Variable(batch['cap'], requires_grad=False)
    captionLens = Variable(batch['cap_len'], requires_grad=False)
    gtQuestions = Variable(batch['ques'], requires_grad=False)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)
    gtAnswers = Variable(batch['ans'], requires_grad=False)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)
    options = Variable(batch['opt'], requires_grad=False)
    optionLens = Variable(batch['opt_len'], requires_grad=False)
    gtAnsId = Variable(batch['ans_id'], requires_grad=False)
    ppsNum = image.size(1)

    mask_A = (image.abs().sum(2) > 0)

    # Initializing optimizer and losses
    q_optimizer.zero_grad()
    a_optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    a_rlLoss = 0
    q_rlLoss = 0
    featLoss = 0
    qBotRLLoss = 0
    aBotRLLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']
    # numRounds = 1 # Override for debugging lesser rounds of dialog

    # Setting training modes for both bots and observing captions, images where needed
    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)
    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)

    # Q-Bot image feature regression ('guessing') only occurs if Q-Bot is present
    initialGuess = qBot.predictImage()
    prevFeatDist = mse_criterion(initialGuess, image_f)
    featLoss += torch.mean(prevFeatDist)
    prevFeatDist = torch.mean(prevFeatDist, 1)

    cum_reward = torch.zeros(params['batchSize'])
    if params['useGPU']:
        cum_reward = cum_reward.cuda()
    if params["useECS_Q"] or params["useECS_A"]:
        cum_reward_d = torch.zeros(params['batchSize'])
        if params['useGPU']:
            cum_reward_d = cum_reward_d.cuda()
        cum_reward_i = torch.zeros(params['batchSize'])
        if params['useGPU']:
            cum_reward_i = cum_reward_i.cuda()

    past_dialog_hidden = None
    cur_dialog_hidden = None

    mean_reward_batch = 0
    mean_reward_batch_a = 0
    mean_reward_batch_q = 0
    # calculate the mean reward value for this batch. This will be used to update baseline.
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:
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

        if round >= rlRound and params["AbotMCTS"]:

            factRNN = qBot.encoder.factRNN
            dialogRNN = qBot.encoder.dialogRNN
            dialogState = qBot.encoder.dialogHiddens[-1]

        # Tracking components which require a forward pass
        # A-Bot dialog model
        forwardABot = round < rlRound
        # Q-Bot dialog model
        forwardQBot = round < rlRound
        # Q-Bot feature regression network
        forwardFeatNet = True

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

        # Diversity Penalty
        if params["useCosSimilarityLoss"] or params["useHuberLoss"]:

            if params['trainMode'] == 'sl-qbot' or params['trainMode'] == 'rl-full-QAf':
                cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
            elif params['trainMode'] == 'sl-abot':
                cur_dialog_hidden = aBot.encoder.dialogHiddens[-1][0]

            if round == 0:

                if params['trainMode'] == 'sl-qbot' or params['trainMode'] == 'rl-full-QAf':
                    past_dialog_hidden = qBot.encoder.dialogHiddens[-1][0]
                elif params['trainMode'] == 'sl-abot':
                    past_dialog_hidden = aBot.encoder.dialogHiddens[-1][0]

            else:
                past_dialog_hidden = cur_dialog_hidden

        # A-Bot and Q-Bot interacting in RL rounds
        if round >= rlRound:

            # Run one round of conversation
            questions, quesLens = qBot.forwardDecode(inference='sample')
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            if params["AbotMCTS"]:
                answers, ansLens = aBot.forwardDecode(inference='sample',run_mcts=True)
            else:
                answers, ansLens = aBot.forwardDecode(inference='sample')
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an image prediction after each round
            predFeatures = qBot.predictImage()
            featDist = mse_criterion(predFeatures, image_f)
            featDistPer = torch.mean(featDist, 1)
            featDistLoss = torch.mean(featDist)
            featLoss += featDistLoss

        # ECS-based Rewards
        if params["useECS_Q"]:
            if round == 0:
                prev_att_distr_m = torch.tensor([[0] * ppsNum] * params['batchSize']).view(params['batchSize'], -1).cuda()
                prev_att_distr = att_ini
            cur_att_distr = aBot.encoder.img_atten_weight
            cur_att_distr_m = polar_att(cur_att_distr, mask_A, gate=params["polarG"]) + prev_att_distr_m
            cur_att_distr_m = cur_att_distr_m.ge(1).long()
        if round >= rlRound:
            reward = prevFeatDist.detach() - featDistPer
            reward = reward * params["rewardCoeff"]
            cum_reward = cum_reward + reward.data  # pure reward

            if params["useECS_Q"] or params["useECS_A"]:
                i_loss = cur_att_distr_m - prev_att_distr_m
                i_loss = i_loss.ge(0).float()
                i_loss = torch.sum(i_loss, 1).ge(1).float()
                d_loss = kl_loss_q(prev_att_distr.log(), cur_att_distr)
                d_loss = torch.mean(d_loss, 1)
                prev_att_distr = cur_att_distr
                prev_att_distr_m = cur_att_distr_m
                cum_reward_d = cum_reward_d + d_loss.data
                cum_reward_i = cum_reward_i + i_loss.data

            if params["useECS_Q"]:
                reward_q = reward + d_loss * params["DRCoeff_Q"] + i_loss * params["IRCoeff_Q"]
            if params["useECS_A"]:
                reward_a = reward + d_loss * params["DRCoeff_A"] + i_loss * params["IRCoeff_A"]

            prevFeatDist = featDistPer
            if params['rlAbotReward']:
                mean_reward_batch += float(torch.mean(reward))
                if params["useECS_A"]:
                    mean_reward_batch_a += float(torch.mean(reward_a))
                    aBotRLLoss = aBot.reinforce(reward_a - baseline_a)
                else:
                    aBotRLLoss = aBot.reinforce(reward - baseline)
                if params["useECS_A"]:
                    mean_reward_batch_q += float(torch.mean(reward_q))
                    qBotRLLoss = qBot.reinforce(reward_q - baseline_q)
                else:
                    qBotRLLoss = qBot.reinforce(reward - baseline)

            a_rlLoss += torch.mean(aBotRLLoss)
            q_rlLoss += torch.mean(qBotRLLoss)
        else:
            if params["useECS_Q"]:
                prev_att_distr = cur_att_distr
                prev_att_distr_m = cur_att_distr_m
            prevFeatDist = featDistPer

    if params["useECS_A"]:
        baseline_a = batch_num / (batch_num + 1) * baseline_a + 1 / (batch_num + 1) * (mean_reward_batch_a-mean_reward_batch) / (params["numRounds"] - rlRound)
    if params["useECS_Q"]:
        baseline_q = batch_num / (batch_num + 1) * baseline_q + 1 / (batch_num + 1) * (mean_reward_batch_q-mean_reward_batch) / (params["numRounds"] - rlRound)

    # Loss coefficients
    q_rlCoeff = params['q_RLLossCoeff']
    a_rlCoeff = params['a_RLLossCoeff']
    q_rlLoss = q_rlLoss * q_rlCoeff
    a_rlLoss = a_rlLoss * a_rlCoeff
    featLoss = featLoss * params['featLossCoeff']
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds
    aBotLoss = (params['CELossCoeff'] * aBotLoss) / numRounds
    featLoss = featLoss / numRounds  # / (numRounds+1)

    avg_reward = torch.mean(cum_reward)
    if params["useECS_Q"] or params["useECS_A"]:
        avg_reward_d = torch.mean(cum_reward_d)
        avg_reward_i = torch.mean(cum_reward_i)

    q_loss = qBotLoss + q_rlLoss + featLoss
    a_loss = aBotLoss + a_rlLoss
    q_loss.backward()
    a_loss.backward()
    if params["clipVal"]:
        _ = nn.utils.clip_grad_norm_(q_parameters, params["clipVal"])
        _ = nn.utils.clip_grad_norm_(a_parameters, params["clipVal"])
    q_optimizer.step()
    a_optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = q_loss.item()
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * q_loss.item()

    # Decay learning rate
    if q_lRate > params['q_minLRate'] and iterId % params['decayCir'] == 0:
        for gId, group in enumerate(q_optimizer.param_groups):
            q_optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        q_lRate *= params['lrDecayRate']

    if a_lRate > params['a_minLRate'] and iterId % params['decayCir'] == 0:
        for gId, group in enumerate(a_optimizer.param_groups):
            a_optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        a_lRate *= params['lrDecayRate']

    # RL Annealing: Every epoch after the first, decrease rlRound
    if iterId % numIterPerEpoch == 0 and iterId > 0:
        curEpoch = int(float(iterId) / numIterPerEpoch)
        if curEpoch % params['annealingReduceEpoch'] == 0:
            if params['trainMode'] == 'rl-full-QAf':
                if params['useCurriculum']:
                    rlRound = max(params["annealingEndRound"], rlRound - 1)
                    if rlRound == params["annealingEndRound"]:
                        rlRound = params['numRounds'] - 1
                print('Using rl starting at round {}'.format(rlRound))
                params['rlRound'] = rlRound

    # Print every now and then
    if iterId % 100 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][QLoss: %.3g][ALoss: %.3g]'
        printFormat += '[lr: %.3g]'
        printFormat += '[rwd: %.3g]'
        if params["useECS_Q"] or params["useECS_A"]:
            printFormat += '[ecs_d: %.3g]'
            printFormat += '[ecs_i: %.3g]'
            printInfo = [
                timeStamp, curEpoch, iterId, end_t - start_t, q_loss.item(), a_loss.item(), q_lRate,
                avg_reward.item(), avg_reward_d.item(), avg_reward_i.item()]
        else:
            printInfo = [
                timeStamp, curEpoch, iterId, end_t - start_t, q_loss.item(), a_loss.item(), q_lRate, avg_reward.item()
            ]
        start_t = end_t
        print(printFormat % tuple(printInfo))

        if isinstance(q_rlLoss, Variable):
            avg_reward = torch.mean(cum_reward)

    # Save the model after every epoch
    if iterId % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId

        if aBot:
            params['a_ckpt_lRate'] = a_lRate
            saveFile = os.path.join(params['savePath'],
                                    'abot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, a_optimizer, saveFile, params)
        if qBot:
            params['q_ckpt_lRate'] = q_lRate
            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, q_optimizer, saveFile, params)

    # Evaluate every epoch
    if iterId % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID
        curEpoch = float(iterId) / numIterPerEpoch
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

            huberLossMean = 0
            if params["useHuberLoss"]:
                huberLossMean = params['HuberLossCoeff'] * (
                    rankMetrics['huberLossMean'])

            cosSimilarityLossMean = 0
            if params["useCosSimilarityLoss"]:
                cosSimilarityLossMean = params['CosSimilarityLossCoeff'] * (
                    rankMetrics['cosSimilarityLossMean'])

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean

                    if params["useHuberLoss"]:
                        valLoss += huberLossMean
                    if params["useCosSimilarityLoss"]:
                        valLoss += cosSimilarityLossMean

                    print("valLoss", valLoss)

        if qBot and aBot:
            split = 'val'
            splitName = 'full Val - {}'.format(params['evalTitle'])

            rankMetrics, roundRanks = rankQABots(
                qBot, aBot, dataset, split, beamSize=params['beamSize'])
            for metric, value in rankMetrics.items():
                plotName = splitName + ' - QABots Rank'
                print(metric, value)
