import os
import time
import json
import h5py
import numpy as np
import torch
from six import iteritems
from six.moves import range
from sklearn.preprocessing import normalize
import torch.nn.functional as F
from torch.utils.data import Dataset
import _pickle as cPickle
from typing import Dict, List, Union

class VisDialDataset(Dataset):
    def __init__(self, params, subsets):
        '''
            Initialize the dataset with splits given by 'subsets', where
            subsets is taken from ['train', 'val', 'test']

            Notation:
                'dtype' is a split taking values from ['train', 'val', 'test']
                'stype' is a sqeuence type from ['ques', 'ans']
        '''

        # By default, load Q-Bot, A-Bot and dialog options for A-Bot
        self.useQuestion = True
        self.useAnswer = True
        self.useOptions = True
        self.useHistory = True
        self.useIm = True
        self.useNDCG = params["useNDCG"]
        self.imgNorm = params["imgNorm"]

        # Absorb parameters
        for key, value in iteritems(params):
            setattr(self, key, value)
        self.subsets = tuple(subsets)
        self.numRounds = params['numRounds']

        self.img_input = './data/image_features/{}_btmup_f.hdf5'
        self.input_img2idx = './data/image_features/{}_imgid2idx.pkl'

        print('\nDataloader loading json file: ' + self.inputJson)
        time1 = time.time()
        with open(self.inputJson, 'r') as fileId:
            info = json.load(fileId)
            # Absorb values
            for key, value in iteritems(info):
                setattr(self, key, value)
        if 'val' in subsets and self.useNDCG:
            with open(self.inputDenseJson, 'r') as fileId:
                dense_annotation = json.load(fileId)
                self.dense_annotation = dense_annotation

        wordCount = len(self.word2ind)
        # Add <START> and <END> to vocabulary
        self.word2ind['<START>'] = wordCount + 1
        self.word2ind['<END>'] = wordCount + 2
        self.startToken = self.word2ind['<START>']
        self.endToken = self.word2ind['<END>']
        # Padding token is at index 0
        self.vocabSize = wordCount + 3
        print('Vocab size with <START>, <END>: %d' % self.vocabSize)
        print("timing......%f" % (time.time()-time1))
        time1 = time.time()
        # Construct the reverse map
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        # Read questions, answers and options
        print('Dataloader loading h5 file: ' + self.inputQues)
        print("timing......%f" % (time.time() - time1))
        time1 = time.time()
        quesFile = h5py.File(self.inputQues, 'r')

        # Number of data points in each split (train/val/test)
        self.numDataPoints = {}
        self.data = {}

        # map from load to save labels
        ioMap = {
            'ques_%s': '%s_ques',
            'ques_length_%s': '%s_ques_len',
            'ans_%s': '%s_ans',
            'ans_length_%s': '%s_ans_len',
            'ans_index_%s': '%s_ans_ind',
            'img_pos_%s': '%s_img_pos',
            'opt_%s': '%s_opt',
            'opt_length_%s': '%s_opt_len',
            'opt_list_%s': '%s_opt_list',
            'num_rounds_%s': '%s_num_rounds'
        }

        # Processing every split in subsets
        for dtype in subsets:  # dtype is in [train, val, test]
            print("\nProcessing split [%s]..." % dtype)
            print("timing......%f" % (time.time() - time1))
            time1 = time.time()
            if ('ques_%s' % dtype) not in quesFile:
                self.useQuestion = False
            if ('ans_%s' % dtype) not in quesFile:
                self.useAnswer = False
            if ('opt_%s' % dtype) not in quesFile:
                self.useOptions = False
            # read the question, answer, option related information
            for loadLabel, saveLabel in iteritems(ioMap):
                if loadLabel % dtype not in quesFile:
                    continue
                dataMat = np.array(quesFile[loadLabel % dtype], dtype='int64')
                self.data[saveLabel % dtype] = torch.from_numpy(dataMat)

            # Read image features, if needed
            if self.useIm:
                # Read images
                print('Dataloader loading h5 file: ' + self.img_input.format(dtype))
                print("timing......%f" % (time.time() - time1))
                time1 = time.time()
                imgFile = h5py.File(self.img_input.format(dtype), 'r')
                print('Reading image features...')
                input_img2idx = self.input_img2idx.format(dtype)
                self.data['%s_img2idx' % dtype] = cPickle.load(open(input_img2idx, 'rb'))
                print("timing......%f" % (time.time() - time1))
                time1 = time.time()
                self.imgFeats = torch.from_numpy(np.array(imgFile.get('image_features')))
                self.spatials = torch.from_numpy(np.array(imgFile.get('spatial_features')))
                self.pos_boxes = torch.from_numpy(np.array(imgFile.get('pos_boxes')))
                # imgIdList = list(imgFile['image_id'])

                # if not self.imgNorm:
                #     continue
                # normalize, if needed
                # print('Normalizing image features..')
                # imgFeats = normalize(imgFeats, axis=1, norm='l2')

                # save img features
                self.data['%s_img_fv' % dtype] = self.imgFeats
                self.data['%s_img_fp' % dtype] = self.pos_boxes
                # self.data['%s_img_fi' % dtype] = imgIdList
                # get img names
                img_fnames = getattr(self, 'unique_img_%s'%dtype)
                self.data['%s_img_fnames' % dtype] = img_fnames

            # read the history, if needed
            if self.useHistory:
                captionMap = {
                    'cap_%s': '%s_cap',
                    'cap_length_%s': '%s_cap_len'
                }
                for loadLabel, saveLabel in iteritems(captionMap):
                    mat = np.array(quesFile[loadLabel % dtype], dtype='int32')
                    self.data[saveLabel % dtype] = torch.from_numpy(mat)

            # Number of data points
            self.numDataPoints[dtype] = self.data[dtype + '_cap'].size(0)

        # Prepare dataset for training
        for dtype in subsets:
            print("\nSequence processing for [%s]..." % dtype)
            print("timing......%f" % (time.time() - time1))
            time1 = time.time()
            self.prepareDataset(dtype)
        print("")

        # Default pytorch loader dtype is set to train
        if 'train' in subsets:
            self._split = 'train'
        else:
            self._split = subsets[0]
        #
        # if "val" in self._split:
        #     self.annotations_reader = DenseAnnotationsReader(params["inputDenseJson"])
        # else:
        #     self.annotations_reader = None

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self.subsets  # ['train', 'val', 'test']
        self._split = split

    #----------------------------------------------------------------------------
    # Dataset-preprocessing
    #----------------------------------------------------------------------------

    def prepareDataset(self, dtype):
        if self.useHistory:
            self.processCaption(dtype)

        # prefix/postfix with <START> and <END>
        if self.useOptions:
            self.processOptions(dtype)
            # options are 1-indexed, changed to 0-indexed
            self.data[dtype + '_opt'] -= 1

        # process answers and questions
        if self.useAnswer:
            self.processSequence(dtype, stype='ans')
            # 1 indexed to 0 indexed
            if dtype != 'test':
                self.data[dtype + '_ans_ind'] -= 1
        if self.useQuestion:
            self.processSequence(dtype, stype='ques')

    def processSequence(self, dtype, stype='ans'):
        '''
        Add <START> and <END> token to answers or questions.
        Arguments:
            'dtype'    : Split to use among ['train', 'val', 'test']
            'sentType' : Sequence type, either 'ques' or 'ans'
        '''
        assert stype in ['ques', 'ans']
        prefix = dtype + "_" + stype

        seq = self.data[prefix]
        seqLen = self.data[prefix + '_len']

        numConvs, numRounds, maxAnsLen = seq.size()
        newSize = torch.Size([numConvs, numRounds, maxAnsLen + 2])
        sequence = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        sequence[:, :, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for thId in range(numConvs):
            for rId in range(numRounds):
                length = seqLen[thId, rId]
                if length == 0:
                    # print('Warning: Skipping empty %s sequence at (%d, %d)'\
                    #       %(stype, thId, rId))
                    continue

                sequence[thId, rId, 1:length + 1] = seq[thId, rId, :length]
                sequence[thId, rId, length + 1] = endTokenId
        if stype == 'ans':
            mask = torch.LongTensor(newSize).fill_(0)
            mask[:, :, :maxAnsLen+1] = sequence[:, :, 1:].gt(0)
            self.data[prefix + "_mask"] = mask

        # Sequence length is number of tokens + 1
        self.data[prefix + "_len"] = seqLen + 1
        self.data[prefix] = sequence

    def processCaption(self, dtype):
        '''
        Add <START> and <END> token to caption.
        Arguments:
            'dtype'    : Split to use among ['train', 'val', 'test']
        '''
        prefix = dtype + '_cap'

        seq = self.data[prefix]
        seqLen = self.data[prefix + '_len']

        numConvs, maxCapLen = seq.size()
        newSize = torch.Size([numConvs, maxCapLen + 2])
        sequence = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        sequence[:, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for thId in range(numConvs):
            length = seqLen[thId]
            if length == 0:
                # print('Warning: Skipping empty %s sequence at (%d)' % (stype,
                #                                                        thId))
                continue

            sequence[thId, 1:length + 1] = seq[thId, :length]
            sequence[thId, length + 1] = endTokenId

        # Sequence length is number of tokens + 1
        self.data[prefix + "_len"] = seqLen + 1
        self.data[prefix] = sequence

    def processOptions(self, dtype):
        ans = self.data[dtype + '_opt_list']
        ansLen = self.data[dtype + '_opt_len']

        ansListLen, maxAnsLen = ans.size()

        newSize = torch.Size([ansListLen, maxAnsLen + 2])
        options = torch.LongTensor(newSize).fill_(0)

        # decodeIn begins with <START>
        options[:, 0] = self.word2ind['<START>']
        endTokenId = self.word2ind['<END>']

        for ansId in range(ansListLen):
            length = ansLen[ansId]
            if length == 0:
                # print('Warning: Skipping empty option answer list at (%d)'\
                #         %ansId)
                continue

            options[ansId, 1:length + 1] = ans[ansId, :length]
            options[ansId, length + 1] = endTokenId

        self.data[dtype + '_opt_len'] = ansLen + 1
        self.data[dtype + '_opt_seq'] = options

    #----------------------------------------------------------------------------
    # Dataset helper functions for PyTorch's datalaoder
    #----------------------------------------------------------------------------

    def __len__(self):
        # Assert that loader_dtype is in subsets ['train', 'val', 'test']
        return self.numDataPoints[self._split]

    def __getitem__(self, idx):
        item = self.getIndexItem(self._split, idx)
        return item

    def collate_fn(self, batch):
        out = {}

        mergedBatch = {key: [d[key] for d in batch] for key in batch[0]}
        for key in mergedBatch:
            if key == 'img_fname' or key == 'index':
                out[key] = mergedBatch[key]
            elif key in {'img_feat'}:
                num_max_boxes = max([x.size(0) for x in mergedBatch['img_feat']])
                out[key] = torch.stack([F.pad(x, (0,0,0,num_max_boxes-x.size(0))).data for x in mergedBatch['img_feat']], 0)
            elif key in {'att_ini'}:
                num_max_boxes = max([x.size(0) for x in mergedBatch['att_ini']])
                out[key] = torch.stack(
                    [F.pad(x, (0, num_max_boxes - x.size(0))).data for x in mergedBatch['att_ini']], 0)

            elif key == 'cap_len':
                # 'cap_lens' are single integers, need special treatment
                out[key] = torch.LongTensor(mergedBatch[key])
            else:
                out[key] = torch.stack(mergedBatch[key], 0)

        # Dynamic shaping of padded batch
        if 'ques' in out.keys():
            quesLen = out['ques_len'] + 1
            out['ques'] = out['ques'][:, :, :torch.max(quesLen)].contiguous()
            # out['ques_mask'] = out['ques_mask'][:, :, :torch.max(quesLen)].contiguous()

        if 'ans' in out.keys():
            ansLen = out['ans_len'] + 1
            out['ans'] = out['ans'][:, :, :torch.max(ansLen)].contiguous()
            out['ans_mask'] = out['ans_mask'][:, :, :torch.max(ansLen)].contiguous()

        if 'cap' in out.keys():
            capLen = out['cap_len'] + 1
            out['cap'] = out['cap'][:, :torch.max(capLen)].contiguous()

        if 'opt' in out.keys():
            optLen = out['opt_len'] + 1
            out['opt'] = out['opt'][:, :, :, :torch.max(optLen) + 2].contiguous()

        return out

    #----------------------------------------------------------------------------
    # Dataset indexing
    #----------------------------------------------------------------------------

    def getIndexItem(self, dtype, idx):
        item = {'index': idx}

        item['num_rounds'] = torch.LongTensor([self.data[dtype + '_num_rounds'][idx]])

        # get question
        if self.useQuestion:
            ques = self.data[dtype + '_ques'][idx]
            quesLen = self.data[dtype + '_ques_len'][idx]
            # quesMask = self.data[dtype + '_ques_mask'][idx]

            # hacky! packpadded sequence error for zero length sequences in 0.3. add 1 here if split is test.
            # zero length seqences have length 1 because of start token
            if dtype == 'test':
                quesLen[quesLen == 1] = 2
                
            item['ques'] = ques
            item['ques_len'] = quesLen
            # item['ques_mask'] = quesMask

        # get answer
        if self.useAnswer:
            ans = self.data[dtype + '_ans'][idx]
            ansLen = self.data[dtype + '_ans_len'][idx]
            ansMask = self.data[dtype + '_ans_mask'][idx]
            # hacky! packpadded sequence error for zero length sequences in 0.3. add 1 here if split is test.
            # zero length seqences have length 1 because of start token

            if dtype == 'test':
                ansLen[ansLen == 1] = 2

            item['ans_len'] = ansLen
            item['ans'] = ans
            item['ans_mask'] = ansMask

        # get caption
        if self.useHistory:
            cap = self.data[dtype + '_cap'][idx]
            capLen = self.data[dtype + '_cap_len'][idx]
            item['cap'] = cap
            item['cap_len'] = capLen

        if self.useOptions:
            optInds = self.data[dtype + '_opt'][idx]
            ansId = None
            if dtype != 'test':
                ansId = self.data[dtype + '_ans_ind'][idx]

            optSize = list(optInds.size())
            newSize = torch.Size(optSize + [-1])

            indVector = optInds.view(-1)
            optLens = self.data[dtype + '_opt_len'].index_select(0, indVector)
            optLens = optLens.view(optSize)

            opts = self.data[dtype + '_opt_seq'].index_select(0, indVector)

            item['opt'] = opts.view(newSize)
            item['opt_len'] = optLens
            if dtype != 'test':
                item['ans_id'] = ansId

        # if image needed
        if self.useIm:
            item['img_fname'] = self.data[dtype + '_img_fnames'][idx]
            img_id = int(item['img_fname'].split('000000')[-1].split('.')[0])
            feature_idx = self.data['%s_img2idx' % dtype][img_id]
            item['img_feat'] = self.data[dtype + '_img_fv'][self.data[dtype + '_img_fp'][feature_idx][0]:self.data[dtype + '_img_fp'][feature_idx][1], :]
            pps_num = self.data[dtype + '_img_fp'][feature_idx][1] - self.data[dtype + '_img_fp'][feature_idx][0]
            if self.imgNorm:
                item['img_feat'] = F.normalize(item['img_feat'], dim=1, p=2)

            if dtype + '_img_labels' in self.data:
                item['img_label'] = self.data[dtype + '_img_labels'][idx]
            item['att_ini'] = torch.ones([pps_num]) / float(pps_num)
            item['img_feat_all'] = torch.sum(item['img_feat'], dim=0) / float(pps_num)

        # dense annotations if val set
        if dtype == 'val' and self.useNDCG:

            round_id = self.dense_annotation[idx]['round_id']
            gt_relevance = self.dense_annotation[idx]['gt_relevance']
            image_id = self.dense_annotation[idx]['image_id']
            item["round_id"] = torch.LongTensor([round_id])
            item["gt_relevance"] = torch.FloatTensor(gt_relevance)
            item["image_id"] = torch.LongTensor([image_id])

        return item
