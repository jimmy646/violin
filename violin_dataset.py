# this code is developed based on https://github.com/jayleicn/TVQA

import numpy as np
import h5py
import os
import json
import re
import torch
import pickle
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from config import get_argparse
from transformers import *

def clean_str(string):
    """ Tokenization/string cleaning for strings.
    Taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?:.\'`]", " ", string)  # <> are added after the cleaning process
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)  # split as two words
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.\.\.", " . ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

class ViolinDataset(Dataset):
    def __init__(self, opt, bert_tokenizer, mode='train'):
        print('='*20)
        super(ViolinDataset, self).__init__()

        self.mode = mode
        self.vid_feat = {}
        self.embed_dim = 300
        self.bert_tokenizer = bert_tokenizer
        self.max_sub_l = opt.max_sub_l
        self.input_streams = opt.input_streams
        self.no_normalize_v = opt.no_normalize_v
        
        entire_clip_info = json.load(open(os.path.join(opt.feat_dir, 'violin_annotation.json'),'r'))
        self.clip_info = []
        
        for clip_id, clip in entire_clip_info.items():
            if clip['split'] == self.mode or self.mode == 'all':
                self.clip_info.append(clip)

        print('dataset mode', self.mode, '\tdata size', len(self.clip_info))
        clip_set = set([clip['file'] for clip in self.clip_info])
        assert len(clip_set) == len(self.clip_info)

        if 'vid' in self.input_streams:
            print('loading video {} features'.format(opt.feat))
            with h5py.File(os.path.join(opt.feat_dir, 'all_res101_pool5_feat.h5' if opt.feat=='resnet' else 'all_c3d_fc6_features.h5'), 'r') as fin:
                 for clip_id in tqdm(fin.keys()):
                    if clip_id in clip_set:
                        if opt.frame == '':
                            self.vid_feat[clip_id] = torch.Tensor(np.array(fin[clip_id+'/{}_features'.format(opt.feat)]))
                        else:
                            tt = torch.Tensor(np.array(fin[clip_id+'/{}_features'.format(opt.feat)]))
                            frame_num = 0
                            if opt.frame == 'last':
                                frame_num = len(tt)-1
                            elif opt.frame == 'mid':
                                frame_num = int(len(tt)/2)
                            self.vid_feat[clip_id] = tt[frame_num].unsqueeze(0)
            assert len(self.vid_feat) == len(self.clip_info)

        print('loading subtitles and statements')
        for clip in tqdm(self.clip_info):
            clip['padded_sub'] = self.tokenize_and_pad(' '.join([clean_str(ss[0]).lower() for ss in clip['sub']]))
            # get statement
            clip['padded_statement'] = [[self.tokenize_and_pad(clean_str(pair[i]).lower()) for i in range(2)] for pair in clip['statement']]

    def tokenize_and_pad(self, sent):
        tokens = self.bert_tokenizer.tokenize(sent)
        if len(tokens) > self.max_sub_l-2:
            tokens = tokens[:self.max_sub_l-2]
        tokens = ['[CLS]']+tokens+['[SEP]']
        sent_len = len(tokens)
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(tokens)
        padding = [0]*(self.max_sub_l-len(tokens))
        input_ids += padding
        input_mask += padding
        return (torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(sent_len,dtype=torch.int))
    
    def __len__(self):
        return len(self.clip_info)*3
    
    def __getitem__(self, idx):
        clip = self.clip_info[int(idx/3)]
        #print(clip['file'])
        
        # visual feat
        if 'vid' in self.input_streams:
            vid_feat = self.vid_feat[clip['file']]
            if not self.no_normalize_v:
                vid_feat = nn.functional.normalize(vid_feat, p=2, dim=1)
        else:
            vid_feat = None

        # subtitles
        sub_input = clip['padded_sub']

        # statements
        state_input = clip['padded_statement'][idx%3]
                       
        return clip['file'], vid_feat, sub_input, state_input

    def get_state_pair(self, idx):
        clip = self.clip_info[int(idx/3)]
        return clip['statement'][idx%3]

def pad_collate(batch):
    def pad_video_seq(vid_seq):
        lengths = torch.LongTensor([len(seq) for seq in vid_seq])
        v_dim = vid_seq[0].size()[1]
        padded_seqs = torch.zeros(len(vid_seq), max(lengths), v_dim).float()
        for idx, seq in enumerate(vid_seq):
            padded_seqs[idx, :lengths[idx]] = seq
        return padded_seqs, lengths
    
    def pad_word_seq(word_list):
        word_seq = [torch.LongTensor(s) for s in word_list]
        lengths = torch.LongTensor([len(seq) for seq in word_seq])
        padded_seqs = torch.zeros(len(word_seq), max(lengths)).long()
        for idx, seq in enumerate(word_seq):
            padded_seqs[idx, :lengths[idx]] = seq
        return padded_seqs, lengths
    
    clip_ids, vid_feat, sub_input, state_input = [[x[i] for x in batch] for i in range(4)]
    padded_vid_feat = pad_video_seq(vid_feat) if type(vid_feat[0]) != type(None) else None

    return clip_ids, padded_vid_feat, sub_input, state_input

def preprocess_batch(batch, bert, opt):
    def clip_seq(seq, lens, max_len):
        if seq.size()[1] > max_len:
            seq = seq[:,:max_len]
            lens = lens.clamp(min=1, max=max_len)
        return seq.to(opt.device), lens.to(opt.device)
    
    def extract_bert_feat(bert_input):
        input_ids = torch.stack([x[0] for x in bert_input]).to(opt.device)
        input_mask = torch.stack([x[1] for x in bert_input]).to(opt.device)
        input_lens = torch.stack([x[2] for x in bert_input]).to(opt.device)
        with torch.no_grad():
            output = bert(input_ids, input_mask)
        return output[0], input_lens
    
    clip_ids, padded_vid_feat, sub_input, state_input = batch
    ret = []
    ret.append(clip_ids)
    if 'vid' in opt.input_streams:
        ret.append(clip_seq(padded_vid_feat[0], padded_vid_feat[1], opt.max_vid_l))
    else:
        ret.append(None)
    if 'sub' in opt.input_streams:
        ret.append(extract_bert_feat(sub_input))
    else:
        ret.append(None)
    ret.append([extract_bert_feat([x[i] for x in state_input]) for i in range(2)])
    return ret
