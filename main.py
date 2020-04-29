# this code is developed based on https://github.com/jayleicn/TVQA

import os
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from violin_dataset import ViolinDataset, pad_collate, preprocess_batch
from config import get_argparse
from model.ViolinBase import ViolinBase

from transformers import *


def check_param(model):
    grad_lst = []
    for name, param in model.named_parameters():
        grad_lst.append(torch.norm(param.grad.data.view(-1)).item())
    return grad_lst

def get_data_loader(opt, dset, batch_size, if_shuffle):
    return DataLoader(dset, batch_size=batch_size, shuffle=if_shuffle, num_workers=0, collate_fn=pad_collate)

def train_epoch(opt, trn_dset, val_dset, tst_dset, model, optimizer, epoch, previous_best_acc):
    model.train()
    train_loader = get_data_loader(opt, trn_dset, opt.batch_size, True)

    #check_param(model)
    train_loss = []
    valid_acc_log = ["epoch\ttrn acc\tval acc"]
    train_corrects = []
    train_real_corrects = []
    train_fake_corrects = []
    print('epoch', epoch, '='*20)
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        cur_clip_ids, padded_vid_feat, sub_feat, state_feat = preprocess_batch(batch, bert, opt)
        real_state_output = model(padded_vid_feat, sub_feat, state_feat[0]).squeeze()
        fake_state_output = model(padded_vid_feat, sub_feat, state_feat[1]).squeeze()
        #assert real_state_output.size() == torch.Size([state_feat[0][0].size()[0]])

        threshold = 0.5
        loss = torch.mean(-torch.log(1.0-fake_state_output)-torch.log(real_state_output), dim=0)
        loss_sum = torch.sum(-torch.log(1.0-fake_state_output)-torch.log(real_state_output), dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        train_loss.append(loss_sum.item())
        train_corrects += (real_state_output>=threshold).to(torch.device('cpu')).tolist()
        train_corrects += (fake_state_output<threshold).to(torch.device('cpu')).tolist()
        #print(train_corrects)
        train_real_corrects += (real_state_output>=threshold).to(torch.device('cpu')).tolist()
        train_fake_corrects += (fake_state_output<threshold).to(torch.device('cpu')).tolist()
    
    train_acc = sum(train_corrects) / float(len(train_corrects))
    train_loss = sum(train_loss) / float(len(train_corrects))
    train_real_acc = sum(train_real_corrects) / float(len(train_real_corrects))
    train_fake_acc = sum(train_fake_corrects) / float(len(train_fake_corrects))
    #print(check_param(model))
    
    # validate
    valid_loader = get_data_loader(opt, val_dset, opt.test_batch_size, False)
    valid_loss, valid_acc, valid_real_acc, valid_fake_acc, valid_real_corrects, valid_fake_corrects = validate(model, valid_loader)

    valid_log_str = "%02d\t%.4f\t%.4f" % (epoch, train_acc, valid_acc)
    valid_acc_log.append(valid_log_str)
    print("\n Epoch %d TRAIN loss %.4f acc %.4f real acc %.4f fake acc %.4f VAL loss %.4f acc %.4f real acc %.4f fake acc %.4f\n"
            % (epoch, train_loss, train_acc, train_real_acc, train_fake_acc, valid_loss, valid_acc, valid_real_acc, valid_fake_acc))
    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("Epoch %d TRAIN loss %.4f acc %.4f real acc %.4f fake acc %.4f VAL loss %.4f acc %.4f real acc %.4f fake acc %.4f\n"
            % (epoch, train_loss, train_acc, train_real_acc, train_fake_acc, valid_loss, valid_acc, valid_real_acc, valid_fake_acc))

    torch.save(model.state_dict(), os.path.join(opt.results_dir, "model_epoch_{}.pth".format(epoch)))
    if valid_acc > previous_best_acc:
        previous_best_acc = valid_acc
        torch.save(model.state_dict(), os.path.join(opt.results_dir, "best_valid.pth"))

        test_loader = get_data_loader(opt, tst_dset, opt.test_batch_size, False)
        test_loss, test_acc, test_real_acc, test_fake_acc, _, _ = validate(model, test_loader)
        print("\n Epoch %d TEST loss %.4f acc %.4f real acc %.4f fake acc %.4f\n"
            % (epoch, test_loss, test_acc, test_real_acc, test_fake_acc))
        with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
            f.write("Epoch %d TEST loss %.4f acc %.4f real acc %.4f fake acc %.4f\n"
            % (epoch, test_loss, test_acc, test_real_acc, test_fake_acc))
    
    return previous_best_acc

def validate(model, valid_loader):
    model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_corrects = []
        valid_real_corrects = []
        valid_fake_corrects = []
        clip_ids = []
        for _, batch in enumerate(tqdm(valid_loader)):
            cur_clip_ids, padded_vid_feat, sub_feat, state_feat = preprocess_batch(batch, bert, opt)
            real_state_output = model(padded_vid_feat, sub_feat, state_feat[0]).squeeze()
            fake_state_output = model(padded_vid_feat, sub_feat, state_feat[1]).squeeze()
            #assert real_state_output.size() == torch.Size([padded_vid_feat[0].size()[0]])

            threshold = 0.5
            loss = torch.mean(-torch.log(1.0-fake_state_output)-torch.log(real_state_output), dim=0)
            loss_sum = torch.sum(-torch.log(1.0-fake_state_output)-torch.log(real_state_output), dim=0)

            # measure accuracy and record loss
            valid_loss.append(loss_sum.item())
            clip_ids += cur_clip_ids
            valid_corrects += (real_state_output>=threshold).to(torch.device('cpu')).tolist()
            valid_corrects += (fake_state_output<threshold).to(torch.device('cpu')).tolist()
            valid_real_corrects += (real_state_output>=threshold).to(torch.device('cpu')).tolist()
            valid_fake_corrects += (fake_state_output<threshold).to(torch.device('cpu')).tolist()
            
        valid_acc = sum(valid_corrects) / float(len(valid_corrects))
        valid_loss = sum(valid_loss) / float(len(valid_corrects))
        valid_real_acc = sum(valid_real_corrects) / float(len(valid_real_corrects))
        valid_fake_acc = sum(valid_fake_corrects) / float(len(valid_fake_corrects))

    return valid_loss, valid_acc, valid_real_acc, valid_fake_acc, valid_real_corrects, valid_fake_corrects

if __name__ == '__main__':
    random_seed = 219373241
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    opt = get_argparse()
    
    bert = BertModel.from_pretrained(opt.bert_dir)
    bert_tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    bert.to(opt.device)
    bert.eval()
    
    DSET = eval(opt.data)
    
    if not opt.test:
        os.makedirs(opt.results_dir)
        trn_dset = DSET(opt, bert_tokenizer, 'train')
        val_dset = DSET(opt, bert_tokenizer, 'validate')
        tst_dset = DSET(opt, bert_tokenizer, 'test')
    else:
        tst_dset = DSET(opt, bert_tokenizer, 'test')
    
    model = eval(opt.model)(opt)
    print(model)
    
    if opt.test:
        model.load_state_dict(torch.load(opt.model_path))
    
    model.to(opt.device)
    
    if opt.test:
        test_loader = get_data_loader(opt, tst_dset, opt.test_batch_size, False)
        test_loss, test_acc, test_real_acc, test_fake_acc, test_real_corrects, test_fake_corrects = validate(model, test_loader)
        print("Test loss %.4f acc %.4f real acc %.4f fake acc %.4f\n"
            % (test_loss, test_acc, test_real_acc, test_fake_acc))
        with open(opt.model_path+'_test.res','w') as ftst:
            ftst.write("Test loss %.4f acc %.4f real acc %.4f fake acc %.4f\n"
            % (test_loss, test_acc, test_real_acc, test_fake_acc))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

        best_acc = 0.
        for epoch in range(opt.n_epoch):
            best_acc = train_epoch(opt, trn_dset, val_dset, tst_dset, model, optimizer, epoch, best_acc)