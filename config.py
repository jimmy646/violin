import os
import time
import torch
import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir_base", type=str, default="results/results")
    parser.add_argument("--feat_dir", type=str, default="../../feat")
    parser.add_argument("--bert_dir", type=str, default="../bert_output")
    
    parser.add_argument("--model", type=str, default="ViolinBase", choices=['ViolinBase'])
    parser.add_argument("--data", type=str, default="ViolinDataset", choices=['ViolinDataset'])
    #parser.add_argument("--log_freq", type=int, default=10, help="print, save training info")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--n_epoch", type=int, default=40, help="number of epochs to run")
    #parser.add_argument("--grad_clip", type=float, default=0.01, help="gradient clip value")
    #parser.add_argument("--init_train_epoch", type=int, default=15, help="number of epochs for initial train (without early stopping)")
    #parser.add_argument("--max_es_cnt", type=int, default=200, help="number of epochs to early stop")
    parser.add_argument("--batch_size", type=int, default=256, help="mini-batch size")
    parser.add_argument("--test_batch_size", type=int, default=256, help="mini-batch size for testing")
    parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")

    parser.add_argument("--vid_feat_size", type=int, default=4096, help="visual feature dimension")
    parser.add_argument("--input_streams", type=str, nargs="+", choices=["vid", "sub", "none"], default=[], help="input streams for the model, or use a single option 'none'")

    parser.add_argument("--hsize1", type=int, default=150, help="hidden size for the video lstm")
    parser.add_argument("--hsize2", type=int, default=300, help="hidden size for the fusion lstm")
    parser.add_argument("--embed_size", type=int, default=768, help="word embedding dim")
    parser.add_argument("--max_sub_l", type=int, default=256, help="max length for subtitle")
    parser.add_argument("--max_vid_l", type=int, default=256, help="max length for video feature")
    parser.add_argument("--no_normalize_v", action="store_true", help="do not normalize video featrue")
    
    parser.add_argument("--feat", type=str, default="resnet", choices=['resnet', 'c3d'])
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--frame", type=str, default="", choices=['first', 'last', 'mid', ''], help="testing with only one frame")
    
    opt = parser.parse_args()
    if opt.device >= 0:
        opt.device = torch.device('cuda:0')
    opt.results_dir = opt.results_dir_base + time.strftime("_%Y_%m_%d_%H_%M_%S")+'_'+opt.model
    if opt.frame != '':
        opt.results_dir+='_frame-'+opt.frame
    opt.results_dir += '_'+'-'.join(opt.input_streams+[opt.feat])
    if 'none' in opt.input_streams:
        assert len(opt.input_streams) == 1
        opt.input_streams = []
    opt.vid_feat_size = 2048 if opt.feat == 'resnet' else 4096

    return opt
