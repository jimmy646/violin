# this code is developed based on https://github.com/jayleicn/TVQA

import torch
from torch import nn
from .rnn import RNNEncoder
from .bidaf import BidafAttn
import pickle

class ViolinBase(nn.Module):
    def __init__(self, opt):
        super(ViolinBase, self).__init__()
        hsize1 = opt.hsize1
        hsize2 = opt.hsize2
        embed_size = opt.embed_size
        
        self.input_streams = opt.input_streams
        self.lstm_raw = RNNEncoder(opt.embed_size, hsize1, bidirectional=True, dropout_p=0, n_layers=1, rnn_type='lstm')
        self.lstm_mature_vid = RNNEncoder(hsize1 * 2 * 5, hsize2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
        
        self.bert_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(opt.embed_size, hsize1*2),
            nn.Tanh()
        )
        if 'vid' in self.input_streams:
            self.video_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(opt.vid_feat_size, embed_size),
                nn.Tanh()
            )
            self.vid_ctx_rnn = RNNEncoder(hsize1 * 2 * 3, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        
        if 'sub' in self.input_streams:
            self.sub_ctx_rnn = RNNEncoder(hsize1 * 2 * 3, hsize2, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        
        if len(self.input_streams) > 0:
            self.bidaf = BidafAttn(hsize1 * 2, method="dot")  

            self.final_fc = nn.Sequential(
                nn.Linear(hsize2*2, hsize2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hsize2, 1),
                nn.Sigmoid()
            )
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(hsize1*2, hsize2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hsize2, 1),
                nn.Sigmoid()
            )
    
    def max_along_time(self, outputs, lengths):
        max_outputs = [outputs[i, :int(lengths[i]), :].max(dim=0)[0] for i in range(len(lengths))]
        ret = torch.stack(max_outputs, dim=0)
        assert ret.size() == torch.Size([outputs.size()[0], outputs.size()[2]])
        return ret

    def forward(self, vid_input, sub_input, state_input):
        final_vectors = []
        
        state_hidden, state_lens = state_input
        state_encoded = self.bert_fc(state_hidden)
        
        #print(type(state_lens))
        if 'vid' in self.input_streams:
            vid_feat, vid_lens = vid_input
            vid_projected = self.video_fc(vid_feat)
            vid_encoded, _ = self.lstm_raw(vid_projected, vid_lens)

            u_va, _ = self.bidaf(state_encoded, state_lens, vid_encoded, vid_lens)
            concat_vid = torch.cat([state_encoded, u_va, state_encoded*u_va], dim=-1)
            vec_vid = self.vid_ctx_rnn(concat_vid, state_lens)[1]
            
            final_vectors.append(vec_vid)

        if 'sub' in self.input_streams:
            sub_hidden, sub_lens = sub_input
            sub_encoded = self.bert_fc(sub_hidden)
            
            u_sa, _ = self.bidaf(state_encoded, state_lens, sub_encoded, sub_lens)
            concat_sub = torch.cat([state_encoded, u_sa, state_encoded*u_sa], dim=-1)

            vec_sub = self.sub_ctx_rnn(concat_sub, state_lens)[1]
            
            final_vectors.append(vec_sub)
        
        if len(self.input_streams) == 0:
            maxout_state = self.max_along_time(state_encoded, state_lens)
            final_vectors.append(maxout_state)
        
        if len(self.input_streams) < 2:
            return self.final_fc(torch.cat(final_vectors, dim=1))
        else:
            concat_all = torch.cat([state_encoded, u_va, u_sa, state_encoded*u_va, state_encoded*u_sa], dim=-1)
            vec_all = self.lstm_mature_vid(concat_all, state_lens)[1]
            return self.final_fc(vec_all)

