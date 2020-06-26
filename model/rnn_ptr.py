import numpy as np
import torch
import torch.nn as nn


class RNNPtr(nn.Module):
    def __init__(self, args):
        super(RNNPtr, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.prop_feat = args.prop_feat
        assert 'loc' in self.prop_feat
        self.prop_score = nn.Linear(5, self.hidden_size)
        self.prop_loc = nn.Linear(2, self.hidden_size)
        self.prop_vis = nn.Linear(512, self.hidden_size)
        self.rnn_enc = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.init_ptr = nn.Linear(self.hidden_size, self.hidden_size)
        self.start_feat = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.end_prop_feat = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.rnn_ptr = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.attn = nn.Linear(2 * self.hidden_size, 1)

    def forward(self, prop_score, prop_loc, prop_vis, iou_labels, lens, test=False):
        # prop_score: [B, N, 5] prop_loc: [B, N, 2] prop_vis: [B, N, 512]
        # iou_labels: [B, S, N + 1] lens: [B,]

        batch_size = prop_score.size(0)
        prop_num = prop_score.size(1)

        # prop features
        prop_score = self.prop_score(prop_score)
        prop_loc = self.prop_loc(prop_loc)
        prop_vis = self.prop_vis(prop_vis)

        prop_feat = prop_loc  # [B, N, H]
        if 'score' in self.prop_feat:
            prop_feat += prop_score
        if 'vis' in self.prop_feat:
            prop_feat += prop_vis

        # encode prop
        enc_hidden, enc_final = self.rnn_enc(prop_feat)
        hidden = torch.relu(self.init_ptr(enc_final)).squeeze(0)

        # add end_prop to prop set
        end_prop_feat = self.end_prop_feat.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        prop_feat = torch.cat([end_prop_feat, prop_feat], dim=1)  # [B, N+1, H]

        results = []
        # pointer network
        max_len = self.args.max_seq_len
        for i in range(max_len):
            if i == 0:
                ptr_input = self.start_feat.unsqueeze(0).repeat(batch_size, 1)
            else:
                if test:  # feed the previous prediction
                    idx = torch.argmax(results[-1][:, 0], dim=-1)
                else:  # feed the max_iou prop
                    idx = torch.argmax(iou_labels[:, i-1], dim=-1)
                ptr_input = []
                for j in range(batch_size):
                    ptr_input.append(prop_feat[j][idx[j]])
                ptr_input = torch.stack(ptr_input)
            hidden = self.rnn_ptr(ptr_input, hidden)
            query = hidden.unsqueeze(1).repeat(1, prop_num + 1, 1)
            attn_score = self.attn(torch.cat([query, prop_feat], dim=-1)).squeeze(-1).unsqueeze(1)  # [B, 1, N+1]
            # attn_score = torch.softmax(attn_score, dim=-1)
            attn_score = torch.sigmoid(attn_score)
            results.append(attn_score)
        results = torch.cat(results, dim=1)

        return results
