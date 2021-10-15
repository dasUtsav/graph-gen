# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import math
from numpy.core.numeric import NaN
from scipy.sparse import compressed
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        torch.manual_seed(5)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn((in_features, out_features), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.randn((out_features), dtype=torch.float))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # hidden = torch.matmul(text, self.weight.double())
        # print("graph weight: {}" .format(self.weight))
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # print("hidden graph: {} size: {}\n" .format(hidden, hidden.shape))
        # print("denom graph: {} size: {}\n" .format(denom, denom.shape))
        output = torch.matmul(adj, hidden.float()) / denom
        # print("output: {} size: {}\n" .format(output, output.shape))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class INTERGCN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(INTERGCN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.gc1 = GraphConvolution(2*hidden_size, 2*hidden_size)
        self.gc2 = GraphConvolution(2*hidden_size, 2*hidden_size)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(4*hidden_size, 1)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(device)
        return mask*x

    def forward(self, encoder, inputs):
        if len(inputs) == 3:
            # print("Split mode\n")
            text_indices, svo, nonsvo = inputs

            text_len = torch.sum(text_indices != 0, dim=-1)
            text_len = text_len.to('cpu')
            # aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
            text = self.embed(text_indices)
            text = self.text_embed_dropout(text)
            # print("pre e: {}\n" .format(text.shape))
            text_out, enc_hidden = encoder(text, text_len)

            # feed hidden
            # print("text size: {}\n svo size: {}\n" .format(text_out.size(), svo.size()))
            x1 = F.relu(self.gc1(text_out, svo))
            x2 = F.relu(self.gc2(text_out, nonsvo))
            # print("x1 size: {} x2 size: {}\n" .format(x1.size(), x2.size()))


        elif len(inputs) == 4:
            # print("Multi Split mode\n")
            text_indices1, text_indices2, svo, nonsvo = inputs

            text_len1 = torch.sum(text_indices1 != 0, dim=-1)
            text_len1 = text_len1.to('cpu')
            text_len2 = torch.sum(text_indices2 != 0, dim=-1)
            text_len2 = text_len2.to('cpu')
            # aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
            text1 = self.embed(text_indices1)
            text1 = self.text_embed_dropout(text1)
            text2 = self.embed(text_indices2)
            text2 = self.text_embed_dropout(text2)
            # print("pre e: {}\n" .format(text.shape))
            text_out1, enc_hidden1 = encoder(text1, text_len1)
            text_out2, enc_hidden2 = encoder(text2, text_len2)

            if text1.size(0) != text2.size(0):
                print("text1: {} text2: {} svo: {} nonsvo: {}\n". format(text1.size(), text2.size(), svo.size(), nonsvo.size()))
                return []
                

            x1 = F.relu(self.gc1(text_out1, svo))
            x2 = F.relu(self.gc2(text_out2, nonsvo))
            # print("x1 size: {} x2 size: {}\n" .format(x1.size(), x2.size()))

       
        # print("text out: {}\n" .format(text_out))

        # x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        # print("x1: {} size: {}\n" .format(x, x.size()))

    
        # print("x1 size: {} x2 size: {}\n" .format(x1.size(), x2.size()))
        x = torch.cat((x1, x2), 2)
        # print("x size: {}\n" .format(x.size()))

        x_t = x
        x_t = x_t.transpose(1, 2)
        intermed = self.fc(x)
        comp = torch.matmul(x_t, intermed)
        comp = comp.transpose(1, 2)

        # print("comp size: {}\n" .format(comp.size()))

        return comp

        #attention
        # x += 0.2 * x
        # x = self.mask(x, aspect_double_idx)
        # alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        # alpha = alpha_mat
        # x = torch.matmul(alpha, text_out).squeeze(1)

