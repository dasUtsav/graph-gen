# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import math
from numpy.core.numeric import NaN
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
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
        # print("adj graph: {} size: {}\n" .format(adj, adj.shape))
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
        self.fc = nn.Linear(2*hidden_size, 1)

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
        text_indices, matrix = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        text_len = text_len.to('cpu')
        # aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
        # print("pre e: {}\n" .format(text.shape))
        text_out, encoder_hidden = encoder(text, text_len)

        # transform enc hidden
        # encoder_hidden = encoder_hidden.transpose(0, 1)
        # encoder_hidden = torch.sum(encoder_hidden, dim=1)
        # encoder_hidden = encoder_hidden.unsqueeze(dim = 1)

        # x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        # print("x1: {} size: {}\n" .format(x, x.size()))

        # feed hidden
        x = F.relu(self.gc1(text_out, matrix))
        # print("x size: {}\n" .format(x.size()))

        # x = text_out

        x_t = x
        x_t = x_t.transpose(1, 2)
        intermed = self.fc(x)
        comp = torch.matmul(x_t, intermed)
        comp = comp.transpose(1, 2)

        #attention
        # x += 0.2 * x
        # x = self.mask(x, aspect_double_idx)
        # alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        # alpha = alpha_mat
        # x = torch.matmul(alpha, text_out).squeeze(1)

        # batch_size * 1 * hidden_size
        return comp
