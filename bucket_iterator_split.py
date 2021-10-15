# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import math
import random
import torch
import numpy
from torch.nn.utils.rnn import pad_sequence
from config import config

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='svo', shuffle=True, sort=False):
        print("Batching data...\n")
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)
        self.random_seed = 5

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data_seq(self, batch_data):
        batch_indices = []
        for item in batch_data:
            text_indices = item['text_indices']
            batch_indices.append(torch.tensor([config['SOS_token']] + text_indices + [config['EOS_token']]))

        batch_indices = pad_sequence(batch_indices, batch_first=True, padding_value=config['PAD_token'])

        return { \
            'text_indices': batch_indices.to(config['device'])}

    def pad_graph(graph, glob_max_len):
        
        batch_graph = []

        # max_len = max([len(v) for k, v in graph.items()])
        
        for k, v in graph.items():
            batch_graph.append(numpy.pad(v, \
                ((0, glob_max_len - len(v)),(0, glob_max_len - len(v))), 'constant'))

        return torch.tensor(batch_graph)

    def pad_data(self, batch_data):
        batch_context = []
        batch_text_indices = []
        max_len_text_indices = max([len(t['text_indices']) for t in batch_data])
        # max_len_graph = max([len(t['svo']) for t in batch_data])
        # if max_len_graph >= max_len_text_indices:
        #     max_len = max_len_graph
        # else:
        #     max_len = max_len_text_indices
        # max_len = max_len_text_indices
        for item in batch_data:
            context, text_indices = \
                item['context'], item['text_indices']
            # text_padding = [config['PAD_token']] * (max_len - len(text_indices))
            batch_context.append(context)
            # batch_text_indices.append(torch.tensor([config['SOS_token']] + text_indices + [config['EOS_token']] + text_padding))
            # batch_text_indices.append([config['SOS_token']] + text_indices + [config['EOS_token']] + text_padding)
            batch_text_indices.append(text_indices)
            # batch_svo.append(numpy.pad(svo, \
            #     ((0, max_len - len(svo)),(0, max_len - len(svo))), 'constant'))
            # batch_nonsvo.append(numpy.pad(nonsvo, \
            #     ((0, max_len - len(nonsvo)),(0, max_len - len(nonsvo))), 'constant'))

        # batch_text_indices = pad_sequence(batch_text_indices, batch_first=True, padding_value=config['PAD_token'])

        return { \
                'context': batch_context, \
                'text_indices': torch.tensor(batch_text_indices), \
                # 'svo': torch.tensor(batch_svo), \
                # 'nonsvo': torch.tensor(batch_nonsvo)
            }

    def __iter__(self):
        if self.shuffle:
            random.Random(self.random_seed).shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
