# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import enum
import os
import pickle
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
import numpy as np
import json
import re
import spacy
from torch.utils import data
from config import config

nlp = spacy.load("en_core_web_sm")

# def load_word_vec(path, word2idx=None):
#     fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     word_vec = {}
#     for line in fin:
#         tokens = line.rstrip().split()
#         if word2idx is None or tokens[0] in word2idx.keys():
#             try:
#                 word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
#             except:
#                 print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
#     return word_vec

# def build_embedding_matrix(word2idx, embed_dim, type):
#     embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
#     if os.path.exists(embedding_matrix_file_name):
#         print('loading embedding_matrix:', embedding_matrix_file_name)
#         embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
#     else:
#         print('loading word vectors ...')
#         embedding_matrix = np.zeros((len(word2idx), embed_dim))
#         embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
#         # fname = './glove.840B.300d.txt'
#         fname = 'snli_w2v'
#         word_vec = load_word_vec(fname, word2idx=word2idx)
#         print('building embedding_matrix:', embedding_matrix_file_name)
#         for word, i in word2idx.items():
#             vec = word_vec.get(word)
#             if vec is not None:
#                 embedding_matrix[i] = vec
#         pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
#     return embedding_matrix

def build_wordvec_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    print('loading word vectors ...')
    embedding_matrix = np.zeros((len(word2idx), embed_dim))
    embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
    fname = 'snli_w2v'
    snli_w2v = Word2Vec.load(fname)
    word_vec = snli_w2v.wv
    print('building embedding_matrix:', embedding_matrix_file_name)
    for word, i in word2idx.items():
        try:
            vec = word_vec.get_vector(word, norm=True)
        except:
            continue
        if vec is not None:
            embedding_matrix[i] = vec
    pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))

    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['EOS'] = self.idx
            self.idx2word[self.idx] = 'EOS'
            self.idx += 1
            self.word2idx['SOS'] = self.idx
            self.idx2word[self.idx] = 'SOS'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}
            self.idx = len(self.word2idx)

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

        return self.idx

    def text_to_sequence(self, text):
        # text = text.lower()
        # words = text.split()
        unknownidx = config['UNK_token']
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in text]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatasetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:    
            for line in fname:
                line = line.lower().strip()
                line = re.sub(r"([.!?])", r" \1", line)
                line = re.sub(r"[^a-zA-Z.!?]+", r" ", line)
                text += line + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer, split, flag):

        print("Loading in {} data...\n" .format(flag))
        
        # if flag == 'train':
        #     # fin = open('./snli_train_gov_dep.graph', 'rb')
        #     fin = open('snli_' + str(split) + '.graph', 'rb')
        # elif flag == 'test':
        #     fin = open('snli_' + str(split/2) + '.graph', 'rb')
        # graph = pickle.load(fin)
        # fin.close()

        # graph = {}
        all_data = []
        text_total = []
        # graph_id = 0
        for line in fname:
            
            line = line.lower().strip()
            line = re.sub(r"([.!?])", r" \1", line)
            line = re.sub(r"[^a-zA-Z.!?]+", r" ", line)
            # line = line.split(' ')
            line = nlp(line)
            listified = [item.text for item in line]
            text_total.append(listified)

            text_indices = tokenizer.text_to_sequence(listified)
            # dependency_graph = graph[graph_id]

            data = {
                'context': line,
                'text_indices': text_indices,
                # 'dependency_graph': dependency_graph,
            }

            all_data.append(data)
            # graph_id += 1

        return all_data, text_total

    def __init__(self, dataset, train, val, test, split, embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        
        text = ABSADatasetReader.__read_text__([train, val, test])

        # if os.path.exists(dataset+'_gcn_word2idx.pkl'):
        #     print("loading {0} tokenizer...".format(dataset))
        #     with open(dataset+'_gcn_word2idx.pkl', 'rb') as f:
        #         word2idx = pickle.load(f)
        #         self.tokenizer = Tokenizer(word2idx=word2idx)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_text(text)
        # with open(dataset+'_gcn_word2idx.pkl', 'wb') as f:
        #         pickle.dump(self.tokenizer.word2idx, f)
        # with open(dataset + '_gcn_idx2word.pkl', 'wb') as f:
        #     pickle.dump(self.tokenizer.idx2word, f)
        # self.embedding_matrix = build_embedding_matrix(self.tokenizer.word2idx, embed_dim, dataset)
        self.embedding_matrix = build_wordvec_matrix(self.tokenizer.word2idx, embed_dim, dataset)
        self.train_data, self.text_train = ABSADataset(ABSADatasetReader.__read_data__(train, self.tokenizer, split, flag = 'train'))
        self.val_data, self.text_val = ABSADataset(ABSADatasetReader.__read_data__(val, self.tokenizer, split, flag = 'val'))
        self.test_data, self.text_test = ABSADataset(ABSADatasetReader.__read_data__(test, self.tokenizer, split, flag = 'test'))

        
    
