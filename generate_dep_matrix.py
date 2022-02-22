# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import numpy as np
from numpy.matrixlib.defmatrix import matrix
import spacy
import pickle
import os
import re
import json
import spacy
from sklearn.model_selection import train_test_split
import sys
# np.set_printoptions(threshold=sys.maxsize)

nlp = spacy.load("en_core_web_sm")


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("dep_matrix.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass        

sys.stdout = Logger()

def gov_dep_matrix(sentence):

    # words = sentence.split()
    doc = sentence
    # doc = nlp(sentence)
    # print(len(doc), len(words))

    matrix = np.zeros((len(doc), len(doc))).astype('float32')
    
    for token in doc:
        if token.dep_ != 'punct':
            # print("text: {}\npos: {}\nhead text: {}\nhead pos: {}\ndep: {}\nchild: {}\n" .format(token.text, token.pos_, token.head.text, token.head.pos_, token.dep_,
            # [child for child in token.children], token.head.i))
            matrix[token.i][token.head.i] = 1   

            if token.children:
                for child in token.children:
                    if child.dep_ != 'punct':
                        matrix[token.i][child.i] = 1
          
    return matrix

def svo_matrix(sentence):

    doc = sentence
    # doc = nlp(sentence)

    matrix = np.zeros((len(doc), len(doc))).astype('float32')

    # print("\nconfig 1")

    # for token in doc:
    #     if token.dep_ != 'punct' and (token.dep_ == 'nsubj' or token.dep_ == 'dobj' or token.dep_ == 'ROOT'):
    #         matrix[token.i][token.head.i] = 1
    #         print("token: {}\n" .format(token))

    # if token.children:
    #     for child in token.children:
    #         if child.dep_ != 'punct' and (child.dep_ == 'nsubj' or child.dep_ == 'dobj' or child.dep_ == 'ROOT'):
    #             matrix[token.i][child.i] = 1
    #             print("token: {}\nchild: {}\n" .format(token, child))

    # print("\nconfig 2")

    for token in doc:
        # if token.pos_ != 'PUNCT' and (token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'PROPN' or token.pos == 'PRON'):
        if (token.pos_ != 'PUNCT' and (token.dep_ == 'nsubj' or token.dep_ == 'dobj' or token.dep_ == "ROOT")) or (token.pos_ == 'AUX' and token.head.pos_ == 'VERB'):
            matrix[token.i][token.head.i] = 1
            matrix[token.i][token.i] = 1
            # print("token: {}\n" .format(token))

        if token.children:
            if (token.dep_ == 'nsubj' or token.dep_ == 'dobj') and token.head.pos_ == 'VERB':
                for child in token.children:
                    matrix[token.i][child.i] = 1
            for child in token.children:
                if child.pos_ != 'PUNCT' and (child.dep_ == 'nsubj' or child.dep_ == 'dobj' or child.dep_ == 'ROOT'):
                # if child.pos_ != 'PUNCT' and (token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'PROPN' or token.pos == 'PRON'):
                    matrix[token.i][child.i] = 1
                    # print("token: {}\nchild: {}\n" .format(token, child))

    # print("***********")

    # print(matrix)
    return matrix

def nonsvo_matrix(sentence):

    # print("\nnonsvo")

    doc = sentence
    # doc = nlp(sentence)

    matrix = np.zeros((len(doc), len(doc))).astype('float32')

    # print("\nconfig 1")

    # for token in doc:
    #     if token.dep_ != 'punct' and not (token.dep_ == 'nsubj' or token.dep_ == 'dobj' or token.dep_ == 'ROOT'):
    #         matrix[token.i][token.head.i] = 1
    #         print("token: {}\n" .format(token))

    # if token.children:
    #     for child in token.children:
    #         if child.dep_ != 'punct' and not (child.dep_ == 'nsubj' or child.dep_ == 'dobj' or child.dep_ == 'ROOT'):
    #             matrix[token.i][child.i] = 1
    #             print("token: {}\nchild: {}\n" .format(token, child))

    # print("\nconfig 2")

    for token in doc:
        # if token.pos_ != 'PUNCT' and not (token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'PROPN' or token.pos == 'PRON'):
        if (token.pos_ != 'PUNCT' and not (token.dep_ == 'nsubj' or token.dep_ == 'dobj' or token.dep_ == 'ROOT')) and not ((token.pos_ == 'AUX' and token.head.pos_ == 'VERB')):
            matrix[token.i][token.head.i] = 1
            matrix[token.i][token.i] = 1    
            # print("token: {}\n" .format(token))

        if token.children:
            for child in token.children:
                # if child.pos_ != 'PUNCT' and not (token.pos_ == 'ADJ' or token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'PROPN' or token.pos == 'PRON'):
                if child.pos_ != 'PUNCT' and not (child.dep_ == 'nsubj' or child.dep_ == 'dobj' or child.dep_ == 'ROOT'):
                    matrix[token.i][child.i] = 1
                    # print("token: {}\nchild: {}\n" .format(token, child))

    # matrix = np.where((svo == 0), 1, 0)
    # print(matrix)

    return matrix

def process_snli(text):
    
    graph = {}
    graph_idx = 0

    for sentence in text:

        matrix = gov_dep_matrix(sentence)
        graph[graph_idx] = matrix
        graph_idx += 1

    return graph

def process_svo(text):

    graph = {}
    graph_idx = 0

    for sentence in text:

        matrix = svo_matrix(sentence)
        graph[graph_idx] = matrix
        graph_idx += 1

    return graph


def process_nonsvo(text):

    graph = {}
    graph_idx = 0

    for sentence in text:

        matrix = nonsvo_matrix(sentence)
        graph[graph_idx] = matrix
        graph_idx += 1

    return graph

def save_graph(text, split):

    graph = {}
    if os.path.exists('./snli_' + str(split) + '.graph'):
        return None

    fout = open('./snli_' + str(split) + '.graph', 'wb')
    graph_idx = 0

    for sentence in text:
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)
        sentence = sentence.lower().strip()
        matrix = gov_dep_matrix(sentence)
        # print(matrix.shape)
        graph[graph_idx] = matrix
        graph_idx += 1
        if graph_idx % 10000 == 0:
            print(graph_idx)

    pickle.dump(graph, fout)
    print("Saved graph!")
    fout.close()

if __name__ == '__main__':

    fname = './snli_sentences_all.txt'
    fin = open(fname, 'r')
    snli_data = fin.readlines()
    sentences = [s.lower().strip() for s in snli_data]

    nlp = spacy.load("en_core_web_sm")

    x_train, x_val_test = train_test_split(snli_data, test_size=0.1, random_state=10)
    x_val, x_test = train_test_split(x_val_test, test_size=0.5, random_state=10)

    print("{} {} {}\n" .format(len(x_train), len(x_val), len(x_test)))

    test1 = "I wont be able to meet you today"
    test2 = "2 people are standing behind an elderly man"
    test3 = "thats the last public phone in this area"
    test4 = "a man with no apron is smoking a pipe."
    test5 = "a couple is walking in a building."
    test6 = "The group is watching the news"
    test7 = "The children are playing outside"
    test8 = "the man is reading the bible in church"
    test9 = "the two people are surfing"
    test10 = "United canceled the morning flights to Houston"

    test11 = "Some men are talking in front of some graffiti"
    test12 = "A boy makes a peace sign at a protest"

    test13 = "I prefer the morning flight through Denver"

    test14 = "Men buying things from the market"

    # gov_dep_matrix(test9)
    # svo = svo_matrix(test9)
    # nonsvo = nonsvo_matrix(test9)
    # print(svo)
    # print('*******')
    # print(nonsvo)

    # gov_dep_matrix(test14)
    # svo = svo_matrix(test14)
    # nonsvo = nonsvo_matrix(test14)
    # print(svo)
    # print('*******')
    # print(nonsvo)
    gov_dep_matrix(test9)
    svo = svo_matrix(test9)
    nonsvo = nonsvo_matrix(test9)
    print(svo)
    print('*******')
    print(nonsvo)

    # process_snli(x_train, './snli_train_svo.graph')
    # process_snli(x_train, './snli_train_gov_dep.graph')
    # process_snli(x_test, './snli_test_svo.graph')
    # process_snli(x_test, './snli_test_gov_dep.graph')


