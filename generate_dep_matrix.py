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

def gov_dep_matrix(sentence):

    # words = sentence.split()
    # doc = nlp(sentence)
    doc = sentence
    # print(len(doc), len(words))

    matrix = np.zeros((len(doc), len(doc))).astype('float32')
    
    for token in doc:
        if token.dep_ != 'punct':
            # print("text: {}\npos: {}\nhead text: {}\nhead pos: {}\nchild: {}\n" .format(token.text, token.pos_, token.head.text, token.head.pos_,
            # [child for child in token.children], token.head.i))
            matrix[token.i][token.head.i] = 1   

            if token.children:
                for child in token.children:
                    if child.dep_ != 'punct':
                        matrix[token.i][child.i] = 1
          
    return matrix

def svo_matrix(sentence):

    doc = sentence

    matrix = np.zeros((len(doc), len(doc))).astype('float32')

    for token in doc:
        if token.dep_ != 'punct' or token.dep_ == 'nsubj' or token.dep_ == 'dobj':
            matrix[token.i][token.head.i] = 1       

        if token.children:
            for child in token.children:
                if child.dep_ != 'punct' or token.dep_ == 'nsubj' or token.dep_ == 'dobj':
                    matrix[token.i][child.i] = 1

    return matrix

def nonsvo_matrix(sentence):

    doc = sentence

    matrix = np.zeros((len(doc), len(doc))).astype('float32')

    for token in doc:
        if token.dep_ != 'punct' or token.dep_ != 'nsubj' or token.dep_ != 'dobj':
            matrix[token.i][token.head.i] = 1   

        if token.children:
            for child in token.children:
                if child.dep_ != 'punct' or token.dep_ != 'nsubj' or token.dep_ != 'dobj':
                    matrix[token.i][child.i] = 1

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

    gov_dep_matrix(test1)

    sentences = [test1, test3]

    # process_snli(x_train, './snli_train_svo.graph')
    # process_snli(x_train, './snli_train_gov_dep.graph')
    # process_snli(x_test, './snli_test_svo.graph')
    # process_snli(x_test, './snli_test_gov_dep.graph')


