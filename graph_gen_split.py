from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import os
import pickle
import time
import math

from intergcn_split import INTERGCN
from data_utils_split import ABSADatasetReader
from bucket_iterator_split import BucketIterator

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from config import config
from generate_dep_matrix import process_snli, process_nonsvo, process_svo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fname = './snli_sentences_all.txt'
fin = open(fname, 'r')
snli_data = fin.readlines()

fname_train, fname_val_test = train_test_split(snli_data, train_size=config['train_split'], test_size=config['test_split'], random_state=10)
fname_val, fname_test = train_test_split(fname_val_test, test_size=0.5, random_state=10)

absa_dataset = ABSADatasetReader(config['dataset'], fname_train, fname_test, config['train_split'], embed_dim=config['embed_dim'])

num_words = absa_dataset.tokenizer.idx
word2idx = absa_dataset.tokenizer.word2idx
idx2word = absa_dataset.tokenizer.idx2word
embed = nn.Embedding.from_pretrained(torch.tensor(absa_dataset.embedding_matrix, dtype=torch.float)).to(device)
embed_dropout = nn.Dropout(config['dropout_rate'])
        

# print("test shuffle: {}\n" .format(test_data_loader.shuffle))

SOS_token = config['SOS_token']
EOS_token = config['EOS_token']
pad_token = config['PAD_token']
# print("SOS token: {} EOS: {}\n" .format(SOS_token, EOS_token))

print("num_words: {}\n" .format(num_words))

text_test = absa_dataset.text_test

train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=config['batch_size'], shuffle=False)
test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=config['batch_size'], shuffle=False)

# print("word2idx: {}\n" .format(word2idx))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(absa_dataset.embedding_matrix, dtype=torch.float))
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional = config['enc_bidirectional'])
        # self.lstm = nn.LSTM(hidden_size, hidden_size, hidden_size)

    def forward(self, input, input_len, hidden=None):
        # output = torch.nn.utils.rnn.pack_padded_sequence(input, input_len, batch_first=True, enforce_sorted=False)
        output = input
        if hidden is None:
            output, hidden = self.gru(output, None)
        else:
            output, hidden = self.gru(output, hidden)

        # output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # output = output[0]
        # print("post e: {}\n" .format(output.shape))
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(absa_dataset.embedding_matrix, dtype=torch.float))
        self.gru = nn.GRU(input_size=5*hidden_size + 100, hidden_size=hidden_size, num_layers=config['dec_num_layers'], batch_first=True,  bidirectional=config['dec_bidirectional'])
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden=None):
        output = F.relu(input)
        # print("gru: {}\n" .format(self.gru))
        # print("out: {}\n" .format(self.out))
        # print("d output size pre gru: {}\n" .format(output.size()))
        if hidden is None:
            output, hidden = self.gru(output, None)
        else:
            output, hidden = self.gru(output, hidden)
        # print("d hidden size: {}\n" .format(hidden.size()))
        # print("d output size: {}\n" .format(output.size()))
        # sleep(5)
        output = self.softmax(self.out(output))
        # print("d last output {} size: {}\n" .format(output, output.size()))
        # sleep(5)
        return output, hidden

def calc_time(start):
    now = time.time()
    end = now - start
    mins = math.floor(end / 60)
    end -= mins*60
    return end, mins

def get_pred_words(total_output):

    decoded_words = []
    
    for sentence in total_output:
        preds = []
        for word in sentence:
            # if idx2word[word.item()] != '<pad>':
                # print("word: {}\n" .format(word.item()))
            preds.append(idx2word[word.item()])
        # print("preds shape: {}\n" .format(len(preds)))
        decoded_words.append(preds)

    return decoded_words

def calc_bleu(candidate, reference):

    new = []
    for x in candidate:
        temp = []
        for word in x:
            if word == 'EOS':
                break
            temp.append(word)
        new.append(temp)

    # print("candidate: ", new)

    bleu_1 = bleu_score(new, reference, weights=[1, 0, 0, 0])
    bleu_2 = bleu_score(new, reference, weights=[0.5, 0.5, 0, 0])
    bleu_3 = bleu_score(new, reference, weights=[0.34, 0.33, 0.33, 0])
    bleu_4 = bleu_score(new, reference, weights=[0.25, 0.25, 0.25, 0.25])

    return bleu_1, bleu_2, bleu_3, bleu_4

def train(input_tensor, target_tensor, encoder, decoder, gcn, encoder_optimizer, decoder_optimizer, gcn_optimizer, criterion):

    # TODO: validation dataset after each epoch

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    gcn_optimizer.zero_grad()

    loss = 0

    gcn_output = gcn(encoder, input_tensor)
    # gcn_output = torch.zeros((target_tensor.size(0), 1, 2*hidden_size), device=device)
    # print("gcn_output: {} size: {}\n" .format(gcn_output, gcn_output.size()))

    target_embed = embed(target_tensor)

    target_length = target_tensor.size(1)

    decoder_hidden = None
    decoder_input = None

    for i in range(target_length - 1):

        if i == 0:

            # decoder_input = torch.full(size = (target_tensor.size(0), 1), fill_value = SOS_token, device = device)
            decoder_input = target_tensor.select(dim = 1, index = i).unsqueeze(dim = 1)
            decoder_input = embed(decoder_input)
            decoder_input = embed_dropout(decoder_input)

            # print("dec ip size: {} dec ip: {}" .format(decoder_input.size(), decoder_input))

        # print("dec ip {} size {}" .format(decoder_input, decoder_input.size()))
        # print("gcn_output: {} size: {}\n" .format(gcn_output, gcn_output.size()))
        decoder_input = torch.cat((decoder_input, gcn_output), dim=2)


        # ip = decoder_input.select(dim=1, index = i)
        # for loss
        tg = target_tensor.select(dim=1, index=i + 1)
        # tg = tg.unsqueeze(dim=1)
        # print("tg: {}" .format(tg.size()))
        # print("tg {} size {} " .format(tg, tg.size()))

        # for teacher forcing
        tg_embed = target_embed.select(dim = 1, index = i + 1)
        tg_embed = tg_embed.unsqueeze(dim=1)
        # print("tg embed {} size {} " .format(tg_embed, tg_embed.size()))

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_output = decoder_output.squeeze(dim=1)
        # print("dec op size", decoder_output, decoder_output.size())

        loss += criterion(decoder_output, tg)

        # teacher forcing
        decoder_input = tg_embed
    
    loss.backward()

    nn.utils.clip_grad_norm_(encoder.parameters(), config['clip_threshold'])
    nn.utils.clip_grad_norm_(decoder.parameters(), config['clip_threshold'])
    nn.utils.clip_grad_norm_(gcn.parameters(), config['clip_threshold'])
    encoder_optimizer.step()
    decoder_optimizer.step()
    gcn_optimizer.step()

    return loss.item()

def evaluate(encoder, decoder, gcn, input_tensor, target_tensor):
    with torch.no_grad():

        total_output = torch.zeros((target_tensor.size(0), 1), device=device)
        # total_output = torch.unsqueeze(total_output, dim = 1)

        gcn_output = gcn(encoder, input_tensor)

        target_length = target_tensor.size(1)

        decoder_hidden = None
        decoder_input = None

        for i in range(target_length):

            if i == 0:

                # decoder_input = torch.full(size = (target_tensor.size(0), 1), fill_value = SOS_token, device = device)
                decoder_input = target_tensor.select(dim = 1, index = i).unsqueeze(dim = 1)
                decoder_input = embed(decoder_input)
                decoder_input = embed_dropout(decoder_input)

            else:
                decoder_input = embed(decoder_input)
            
            # print("dec ip size: {} dec ip: {}" .format(decoder_input.size(), decoder_input))
            decoder_input = torch.cat((decoder_input, gcn_output), dim=2)

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # print("dec op size: {} dec op: {}" .format(decoder_output.size(), decoder_output))

            topv, topi = decoder_output.data.topk(1)
            # print("top i: {} size: {}" .format(topi, topi.size()))
            topi = topi.squeeze(dim = -1).detach()
            # print("top i: {} size: {}" .format(topi, topi.size()))
            # decoder_input = decoder_input.type(torch.float)

            total_output = torch.cat((total_output, topi), dim = 1)

            decoder_input = topi
            # print("dec ip eval", decoder_input.size())

        total_output = total_output[:, 1:]
        # print("total op size: {}" .format(total_output.size()))
       
        return total_output

def evaluateTest(encoder, decoder, gcn, test_data_loader):

    candidate = []
    reference = []
    for item in text_test:
        reference.append([item])
    
    print("Calculating candidate...\n")
    for batch in test_data_loader:
        svo = process_svo(batch['context'])
        nonsvo = process_nonsvo(batch['context'])
        svo = BucketIterator.pad_graph(svo, batch['max_len'])
        nonsvo = BucketIterator.pad_graph(nonsvo, batch['max_len'])
        input_tensor = [batch['text_indices'].to(device), svo.to(device), nonsvo.to(device)]
        target_tensor = batch['text_indices'].to(device)

        total_output = evaluate(encoder, decoder, gcn, input_tensor, target_tensor)
        output_sentences = get_pred_words(total_output)
        candidate.append(output_sentences)

    candidate = [val for sublist in candidate for val in sublist]

    with open(config['dataset'] + '_gcn_candidate.pkl', 'wb') as file:
        pickle.dump(candidate, file)
        print("pickled candidate corpus!!!!")

    print("Reference size: {}\n" .format(len(reference)))
    print("candidate size: {}\n" .format(len(candidate)))

    # print("candidate: {}\n" .format(candidate))
    # print("\n*******\nreference: {}\n" .format(reference_corpus))
    bleu_1, bleu_2, bleu_3, bleu_4 = calc_bleu(candidate, reference)

    return bleu_1, bleu_2, bleu_3, bleu_4

def trainIters(encoder, decoder, gcn, encoder_optimizer, decoder_optimizer, gcn_optimizer, train_data_loader, current_epochs, total_epochs):

    start = time.time()
    print("start time: {}\n" .format(start))

    loss_total = 0 # Reset every print_every

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    epochs = current_epochs

    for epoch in range(epochs, total_epochs + 1):
        # print("\n\n!!!!!!!!!!!!! IN EPOCH {} !!!!!!!!!\n\n" .format(epoch))
        for num_batch, batch in enumerate(train_data_loader):

            # input_tensor = [batch[col].to(device) for col in input_cols]
            svo = process_svo(batch['context'])
            nonsvo = process_nonsvo(batch['context'])

            svo = BucketIterator.pad_graph(svo, batch['max_len'])
            nonsvo = BucketIterator.pad_graph(nonsvo, batch['max_len'])
            input_tensor = [batch['text_indices'].to(device), svo.to(device), nonsvo.to(device)]
            target_tensor = batch['text_indices'].to(device)

            # for item in batch['text_indices']:
            #     print(item)

            loss = train(input_tensor, target_tensor, encoder, decoder, gcn, encoder_optimizer, decoder_optimizer, gcn_optimizer, criterion)

            loss_total += loss

        if epoch % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'enc_model_state_dict': encoder.state_dict(),
                'dec_model_state_dict': decoder.state_dict(),
                'gcn_model_state_dict': gcn.state_dict(),
                'enc_optimizer_state_dict': encoder_optimizer.state_dict(),
                'dec_optimizer_state_dict': decoder_optimizer.state_dict(),
                'gcn_optimizer_state_dict': gcn_optimizer.state_dict(),
                'loss': loss_total / len(fname_train)
            }, 'model_chkpt_split_{}_{}.pkl' .format(config['dataset'], epoch))
            print("Saving model at epoch: {}" .format(epoch))

        if epoch % config['validate_every'] == 0:

            encoder.eval()
            decoder.eval()

            bleu_1, bleu_2, bleu_3, bleu_4 = evaluateTest(encoder, decoder, gcn, test_data_loader)
            print("bleu_1: {}, bleu_2: {}, bleu_3: {}, bleu_4: {}\n" .format(bleu_1, bleu_2, bleu_3, bleu_4))

            with open(config['dataset'] + '_gcn_candidate.pkl', 'rb') as f:
                candidate = pickle.load(f)

            choice_indices = np.random.choice(len(candidate), 10, replace=False)
            x = [candidate[i] for i in choice_indices]
            y = [text_test[i] for i in choice_indices]
            for i, j in zip(x, y):
                print("Prediction: {}\nGround Truth: {}\n\n" .format(i, j))

            encoder.train()
            decoder.train()

        loss_avg = loss_total / len(fname_train)
        loss_total = 0
        end, mins = calc_time(start)

        print("Epochs: {}, loss avg: {}, mins: {}, secs: {}\n" .format(epoch, loss_avg, mins, end))

# **************************************************************************************************************
# **************************************************************************************************************

# TODO: test get_pred_words

encoder = EncoderRNN(config['embed_dim'], config['hidden_size'], config['batch_size'], config['enc_num_layers']).to(device)
gcn = INTERGCN(absa_dataset.embedding_matrix, config['hidden_size']).to(device)
decoder = DecoderRNN(config['hidden_size'], num_words).to(device)
# l2 loss
encoder_optimizer = optim.Adam(encoder.parameters(), lr=config['learning_rate'])
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config['learning_rate'])
gcn_optimizer = optim.Adam(gcn.parameters(), lr=config['learning_rate'])
epoch = 0

# checkpoint = torch.load(config['model_path'])
# encoder.load_state_dict(checkpoint['enc_model_state_dict'])
# decoder.load_state_dict(checkpoint['dec_model_state_dict'])
# gcn.load_state_dict(checkpoint['gcn_model_state_dict'])
# encoder_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
# decoder_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
# gcn_optimizer.load_state_dict(checkpoint['gcn_optimizer_state_dict'])
# epoch = checkpoint['epoch']
# prev_loss = checkpoint['loss']

# print("prev loss: {}\n" .format(prev_loss))
# print("starting from epoch: {}\n" .format(epoch + 1))

encoder.train()
decoder.train()
gcn.train()
trainIters(encoder, decoder, gcn, encoder_optimizer, decoder_optimizer, gcn_optimizer, train_data_loader, current_epochs = epoch + 1, total_epochs = config['total_epochs'])

print("starting evaluation...\n")

encoder.eval()
decoder.eval()
gcn.eval()

print("word2idx: {}\n" .format(word2idx['a']))

bleu_1, bleu_2, bleu_3, bleu_4 = evaluateTest(encoder, decoder, gcn, test_data_loader)
print("bleu_1: {}, bleu_2: {}, bleu_3: {}, bleu_4: {}\n" .format(bleu_1, bleu_2, bleu_3, bleu_4))

with open(config['dataset'] + '_gcn_candidate.pkl', 'rb') as f:
    candidate = pickle.load(f)

count = 0
for x, y in zip(candidate, text_test):
    print("Prediction: {}\nGround Truth: {}\n\n" .format(x, y))
    if count == 10:
        break
    count += 1