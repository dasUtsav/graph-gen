from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import os
import pickle

from numpy.core.defchararray import replace

from data_utils import ABSADatasetReader
from bucket_iterator import BucketIterator

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(5)

# fname_train = '../inter-gcn/con_datasets/rest16_train.raw' # local
# fname_train = './rest16_train.raw' # remote
# fname_test = './rest16_test.raw'
# dataset = 'rest16'
dataset = 'snli'
fname = './snli_sentences_all.txt'
fin = open(fname, 'r')
snli_data = fin.readlines()
fin.close()
fname_train, fname_val_test = train_test_split(snli_data, train_size=0.01, test_size=0.005, random_state=10)
fname_val, fname_test = train_test_split(fname_val_test, test_size=0.5, random_state=10)

embed_dim = 300
hidden_size = 200
batch_size = 100
num_layers = 1
learning_rate = 0.0001
teacher_forcing_ratio = 1
dropout_rate = 0
weight_decay = 0.0001
clip_threshold = 50
total_epochs = 200
model_path = 'model_chkpt_snli_2000.pkl'
save_every = 200
test_every = 10
input_cols = ['text_indices']
train_split = 0.01

absa_dataset = ABSADatasetReader(dataset, fname_train, fname_test, train_split, embed_dim=embed_dim)

num_words = absa_dataset.tokenizer.idx
word2idx = absa_dataset.tokenizer.word2idx
idx2word = absa_dataset.tokenizer.idx2word

# print(word2idx)
# print(idx2word)

embed = nn.Embedding.from_pretrained(torch.tensor(absa_dataset.embedding_matrix, dtype=torch.float)).to(device)
embed_dropout = nn.Dropout(dropout_rate)

SOS_token = word2idx['SOS']
EOS_token = word2idx['EOS']
pad_token = word2idx['<pad>']
print("SOS token: {} EOS: {}\n" .format(SOS_token, EOS_token))

print("num_words: {}\n" .format(num_words))

text_test = absa_dataset.text_test
        
train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=batch_size, shuffle=False)
test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=batch_size, shuffle=False)
print("train set size: {} {}\n" .format(len(fname_train), len(absa_dataset.train_data)))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, dropout_rate):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(absa_dataset.embedding_matrix, dtype=torch.float))
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional = True, batch_first = True, num_layers=3)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, hidden_size)
        self.embed_dropout = nn.Dropout(dropout_rate)

    def forward(self, input, hidden=None):
        output = self.embedding(input)
        output = self.embed_dropout(output)
        # print("output size: {}\n hidden size: {}\n" .format(output.size(), hidden.size()))
        # print("e output size pre gru: {}\n" .format(output.size()))
        # output, (hidden, cell) = self.lstm(output, (hidden, cell))
        if hidden is None:
            output, hidden = self.gru(output, None)
        else:
            output, hidden = self.gru(output, hidden)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(absa_dataset.embedding_matrix, dtype=torch.float))
        self.gru = nn.GRU(input_size=2*hidden_size + 100, hidden_size=hidden_size, batch_first = True, bidirectional=False)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden=None):
        # output = F.relu(input)
        output = input
        # print("gru: {}\n" .format(self.gru))
        # print("d output size pre gru: {}\n" .format(output.size()))
        # print("d hidden: {} size: {}\n" .format(hidden, hidden.size()))
        if hidden is None:
            output, hidden = self.gru(output, None)
        else:
            output, hidden = self.gru(output, hidden)
        # print("d hidden size: {}\n" .format(hidden.size()))
        # print("d output size: {}\n" .format(output.size()))
        output = self.softmax(self.out(output))
        # print("d last output size: {}\n" .format(output.size()))
        return output, hidden

def get_pred_words(total_output):

    decoded_words = []
    
    for sentence in total_output:
        preds = []
        for word in sentence:
            # if idx2word[word.item() == 'EOS']:
            #     break
            preds.append(idx2word[word.item()])
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
            

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):


    # target tensor size: batch_size * seq_len

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    # enc hidden size: num_directions * num_layers, batch_size, hidden_size
    # 2, 100, 100
    # 100, 1, 200

    # 100, 6, 200
    # 100, 1, 200

    encoder_output, encoder_hidden = encoder(input_tensor)

    # enc hidden size: batch_size, 1, hidden_size * num_directions
    # encoder_hidden = encoder_hidden.transpose(0, 1).flatten(start_dim = 1, end_dim = 2).unsqueeze(1)
    # enc hidden size: batch_size, 1, hidden_size
    encoder_hidden = encoder_hidden.transpose(0, 1)
    encoder_hidden = torch.sum(encoder_hidden, dim=1)
    encoder_hidden = encoder_hidden.unsqueeze(dim = 1)
    # print(encoder_hidden.size())


    # target embed size: batch_size * seq_len * embed_dim
    target_embed = embed(target_tensor)

    target_length = target_tensor.size(1)

    decoder_hidden = None
    decoder_input = None

    for i in range(target_length - 1):

        if i == 0:

            # dec ip size: batch_size, 1
            # decoder_input = torch.full(size = (target_tensor.size(0), 1), fill_value = SOS_token, device = device)
            decoder_input = input_tensor.select(dim = 1, index = i).unsqueeze(dim = 1)
            # dec ip size: batch_size, 1, embed_dim
            decoder_input = embed(decoder_input)
            decoder_input = embed_dropout(decoder_input)
    
        # append enc hidden to dec i/p
        decoder_input = torch.cat((decoder_input, encoder_hidden), dim=2)
        # print("dec ip size ", decoder_input.size())

        # np.savetxt('xyz1.txt', decoder_input.cpu().squeeze(dim = 1).detach().numpy())

        # print("dec input size: ", decoder_input.size())
        # ip = decoder_input.select(dim=1, index = i)
        # tg size: batch_size
        tg = target_tensor.select(dim = 1, index = i + 1)
        # tg = target_tensor.select(dim = 1, index = i)
        # tg = tg.unsqueeze(dim=1)
        # print("tg size ", tg.size())

        tg_embed = target_embed.select(dim = 1, index = i + 1)
        # tg embed size: batch_size * 1 * embed_dim
        tg_embed = tg_embed.unsqueeze(dim=1)
        # print("tg embed size: ", tg_embed.size())

        # dec op size: batch_size, 1, V
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        # just to compare
        # topv, topi = decoder_output.data.topk(1)
        # topi = topi.squeeze(dim = -1).detach()
        # topi = topi.squeeze(dim = -1)
        # end of this

        # dec op size: batch_size, V
        decoder_output = decoder_output.squeeze(dim=1)
        # print("dec op size", decoder_output.size())

        # print("tg: {} \ntopi: {}\n" .format(tg, topi))

        # print(idx2word[2906], idx2word[12])

        loss += criterion(decoder_output, tg)

        # teacher forcing
        decoder_input = tg_embed
    
    loss.backward()

    nn.utils.clip_grad_norm_(encoder.parameters(), clip_threshold)
    nn.utils.clip_grad_norm_(decoder.parameters(), clip_threshold)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def evaluate(encoder, decoder, input_tensor):
    with torch.no_grad():

        total_output = torch.zeros((input_tensor.size(0), 1), device = device)
        # print("total output size: ", total_output.size())

        input_length = input_tensor.size(1)

        encoder_output, encoder_hidden = encoder(input_tensor)
        # encoder_hidden = encoder_hidden.transpose(0, 1).flatten(start_dim = 1, end_dim = 2).unsqueeze(1)
        encoder_hidden = encoder_hidden.transpose(0, 1)
        encoder_hidden = torch.sum(encoder_hidden, dim=1)
        encoder_hidden = encoder_hidden.unsqueeze(dim = 1)

        decoder_input = None
        decoder_hidden = None

        for i in range(input_length):

            if i == 0:

                # decoder_input = torch.full(size = (input_tensor.size(0), 1), fill_value = SOS_token, device = device)
                decoder_input = input_tensor.select(dim = 1, index = i).unsqueeze(dim = 1)
                decoder_input = embed(decoder_input)
                decoder_input = embed_dropout(decoder_input)

            else:
                decoder_input = embed(decoder_input)

            decoder_input = torch.cat((decoder_input, encoder_hidden), dim = 2)

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            topi = topi.squeeze(dim = -1).detach()

            total_output = torch.cat((total_output, topi), dim = 1)

            decoder_input = topi

        # total op size: batch_size * seq_len
        total_output = total_output[:, 1:]

        return total_output

def evaluateTest(encoder, decoder, test_data_loader):

    candidate = []
    reference = []
    for item in text_test:
        reference.append([item])

    for batch in test_data_loader:
        input_tensor = batch['text_indices'].to(device)
        total_output = evaluate(encoder, decoder, input_tensor)
        output_sentences = get_pred_words(total_output)

        # print("Shape of output: {} type: {}\n" .format(len(output_sentences), type(output_sentences)))
        # print("Output sentences: {}\n" .format(output_sentences))
        # for item in output_sentences:
        #     for sent in item:
        #         target_idx = sent.index('SOS')
        #         sent = sent[:target_idx]
        #         candidate_corpus.append(sent)
        candidate.append(output_sentences)

    candidate = [val for sublist in candidate for val in sublist] 
        
    print("Reference size: {}\n" .format(len(reference)))
    print("candidate size: {}\n" .format(len(candidate)))

    with open(dataset+'_auto_candidate.pkl', 'wb') as file:
        pickle.dump(candidate, file)
        print("pickled candidate corpus!!!!")

    # print("candidate: {}\n" .format(candidate_corpus))
    # print("\n*******\nreference: {}\n" .format(reference_corpus))
    bleu_1, bleu_2, bleu_3, bleu_4 = calc_bleu(candidate, reference)

    return bleu_1, bleu_2, bleu_3, bleu_4


def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, train_data_loader, current_epochs, total_epochs):
    loss_total = 0 # Reset every print_every

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    epoch = current_epochs

    for _ in range(1, total_epochs + 1):
        for batch in train_data_loader:

            input_tensor = batch['text_indices'].to(device)
            target_tensor = batch['text_indices'].to(device)

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            loss_total += loss

        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'enc_model_state_dict': encoder.state_dict(),
                'dec_model_state_dict': decoder.state_dict(),
                'enc_optimizer_state_dict': encoder_optimizer.state_dict(),
                'dec_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss_total / len(fname_train)
            }, 'model_chkpt_{}_{}.pkl' .format(dataset, epoch))
            print("Saving model at epoch: {}" .format(epoch))

        if epoch % test_every == 0:

            encoder.eval()
            decoder.eval()

            bleu_1, bleu_2, bleu_3, bleu_4 = evaluateTest(encoder, decoder, test_data_loader)
            print("bleu_1: {}, bleu_2: {}, bleu_3: {}, bleu_4: {}\n" .format(bleu_1, bleu_2, bleu_3, bleu_4))

            with open(dataset+'_auto_candidate.pkl', 'rb') as f:
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
        print("Epochs: {}, loss avg: {}\n" .format(epoch, loss_avg))
        epoch += 1

# ******************************************************************************************************************
# ******************************************************************************************************************

encoder = EncoderRNN(embed_dim, hidden_size, num_layers, batch_size, dropout_rate).to(device)
decoder = DecoderRNN(hidden_size, num_words).to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
epoch = 0

# checkpoint = torch.load(model_path)
# encoder.load_state_dict(checkpoint['enc_model_state_dict'])
# decoder.load_state_dict(checkpoint['dec_model_state_dict'])
# encoder_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
# decoder_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
# epoch = checkpoint['epoch']
# prev_loss = checkpoint['loss']

# print("prev loss: {}\n" .format(prev_loss))
# print("starting from epoch: {}\n" .format(epoch + 1))

encoder.train()
decoder.train()

trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, train_data_loader, current_epochs=epoch + 1, total_epochs=total_epochs)

print("starting evaluation...\n")

encoder.eval()
decoder.eval()

bleu_1, bleu_2, bleu_3, bleu_4 = evaluateTest(encoder, decoder, test_data_loader)
print("bleu_1: {}, bleu_2: {}, bleu_3: {}, bleu_4: {}\n" .format(bleu_1, bleu_2, bleu_3, bleu_4))

with open(dataset+'_auto_candidate.pkl', 'rb') as f:
    candidate = pickle.load(f)

count = 0
for x, y in zip(candidate, text_test):
    print("Prediction: {}\nGround Truth: {}\n\n" .format(x, y))
    if count == 10:
        break
    count += 1



# Misc

# np.savetxt('xyz.txt', decoder_input.cpu().squeeze(dim = 1).numpy())