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
import sys
from paddlenlp.metrics import Distinct

import kenlm
from nltk.tokenize.treebank import TreebankWordDetokenizer

ken_model = kenlm.Model("./kenlm/build/snli_text.arpa")

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
from torch.utils.tensorboard import SummaryWriter
from config import config
from generate_dep_matrix import process_snli, process_nonsvo, process_svo

device = config['device_split']
print("Device: {}\n" .format(device))

torch.cuda.empty_cache()

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

fname = './snli_sentences_all.txt'
fin = open(fname, 'r')
snli_data = fin.readlines()

fname_train, fname_val_test = train_test_split(snli_data, train_size=config['train_split'], test_size=config['test_split'], random_state=10)
fname_val, fname_test = train_test_split(fname_val_test, test_size=0.5, random_state=10)

print("train size: {}, val: {}, test: {}\n" .format(len(fname_train), len(fname_val), len(fname_test)))

# fname_train = ["The man makes dinner", "The woman makes dinner"]
# fname_val = ["The man makes dinner", "The woman makes dinner"]
# fname_test = ["The man makes dinner", "The woman makes dinner", "The man makes fish", "The woman makes fish"]

absa_dataset = ABSADatasetReader(config['dataset'], fname_train, fname_val, fname_test, config['train_split'], embed_dim=config['embed_dim'])

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

# train text is not used, as reference is not required for training, only validation and test
# text_train = absa_dataset.text_train

text_val = absa_dataset.text_val
text_test = absa_dataset.text_test

# glob_max_len = len(max(max(text_train), max(text_test), max(text_val)))
glob_max_len = absa_dataset.max_len
glob_max_len += 2

# print("global max sentence len: {}\n" .format(glob_max_len))

# pickling batching not required
# if os.path.exists("train_loader_" + str(config['train_split'])):
#     print("Loading in batched data from pickle\n")
#     with open("train_loader_" + str(config['train_split']), 'rb') as f:
#         train_data_loader = pickle.load(f)
#     f.close()
#     with open("val_loader_" + str(config['train_split']), 'rb') as f:
#         val_data_loader = pickle.load(f)
#     f.close()
#     with open("test_loader_" + str(config['train_split']), 'rb') as f:
#         test_data_loader = pickle.load(f)
#     f.close()

# else:
train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=config['batch_size'], shuffle=False)
val_data_loader = BucketIterator(data=absa_dataset.val_data, batch_size=config['batch_size'], shuffle=False)
test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=config['batch_size'], shuffle=False)

# pickling batching not required
# with open("train_loader_" + str(config['train_split']), "wb") as f:
#     pickle.dump(train_data_loader, f)
# f.close()
# with open("val_loader_" + str(config['train_split']), "wb") as f:
#     pickle.dump(val_data_loader, f)
# f.close()
# with open("test_loader_" + str(config['train_split']), "wb") as f:
#     pickle.dump(test_data_loader, f)
# f.close()

split1_loader = test_data_loader.batches[:test_data_loader.batch_len//2]
split1_text = text_test[:(len(split1_loader)*config['batch_size'])]
split2_loader = test_data_loader.batches[test_data_loader.batch_len//2:]
split2_text = text_test[(len(split2_loader)*config['batch_size']):]

print("split loader lens: {} {} {}\n" .format(test_data_loader.batch_len, len(split1_loader), len(split2_loader)))

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(config['logs_path_config_2_78k'], "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass        

sys.stdout = Logger()

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
        self.gru = nn.GRU(input_size=2*hidden_size + 100, hidden_size=hidden_size, num_layers=config['dec_num_layers'], batch_first=True,  bidirectional=config['dec_bidirectional'])
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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc_mu = nn.Linear(4 * config['hidden_size'], config['vae_latent_dim'])
        self.fc_var = nn.Linear(4 * config['hidden_size'], config['vae_latent_dim'])
        self.max_sample = True

    def get_mu_logvar(self, input):

        mu = self.fc_mu(input)
        log_var = self.fc_var(input)

        return mu, log_var

    def reparameterize(self, mu, log_var):

        # reparameterization trick
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        # temperature reference: self.z_mean + tf.scalar_mul(self.z_temperature, epsilon * tf.exp(self.z_log_sigma))  # N(mu, I * sigma**2)
        z = config['vae_temp'] * eps * std + mu

        return z

    def forward(self, input):
        # Add VAE here

        mu, log_var = self.get_mu_logvar(input)

        z = self.reparameterize(mu, log_var)

        # KL loss
        kld = (-0.5 * torch.sum(log_var - torch.pow(mu, 2) - torch.exp(log_var) + 1, 1)).mean().squeeze()

        return z, kld


def calc_time(start):
    now = time.time()
    end = now - start
    mins = math.floor(end / 60)
    end -= mins*60
    return end, mins

def get_pred_words(total_output):

    # TODO: test get_pred_words

    decoded_words = []
    
    for sentence in total_output:
        preds = []
        for word in sentence:
            # if idx2word[word.item()] != '<pad>':
                # print("word: {}\n" .format(word.item()))
            preds.append(idx2word[word.item()])
            if idx2word[word.item()] == 'EOS':
                break
        # print("preds shape: {}\n" .format(len(preds)))
        decoded_words.append(preds)

    return decoded_words

def format_candidate(candidate):

    new = []
    for x in candidate:
        temp = []
        for word in x:
            if word == 'EOS':
                break
            temp.append(word)
        new.append(temp)

    return new


def calc_bleu(candidate, reference, flag = 'regular'):

    new = format_candidate(candidate)

    # print("candidate: ", new)

    bleu_1 = bleu_score(new, reference, weights=[1, 0, 0, 0])

    if flag == 'split':
        return bleu_1
    else:
        bleu_2 = bleu_score(new, reference, weights=[0.5, 0.5, 0, 0])
        bleu_3 = bleu_score(new, reference, weights=[0.34, 0.33, 0.33, 0])
        bleu_4 = bleu_score(new, reference, weights=[0.25, 0.25, 0.25, 0.25])

    return bleu_1, bleu_2, bleu_3, bleu_4

def calc_ppl(candidate, ken_model):

    ppl = 0

    new = format_candidate(candidate)

    avg_len = len(new)

    for x in new:
        x = TreebankWordDetokenizer().detokenize(x)
        ppl += ken_model.perplexity(x)

    ppl = ppl / avg_len

    return ppl

def calc_distinct(candidate):


    distinct_1 = Distinct(n_size = 1)
    distinct_2 = Distinct(n_size = 2)

    distinct_1_total = 0
    distinct_2_total = 0

    new = format_candidate(candidate)

    for x in new:
        distinct_1.add_inst(x)
        distinct_1_total += distinct_1.score()
        distinct_2.add_inst(x)
        distinct_2_total += distinct_2.score()

    return distinct_1_total, distinct_2_total


# TODO: add distinct1, distinct2, entropy, tree distance edit

def train(input_tensor, target_tensor, encoder, decoder, gcn, vae, encoder_optimizer, decoder_optimizer, gcn_optimizer, vae_optimizer, criterion):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    gcn_optimizer.zero_grad()
    vae_optimizer.zero_grad()

    loss = 0

    gcn_output = gcn(encoder, input_tensor)
    # gcn_output = torch.zeros((target_tensor.size(0), 1, 2*hidden_size), device=device)
    # print("gcn_output: {} size: {}\n" .format(gcn_output, gcn_output.size()))

    z, kld = vae(gcn_output)

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
        # [batch_size, 1, 800]
        decoder_input = torch.cat((decoder_input, z), dim=2)
        # print("dec ip {} size {}" .format(decoder_input, decoder_input.size()))
        # print("z size: {}" .format(z.size()))

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

        try:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        except:
            print("gcn_output: {} size: {}\n" .format(gcn_output, gcn_output.size()))
            print("dec op size", decoder_output, decoder_output.size())
        decoder_output = decoder_output.squeeze(dim=1)        

        loss += criterion(decoder_output, tg)
        loss += kld * config['kld_weight']

        # teacher forcing
        decoder_input = tg_embed
    
    loss.backward()

    # nn.utils.clip_grad_norm_(encoder.parameters(), config['clip_threshold'])
    # nn.utils.clip_grad_norm_(decoder.parameters(), config['clip_threshold'])
    # nn.utils.clip_grad_norm_(gcn.parameters(), config['clip_threshold'])
    # nn.utils.clip_grad_norm(vae.parameters(), config['clip_threshold'])
    encoder_optimizer.step()
    decoder_optimizer.step()
    gcn_optimizer.step()
    vae_optimizer.step()

    return loss.item(), kld

def evaluate(encoder, decoder, gcn, vae, input_tensor, target_tensor, sample_mode):
    with torch.no_grad():

        total_output = torch.zeros((target_tensor.size(0), 1), device=device)
        # total_output = torch.unsqueeze(total_output, dim = 1)

        gcn_output = gcn(encoder, input_tensor)
        # print(gcn_output, gcn_output.size())

        # in case its last batch with different sizes
        if isinstance(gcn_output, list):
            return []

        # VAE Inference, sampling
        # TODO: annealing
        
        # TODO: evaluate/train with enc output instead of gcn output, i.e., use VAE

        # TODO: 10 samples from the posterior
        z = None
        if sample_mode == 'posterior':

            for i in range(config['num_posterior_samples']):
                print("sampling from posterior")
                mu, log_var = vae.get_mu_logvar(gcn_output)
                std = torch.sqrt(torch.exp(log_var))
                z = torch.normal(mean = mu, std = std)

        if sample_mode == 'prior':
            # sample from standard normal
            print("sampling from prior\n")
            mean = torch.zeros([config['batch_size'], 1, config['vae_latent_dim']])
            std = torch.ones([config['batch_size'], 1, config['vae_latent_dim']])
            z = torch.normal(mean = mean, std = std).to(device)
        # print("z inference size: {}" .format(z.size()))
        # z = z.unsqueeze(dim = 1)
        # 250, 1, 200
        # 250, 1, 300

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
            # print(decoder_input.size())
            # print(z.size())
            decoder_input = torch.cat((decoder_input, z), dim=2)
            # print(decoder_input.size())
            # try:
            #     print(decoder_input.size())
            #     print(z.size())
            #     decoder_input = torch.cat((decoder_input, z), dim=2)
            #     print(decoder_input.size())
            # except:
            #     # print(gcn_output, gcn_output.size())
            #     print("dec ip size: {} dec ip: {}" .format(decoder_input.size(), decoder_input))


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

def multi_split(encoder, decoder, gcn, vae, split1_loader, split2_loader, split1_text, split2_text):

    candidate1 = []
    candidate2 = []
    reference1 = []
    reference2 = []

    for item in split1_text:
        reference1.append([item])
    for item in split2_text:
        reference2.append([item])

    print("reference 1 size: {}\n" .format(len(reference1)))
    print("reference 2 size: {}\n" .format(len(reference2)))
    
    print("Calculating candidate...\n")

    for batch1, batch2 in zip(split1_loader, split2_loader):

        # print(len(batch1['text_indices'][0]), len(batch2['text_indices'][0]))
        # if len(batch1['text_indices'][0]) != len(batch2['text_indices'][0]):
        #     continue

        svo1 = process_svo(batch1['context'])
        nonsvo2 = process_nonsvo(batch2['context'])
        svo1 = BucketIterator.pad_graph(svo1, glob_max_len)
        nonsvo2 = BucketIterator.pad_graph(nonsvo2, glob_max_len)

        svo2 = process_svo(batch2['context'])
        nonsvo1 = process_nonsvo(batch1['context'])
        svo2 = BucketIterator.pad_graph(svo2, glob_max_len)
        nonsvo1 = BucketIterator.pad_graph(nonsvo1, glob_max_len)

        input_tensor1 = [batch1['text_indices'].to(device), batch2['text_indices'].to(device), svo1.to(device), nonsvo2.to(device)]
        target_tensor1 = batch1['text_indices'].to(device)

        input_tensor2 = [batch2['text_indices'].to(device), batch1['text_indices'].to(device), svo2.to(device), nonsvo1.to(device)]
        target_tensor2 = batch2['text_indices'].to(device)

        total_output = evaluate(encoder, decoder, gcn, vae, input_tensor1, target_tensor1, config['sample_mode'])
        output_sentences = get_pred_words(total_output)
        candidate1.append(output_sentences)

        total_output = evaluate(encoder, decoder, gcn, vae, input_tensor2, target_tensor2, config['sample_mode'])
        output_sentences = get_pred_words(total_output)
        candidate2.append(output_sentences)

    candidate1 = [val for sublist in candidate1 for val in sublist]
    candidate2 = [val for sublist in candidate2 for val in sublist]

    # add ppl calc here
    ppl_1 = calc_ppl(candidate1, ken_model)
    ppl_2 = calc_ppl(candidate2, ken_model)

    print("ppl 1: {}, ppl: 2: {}\n" .format(ppl_1, ppl_2))

    distinct_1_set_1, distinct_2_set_1 = calc_distinct(candidate1)
    distinct_1_set_2, distinct_2_set_2 = calc_distinct(candidate2)

    print("distinct_1_set_1: {}, distinct_2_set_1: {}\ndistinct_1_set_2: {}, distinct_2_set_2: {}\n" .format(distinct_1_set_1, distinct_2_set_1, distinct_1_set_2, distinct_2_set_2))

    print("candidate 1 size: {}\n" .format(len(candidate1)))
    print("candidate 2 size: {}\n" .format(len(candidate2)))

    # remove last batch to make candidate and reference same
    # drop_last = len(reference2) - len(candidate2)
    # reference1 = reference1[:len(reference1) - config['batch_size']]
    # size of last batch of split_2 text
    # reference2 = reference2[:len(reference2) - drop_last]

    print("reference 1 size: {}\n" .format(len(reference1)))
    print("reference 2 size: {}\n" .format(len(reference2)))

    svo_1 = calc_bleu(candidate1, reference1, flag = 'split')
    nonsvo_1 = calc_bleu(candidate2, reference1, flag = 'split')
    svo_2 = calc_bleu(candidate2, reference2, flag = 'split')
    nonsvo_2 = calc_bleu(candidate1, reference2, flag = 'split')

    print("bleu scores, svo1: {}, nonsvo1: {}, svo2: {}, nonsvo2: {}\n" .format(svo_1, nonsvo_1, svo_2, nonsvo_2))

    print("SVO 1 NONSVO 2\n")
    # print(candidate1)
    choice_indices = np.random.choice(len(candidate1), config['multi_split_samples'], replace=False)
    x = [candidate1[i] for i in choice_indices]
    y = [split1_text[i] for i in choice_indices]
    z = [split2_text[i] for i in choice_indices]
    for i, j, k in zip(x, y, z):
        print("Prediction: {}\nset1 Truth: {}\nset2 Truth: {}\n\n" .format(i, j, k))

    print("SVO 2 NONSVO 1\n")
    # print(candidate2)
    choice_indices = np.random.choice(len(candidate2), config['multi_split_samples'], replace=False)
    x = [candidate2[i] for i in choice_indices]
    y = [split1_text[i] for i in choice_indices]
    z = [split2_text[i] for i in choice_indices]
    for i, j, k in zip(x, y, z):
        print("Prediction: {}\nset1 Truth: {}\nset2 Truth: {}\n\n" .format(i, j, k))

def evaluateTest(encoder, decoder, gcn, vae, test_data_loader, val_data_loader, epoch, total_epochs):

    candidate = []
    reference = []
    ppl_total = 0
    
    print("Calculating candidate...\n")

    if epoch == total_epochs:
        print("Using test loader\n")
        data_loader = test_data_loader
        for item in text_test:
            reference.append([item])
    else:
        print("Using val loader\n")
        data_loader = val_data_loader
        for item in text_val:
            reference.append([item])

    for batch in data_loader:
        svo = process_svo(batch['context'])
        nonsvo = process_nonsvo(batch['context'])
        svo = BucketIterator.pad_graph(svo, glob_max_len)
        nonsvo = BucketIterator.pad_graph(nonsvo, glob_max_len)
        input_tensor = [batch['text_indices'].to(device), svo.to(device), nonsvo.to(device)]
        target_tensor = batch['text_indices'].to(device)

        total_output = evaluate(encoder, decoder, gcn, vae, input_tensor, target_tensor)
        output_sentences = get_pred_words(total_output)
        candidate.append(output_sentences)

    candidate = [val for sublist in candidate for val in sublist]

    ppl = calc_ppl(candidate, ken_model)

    print("perplexity: {}\n" .format(ppl))

    with open(config['dataset'] + '_gcn_candidate.pkl', 'wb') as file:
        pickle.dump(candidate, file)
        print("pickled candidate corpus!!!!")

    print("Reference size: {}\n" .format(len(reference)))
    print("Candidate size: {}\n" .format(len(candidate)))

    # print("candidate: {}\n" .format(candidate))
    # print("\n*******\nreference: {}\n" .format(reference_corpus))
    bleu_1, bleu_2, bleu_3, bleu_4 = calc_bleu(candidate, reference)

    if epoch == total_epochs:
        print("bleu_1: {}, bleu_2: {}, bleu_3: {}, bleu_4: {}\n" .format(bleu_1, bleu_2, bleu_3, bleu_4))
        
        choice_indices = np.random.choice(len(candidate), config['val_samples'], replace=False)
        x = [candidate[i] for i in choice_indices]
        y = [text_test[i] for i in choice_indices]
        for i, j in zip(x, y):
            print("Prediction: {}\nGround Truth: {}\n\n" .format(i, j))

    return bleu_1, bleu_2, bleu_3, bleu_4

def trainIters(encoder, decoder, gcn, vae, encoder_optimizer, decoder_optimizer, gcn_optimizer, vae_optimizer, train_data_loader, current_epochs, total_epochs):

    start = time.time()
    print("start time: {}\n" .format(start))

    loss_total = 0 # Reset every print_every
    kld_total = 0

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    epochs = current_epochs

    for epoch in range(epochs, total_epochs + 1):
        # print("\n\n!!!!!!!!!!!!! IN EPOCH {} !!!!!!!!!\n\n" .format(epoch))
        for num_batch, batch in enumerate(train_data_loader):

            # input_tensor = [batch[col].to(device) for col in input_cols]
            svo = process_svo(batch['context'])
            nonsvo = process_nonsvo(batch['context'])

            svo = BucketIterator.pad_graph(svo, glob_max_len)
            nonsvo = BucketIterator.pad_graph(nonsvo, glob_max_len)
            input_tensor = [batch['text_indices'].to(device), svo.to(device), nonsvo.to(device)]
            target_tensor = batch['text_indices'].to(device)

            # for item in batch['text_indices']:
            #     print(item)

            # print("Split mode\n")

            loss, kld_loss = train(input_tensor, target_tensor, encoder, decoder, gcn, vae, encoder_optimizer, decoder_optimizer, gcn_optimizer, vae_optimizer, criterion)

            loss_total += loss
            kld_total += kld_loss

        if epoch % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'enc_model_state_dict': encoder.state_dict(),
                'dec_model_state_dict': decoder.state_dict(),
                'gcn_model_state_dict': gcn.state_dict(),
                'vae_model_state_dict': vae.state_dict(),
                'enc_optimizer_state_dict': encoder_optimizer.state_dict(),
                'dec_optimizer_state_dict': decoder_optimizer.state_dict(),
                'gcn_optimizer_state_dict': gcn_optimizer.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'loss': loss_total / len(fname_train),
                'kld_loss': kld_total / len(fname_train),
            }, config['save_path'] + '_' + str(epoch))
            print("Saving model at epoch: {}" .format(epoch))

        if epoch % config['validate_every'] == 0:

            encoder.eval()
            gcn.eval()
            decoder.eval()
            vae.eval()

            bleu_1, bleu_2, bleu_3, bleu_4 = evaluateTest(encoder, decoder, gcn, vae, test_data_loader, val_data_loader, epoch, total_epochs)
            print("bleu_1: {}, bleu_2: {}, bleu_3: {}, bleu_4: {}\n" .format(bleu_1, bleu_2, bleu_3, bleu_4))

            with open(config['dataset'] + '_gcn_candidate.pkl', 'rb') as f:
                candidate = pickle.load(f)

            choice_indices = np.random.choice(len(candidate), config['val_samples'], replace=False)
            x = [candidate[i] for i in choice_indices]
            y = [text_val[i] for i in choice_indices]
            for i, j in zip(x, y):
                print("Prediction: {}\nGround Truth: {}\n\n" .format(i, j))

            encoder.train()
            gcn.train()
            decoder.train()

        loss_avg = loss_total / len(fname_train)
        kld_avg = kld_total / len(fname_train)
        writer.add_scalar('loss', loss_avg, epoch)
        writer.add_scalar('Loss/KL', kld_avg, epoch)
        loss_total = 0
        end, mins = calc_time(start)

        print("Epochs: {}, loss avg: {}, mins: {}, secs: {}\n" .format(epoch, loss_avg, mins, end))

# **************************************************************************************************************
# **************************************************************************************************************

encoder = EncoderRNN(config['embed_dim'], config['hidden_size'], config['batch_size'], config['enc_num_layers']).to(device)
gcn = INTERGCN(absa_dataset.embedding_matrix, config['hidden_size']).to(device)
decoder = DecoderRNN(config['vae_latent_dim'], num_words).to(device)
vae = VAE().to(device)
# l2 loss
encoder_params = sum(p.numel() for p in encoder.parameters())
gcn_params = sum(p.numel() for p in gcn.parameters())
decoder_params = sum(p.numel() for p in decoder.parameters())

encoder_trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
gcn_trainable_params = sum(p.numel() for p in gcn.parameters() if p.requires_grad)
decoder_trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

print("Params: {} {} {} {} {} {}" .format(encoder_params, gcn_params, decoder_params, encoder_trainable_params, gcn_params, decoder_params))

encoder_optimizer = optim.Adam(encoder.parameters(), lr=config['learning_rate'])
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config['learning_rate'])
gcn_optimizer = optim.Adam(gcn.parameters(), lr=config['learning_rate'])
vae_optimizer = optim.Adam(vae.parameters(), lr = config['learning_rate'])
epoch = 0

checkpoint = torch.load(config['model_path'], map_location=config['device_split'])
encoder.load_state_dict(checkpoint['enc_model_state_dict'])
decoder.load_state_dict(checkpoint['dec_model_state_dict'])
gcn.load_state_dict(checkpoint['gcn_model_state_dict'])
vae.load_state_dict(checkpoint['vae_model_state_dict'])
encoder_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
decoder_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
gcn_optimizer.load_state_dict(checkpoint['gcn_optimizer_state_dict'])
vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
epoch = checkpoint['epoch']
prev_loss = checkpoint['loss']

print("prev loss: {}\n" .format(prev_loss))
print("starting from epoch: {}\n" .format(epoch + 1))

# encoder.train()
# decoder.train()
# gcn.train()
# vae.train()
# trainIters(encoder, decoder, gcn, vae, encoder_optimizer, decoder_optimizer, gcn_optimizer, vae_optimizer, train_data_loader, current_epochs = epoch + 1, total_epochs = config['total_epochs'])

print("starting evaluation...\n")

encoder.eval()
decoder.eval()
gcn.eval()
vae.eval()

epoch = 100

# evaluateTest(encoder, decoder, gcn, vae, test_data_loader, val_data_loader, epoch, config['total_epochs'])

print("Performing multi split\n")

multi_split(encoder, decoder, gcn, vae, split1_loader, split2_loader, split1_text, split2_text)