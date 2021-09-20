from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import os
import pickle

from intergcn import INTERGCN

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print("Loading embedding matrix: {}\n" .format(embedding_matrix_file_name))
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print("loading word vectors...\n")
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        # fname = '../../manulife-project-2/absa/inter-gcn-absa/glove.42B.300d.txt' # local
        fname = './glove.42B.300d.txt' # remote
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print("building embedding_matrix: {}\n" .format(embedding_matrix_file_name))
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {0: "SOS", 1: "EOS"}
            # self.idx2word = {}
            # self.idx  = 0
            self.idx = 2
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

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
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

def read_text(fname):
    text = ""
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(0, len(lines)):
        _, _, _, text_raw = lines[i].split('\t')
        text_raw = text_raw.lower().strip()
        # text_raw = " ".join(text_raw.split())
        # text_raw = text_raw.replace("'", "")
        # text_raw = text_raw.replace('"', "")
        text_raw = re.sub(r"([.!?])", r" \1", text_raw)
        text_raw = re.sub(r"[^a-zA-Z.!?]+", r" ", text_raw)
        text += text_raw + " "

    return text

def get_list(fname):
    text_total = []
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(len(lines)):
        _, _, _, text = lines[i].split('\t')
        text = text.lower().strip()
        # text = text.replace("'", "")
        # text = text.replace('"', "")
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        text = text.split(' ')
        text_total.append(text)

    return text_total

fname_train = '../inter-gcn/con_datasets/rest16_train.raw' # local
fname_train = './rest16_train.raw' # remote
fname_test = './rest16_test.raw'
dataset = 'rest16'
# fname = './snli_sentences_all.txt'
# fin = open(fname, 'r')
# snli_data = fin.readlines()
# fname_train, fname_val_test = train_test_split(snli_data, test_size=0.1, random_state=10)
# fname_val, fname_test = train_test_split(fname_val_test, test_size=0.5, random_state=10)

embed_dim = 300
hidden_size = 100
text = read_text(fname_train)
# print("here: {}\n" .format(text))
text_train = get_list(fname_train)
text_test = get_list(fname_test)

print("len text total: {}\n" .format(len(text_train)))

max_length = 0
for i in range(len(text_train)):
    # print(text_train[i])
    # print(len(text_train[i]))
    if len(text_train[i]) > max_length:
        max_length = len(text_train[i])

max_length += 1

print("Max length of sentence in dataset: {}\n" .format(max_length))

num_words = 0

def indexesFromSentence(word2idx, sentence):
    return [word2idx[word] for word in sentence if word in word2idx]

def tensorFromSentence(word2idx, sentence):
    indexes = indexesFromSentence(word2idx, sentence)
    padding = [0] * (max_length - len(indexes))
    indexes += padding
    # indexes.append(EOS_token)
    return torch.tensor(indexes, device=device)

tokenizer = Tokenizer()
num_words = tokenizer.fit_on_text(text)
with open(dataset+'_word2idx.pkl', 'wb') as f:
    pickle.dump(tokenizer.word2idx, f)
with open(dataset + '_idx2word.pkl', 'wb') as f:
    pickle.dump(tokenizer.idx2word, f)
print("loading {0} tokenizer...\n" .format(dataset))
with open(dataset+'_word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open(dataset+'_idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

print("Num words: {}\n" .format(num_words))
embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
print("Embedding matrix shape: {}\n" .format(embedding_matrix.shape))

train_indices = [tensorFromSentence(word2idx, sentence) for sentence in text_train]
test_indices = [tensorFromSentence(word2idx, sentence) for sentence in text_test]

# print("Random train sentence: {} size: {} total len: {}\n" .format(train_indices[10], len(train_indices[10]), len(train_indices)))

train_data_loader = DataLoader(train_indices, batch_size=100, shuffle=True)
test_data_loader = DataLoader(test_indices, batch_size=100)

# print(train_data_loader)

# for batch in train_data_loader:
#     print("batch i : {}\n size i: {}\n len: {}\n" .format(batch[0], batch.size(), len(batch[0])))
#     exit()



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional = True, batch_first=True)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, hidden_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input)
        output = embedded
        # print("output size: {}\n hidden size: {}\n" .format(output.size(), hidden.size()))
        # print("e output size pre gru: {}\n" .format(output.size()))
        # output, (hidden, cell) = self.lstm(output, (hidden, cell))
        if hidden is None:
            output, hidden = self.gru(output, None)
        else:
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, self.hidden_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.gru = nn.GRU(input_size=2*hidden_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.out = nn.Linear(2*hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden=None):
        output = F.relu(input)
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

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=max_length):

    loss = 0
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    sentence_length = target_tensor.size(1)

    # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # print("input tensor size: {} hidden: {} length: {}\n" .format(input_tensor.size(), encoder_hidden.size(), input_length))
    encoder_output, encoder_hidden = encoder(input_tensor)
    # print("Encoder outputs size: {}\n" .format(encoder_output.size()))

    # decoder_input = torch.tensor([[SOS_token]], device=device).view(1, 1, 1)
    decoder_input = encoder_output
    # decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: feed the target as the next input

        decoder_output, decoder_hidden = decoder(decoder_input)

        for sentence, target_sentence in zip(decoder_output, target_tensor):        
            # print("Decoder output size: {} type: {}\n" .format(decoder_output.size(), decoder_output.dtype))
            # topv, topi = decoder_output.topk(1)
            # target_tensor = target_tensor.float()
            # topi = topi.squeeze().detach() # detach from history as input
            # print("target tensor size: {} type: {}\n" .format(target_tensor.size(), target_tensor.dtype))
            loss += criterion(sentence.float(), target_sentence)

        decoder_input = target_tensor # teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_output, decoder_hidden = decoder(decoder_input)
        
        for sentence, target_sentence in zip(decoder_output, target_tensor):        
            # print("Decoder output size: {} type: {}\n" .format(decoder_output.size(), decoder_output.dtype))
            # print("Decoder sentence size: {} type: {}\n" .format(sentence.size(), sentence.dtype))
            # topv, topi = decoder_output.topk(1)
            # print(topi)
            # decoder_input = topi.squeeze().detach() # detach from history as input
            # print("Decoder input size: {} type: {}\n" .format(decoder_output.size(), decoder_output.dtype))
            # print("target tensor size: {} type: {}\n" .format(target_sentence.size(), target_sentence.dtype))
            
            loss += criterion(sentence.float(), target_sentence)
            # print(decoder_output)
            # if decoder_input.item() == EOS_token:
            #     break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, current_epochs, total_epochs, learning_rate=0.01):
    loss_total = 0 # Reset every print_every

    criterion = nn.CrossEntropyLoss()
    epoch = current_epochs

    for _ in range(1, total_epochs + 1):
        for batch in train_data_loader:

            input_tensor = batch
            target_tensor = batch

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            loss_total += loss

        if epoch % 1000 == 0:
            torch.save({
                'epoch': epoch,
                'enc_model_state_dict': encoder.state_dict(),
                'dec_model_state_dict': decoder.state_dict(),
                'enc_optimizer_state_dict': encoder_optimizer.state_dict(),
                'dec_optimizer_state_dict': decoder_optimizer.state_dict(),
                'loss': loss_total / len(train_data_loader)
            }, 'model_chkpt_{}_{}.pkl' .format(dataset, epoch))
            print("Saving model at epoch: {}" .format(epoch))

        loss_avg = loss_total / len(train_data_loader)
        loss_total = 0
        print("Epochs: {}, loss avg: {}\n" .format(epoch, loss_avg))
        epoch += 1
            

def evaluate(encoder, decoder, input_tensor, max_length=max_length):
    with torch.no_grad():

        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        encoder_output, encoder_hidden = encoder(input_tensor)

        decoder_input = encoder_output

        decoded_words = []

        decoder_output, decoder_hidden = decoder(
            decoder_input)
        topv, topi = decoder_output.data.topk(1)
        topi = topi.squeeze().detach()

        # print("top i size: {} type: {}\n" .format(topi.size(), type(topi)))
        
        for sentence in topi:
            preds = []
            for word in sentence:
                if idx2word[word.item()] != 'SOS':
                    preds.append(idx2word[word.item()])
            # print("intermed: {}\n" .format(intermed))
            # print("preds shape: {}\n" .format(len(preds)))
            decoded_words.append(preds)

        print("decoded words size: {}" .format(len(decoded_words)))

        decoder_input = topi

        return decoded_words

# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         sentence = random.choice(text_train)
#         print('>', sentence)
#         output_words = evaluate(encoder, decoder, sentence)
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')

def evaluateTest(encoder, decoder):

    candidate_corpus = []
    reference_corpus = []
    for item in text_test:
        reference_corpus.append([item])

    for batch in test_data_loader:
        input_tensor = batch
        output_sentences = evaluate(encoder, decoder, input_tensor)
        # print("Shape of output: {} type: {}\n" .format(len(output_sentences), type(output_sentences)))
        # print("Output sentences: {}\n" .format(output_sentences))
        # for item in output_sentences:
        #     for sent in item:
        #         target_idx = sent.index('SOS')
        #         sent = sent[:target_idx]
        #         candidate_corpus.append(sent)
        candidate_corpus.append(output_sentences)

    candidate_corpus = [val for sublist in candidate_corpus for val in sublist] 
        
    print("Reference size: {}\n" .format(len(reference_corpus)))
    print("candidate size: {}\n" .format(len(candidate_corpus)))

    with open(dataset+'_candidate.pkl', 'wb') as file:
        pickle.dump(candidate_corpus, file)
        print("pickled candidate corpus!!!!")

    # print("candidate: {}\n" .format(candidate_corpus))
    # print("\n*******\nreference: {}\n" .format(reference_corpus))
    bleu_1 = bleu_score(candidate_corpus, reference_corpus, weights=[1, 0, 0, 0])
    bleu_2 = bleu_score(candidate_corpus, reference_corpus, weights=[0, 1, 0, 0])
    bleu_3 = bleu_score(candidate_corpus, reference_corpus, weights=[0, 0, 1, 0])
    bleu_4 = bleu_score(candidate_corpus, reference_corpus, weights=[0, 0, 0, 1])

    return bleu_1, bleu_2, bleu_3, bleu_4


learning_rate = 0.01
gcn = INTERGCN(embedding_matrix, hidden_size)
encoder = EncoderRNN(embed_dim, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, num_words).to(device)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
epoch = 1

checkpoint = torch.load('model_chkpt_rest16_200.pkl')
encoder.load_state_dict(checkpoint['enc_model_state_dict'])
decoder.load_state_dict(checkpoint['dec_model_state_dict'])
encoder_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
decoder_optimizer.load_state_dict(checkpoint['dec_optimizer_state_dict'])
epoch = checkpoint['epoch']
prev_loss = checkpoint['loss']

print("prev loss: {}\n" .format(prev_loss))
print("starting from epoch: {}\n" .format(epoch + 1))

encoder.train()
decoder.train()
trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, current_epochs=epoch + 1, total_epochs=800, learning_rate=learning_rate)

encoder.eval()
decoder.eval()
# evaluateRandomly(encoder1, decoder1)

# print("idxword: {}\n" .format(idx2word))

bleu_1, bleu_2, bleu_3, bleu_4 = evaluateTest(encoder, decoder)
print("bleu_1: {}, bleu_2: {}, bleu_3: {}, bleu_4: {}\n" .format(bleu_1, bleu_2, bleu_3, bleu_4))

with open(dataset+'_candidate.pkl', 'rb') as f:
    candidate = pickle.load(f)

count = 0
for x, y in zip(candidate, text_test):
    print("Prediction: {}\nGround Truth: {}\n\n" .format(x, y))
    if count == 10:
        break
    count += 1