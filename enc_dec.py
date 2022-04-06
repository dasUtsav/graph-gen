import torch
from torch import nn
import torch.nn.functional as F
from config import config

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers, embedding_matrix):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
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
    def __init__(self, hidden_size, output_size, embedding_matrix):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
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