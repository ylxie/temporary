
from __future__ import print_function, division

import torch
import torch.autograd

import datasets
import time
import math

import sklearn.metrics
import numpy as np

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{0}m {1:2.2f}s'.format(m, s)

time_start = time.time()

class LinearRNNModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.cell(combined)
        output = self.output_layer(hidden)
        return output, hidden

    def get_init_hidden(self):
        return torch.autograd.Variable(torch.zeros(1, self.hidden_size))

class VanillaRNNModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaRNNModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, dropout=0, bidirectional=False)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, input, input_lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        out, hiddens = self.rnn(packed_input)
        # unpacked_output, unpacked_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        output = self.output_layer(hiddens)
        return output

class LSTMRNNModule(torch.nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMRNNModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0, bidirectional=False)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, input, input_lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        out, (hidden, _) = self.lstm(packed_input)
        # unpacked_output, unpacked_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        output = self.output_layer(hidden)
        return output

class GRURNNModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRURNNModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0, bidirectional=False)
        self.output_layer = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, input, input_lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        out, hidden = self.gru(packed_input)
        output = self.output_layer(hidden)
        return output

class RNNBinaryClassifierTrainer(object):

    def __init__(self, data_set, hidden_size, batch_size = 1, device='cuda'):
        self.data_set = data_set
        self.batch_size = batch_size
        # self.rnn = RNNModule(data_set.n_words, hidden_size)
        # self.rnn = VanillaRNNModule(data_set.n_words, hidden_size)
        # self.rnn = LSTMRNNModule(data_set.n_words, hidden_size)        
        self.rnn = GRURNNModule(data_set.n_words, hidden_size)        
        if device == 'cuda':
            self.rnn = self.rnn.cuda()
        self.all_losses = []
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=True)
        if device == 'cuda':
            self.criterion = self.criterion.cuda()
        self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr = 0.001, momentum = 0.9)

    def train(self, max_iter, evl_every):
        current_loss = 0
        for i in range(1, max_iter + 1):
            st = time.time()
            for X, L, Y in self.data_set.next_train_batch(self.batch_size):
                    current_loss += self.train_iter(X, L, Y)               
            ed0 = time.time()
            if i % evl_every == 0:
                avg_loss = current_loss / evl_every
                self.all_losses.append(avg_loss)
                # correct_count, total_count = self.evaluate()
                auc = self.evaluate_auc(self.batch_size)
                ed1 = time.time()
                # print("Elapsed: {4}, Iter: {0}, Loss: {1:8.3f}, AUC: {}, Total: {2}, Correct: {3}".format(i, avg_loss, total_count, correct_count, timeSince(time_start), ))
                print("Epoch Used: {0:0.3f}, Eval Used: {1:0.3f}".format(ed0-st, ed1-st))
                print("Elapsed: {3}, Iter: {0}, Loss: {1:8.3f}, AUC: {2:0.3f}".format(i, avg_loss, auc, timeSince(time_start)))
                current_loss = 0


    def train_iter(self, input, input_lengths, target):

        self.optimizer.zero_grad()
        output = self.predict(input, input_lengths)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

        # return loss.data.item()
        # return loss.item()
        return float(loss)

    # def predict(self, input):
    #     hidden = self.rnn.get_init_hidden()
    #     for i in range(len(input)):
    #         output, hidden = self.rnn(input[i], hidden)
    #     return output

    def predict(self, input, input_lengths):
        output = self.rnn(input, input_lengths)
        return output[0]

    def evaluate(self, threshold = 0, batch_size=1):
        correct_count = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for (X, L, Y) in self.data_set.next_test_batch(batch_size):
            output = self.predict(X,L)
            for i in range(len(X)):
                if output.data[i,0] > threshold:
                    correct_count += 1 - Y.data[i,0]
                    if Y.data[i,0] > 0:
                        TP += 1
                    else:
                        FP += 1
                else:
                    correct_count += Y.data[i,0]
                    if Y.data[i,0] > 0:
                        FN += 1
                    else:
                        TN += 1
        print("TP: {0}, FP: {1}, TN: {2}, FN: {3}".format(TP, FP, TN, FN))
        return correct_count, len(self.data_set.test_set)

    def evaluate_auc(self, batch_size=1):
        scores = np.zeros(self.data_set.n_test_samples)
        labels = np.zeros(self.data_set.n_test_samples)
        idx = 0
        for (X,L,Y)  in self.data_set.next_test_batch(batch_size):
            output = self.predict(X,L)
            for i in range(len(X)):
                scores[idx] = round(output.data[i,0].item(), 5)
                labels[idx] = Y.data[i,0]
                idx += 1

        precions, recalls, thresholds = sklearn.metrics.precision_recall_curve(labels, scores)
        auc = sklearn.metrics.auc(recalls, precions)
        return auc



