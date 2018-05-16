import torch
import torch.autograd

import random

random.seed(1)
torch.manual_seed(1)

class DataGenerator:
    def __init__(self, feature_dim, max_seq_len, num_train_samples):
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.num_train_samples = num_train_samples
    
    def get_next_batch(self, batch_size=1, num_batches=10):                
        for b in range(num_batches):
            samples = []
            for i in range(batch_size):
                length = random.randint(self.max_seq_len - 100, self.max_seq_len)
                features = [self.create_feature() for j in range(length)]
                target = random.randint(0, 1)
                samples.append((features, target))
            yield self.create_padded_batch(samples)
    
    def create_feature(self):
        tensor = torch.zeros(1, self.feature_dim)
        d = random.randint(0, self.feature_dim - 1)
        tensor[0][d] = 1.0
        return tensor

    def create_padded_batch(self, samples):
        sorted_samples = sorted(samples, key=lambda x:len(x[0]), reverse=True)
        inputs = [torch.cat(x[0], dim=0) for x in sorted_samples]
        input_lengths = [len(x) for x in inputs]
        targets = [x[1] for x in sorted_samples]
        target_batch = torch.FloatTensor(targets).view(-1,1)
        packed_input = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        return packed_input, input_lengths, target_batch


class LSTMRNNModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMRNNModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True, dropout=0)
        self.output_layer = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, input, input_lengths):
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first=True)
        out, (hidden, cstate) = self.lstm(packed_input)
        output = self.output_layer(hidden)
        return output

class Trainer(object):

    def __init__(self, feature_dim, hidden_size, data_set):
        self.data_set = data_set
        self.rnn = LSTMRNNModule(feature_dim, hidden_size)
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=True)
        self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=0.001, momentum=0.9)
    
    def train(self, max_iter, eval_every, batch_size, num_train_samples):

        for it in range(max_iter):            
            batch_num = 0
            for X,L,Y in self.data_set.get_next_batch(batch_size, num_train_samples):
                batch_num += 1
                loss = self.train_iter(X,L,Y)                
                print("{3}/{0}/{1}, Loss: {2:0.3f}".format(it + 1, max_iter, loss, batch_num))
    
    def train_iter(self, input, input_lengths, target):
        self.optimizer.zero_grad()
        output = self.rnn(input, input_lengths)
        output = output[0]
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


FEATURE_DIM = 1000
HIDDEN_SIZE = 32
MAX_SEQ_LEN = 1500
NUM_TRAIN_SAMPLES = 2000
MAX_ITER = 30
EVAL_EVERY = 1
BATCH_SIZE = 1

data_set = DataGenerator(FEATURE_DIM, MAX_SEQ_LEN, NUM_TRAIN_SAMPLES)

trainer = Trainer(FEATURE_DIM, HIDDEN_SIZE, data_set)

trainer.train(MAX_ITER, EVAL_EVERY, BATCH_SIZE, NUM_TRAIN_SAMPLES)
