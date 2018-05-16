

import torch
import torch.nn as nn
from torch.autograd import Variable

batch_size = 4
max_length = 5
hidden_size = 3
feature_dim = 2
n_layers =1

# container
batch_in = torch.zeros((batch_size, max_length, feature_dim))

#data
#vec_1 = torch.FloatTensor([[1], [2], [3], [4], [5]])
#vec_2 = torch.FloatTensor([[1], [2], [3], [0], [0]])
#vec_3 = torch.FloatTensor([[1], [3], [0], [0], [0]])
#vec_4 = torch.FloatTensor([[1], [0], [0], [0], [0]])

# vec_1 = torch.FloatTensor([[1,1], [2,2], [3,3], [4,4], [5,5]])
# vec_2 = torch.FloatTensor([[1,1], [2,2], [3,3], [0,0], [0,0]])
# vec_3 = torch.FloatTensor([[1,1], [3,3], [0,0], [0,0], [0,0]])
# vec_4 = torch.FloatTensor([[1,1], [0,0], [0,0], [0,0], [0,0]])

vec_1 = torch.FloatTensor([[1,1], [2,2], [3,3], [4,4], [5,5]])
vec_2 = torch.FloatTensor([[1,1], [2,2], [3,3]])
vec_3 = torch.FloatTensor([[1,1], [3,3]])
vec_4 = torch.FloatTensor([[1,1]])


batch_in[0] = vec_1
batch_in[1] = vec_2
batch_in[2] = vec_3
batch_in[3] = vec_4

batch_in = Variable(batch_in)

seq_lengths = [5,3,2,1] # list of integers holding information about the batch size at each sequence step

# pack it
pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)

# initialize
rnn = nn.RNN(feature_dim, hidden_size, n_layers, batch_first=True) 
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))

#forward 

out, hidden = rnn(pack, h0)

# unpack
unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
