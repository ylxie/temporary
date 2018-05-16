
from __future__ import print_function, division, unicode_literals
import glob
import unicodedata
import string
import random
import numpy as np
import torch
import torch.autograd 

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class NameClassification(object):

    def __init__(self, data_folder):
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)
        # Build the category_lines dictionary, a list of names per language
        self.category_lines = {}
        self.all_categories = []
        self.all_categories_tensors = []
        self.category_lines_tensors = {}
    
        for filename in self.findFiles(data_folder + '/*.txt'):
            category = filename.split('\\')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)
        self.createTensors()

    def createTensors(self):
        for i in range(self.n_categories):
            self.all_categories_tensors.append(torch.LongTensor([i]))
            self.category_lines_tensors[i] =\
                [self.name2Tensor(it) for it in self.category_lines[self.all_categories[i]]]


    def findFiles(self, path):
        return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]

    def name2Tensor(self, name):
        tensor = torch.zeros(len(name), 1, self.n_letters)
        for i, c in enumerate(name):
            tensor[i][0][self.all_letters.find(c)] = 1
        return  tensor

    def all(self):
        for cat, items in self.category_lines_tensors.items():            
            for name in items:                
                yield torch.autograd.Variable(name), torch.autograd.Variable(self.all_categories_tensors[cat])

    def nextBatch(self):        
        cat = random.randint(0, self.n_categories - 1)
        cat_name = self.all_categories[cat]
        idx = random.randint(0, len(self.category_lines[cat_name]) - 1)        

        cat_tensor = self.all_categories_tensors[cat]
        input_tensor = self.category_lines_tensors[cat][idx]
        #print("{0} - {1}".format(cat_name, it))
        yield torch.autograd.Variable(input_tensor), torch.autograd.Variable(cat_tensor)
    
    def nextBatch2(self):

        pass



class RNNEncoder(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hiddenLayer = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.outputLayer = torch.nn.Linear(hidden_size, output_size)        
        self.softmax = torch.nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):        
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.hiddenLayer(combined)
        output = self.outputLayer(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.autograd.Variable(torch.zeros(1, self.hidden_size))

class RNNTrainer():

    def __init__(self, max_iter, data, hidden_size):
        self.max_iter = max_iter
        self.data = data        
        self.rnn = RNNEncoder(data.n_letters, data.n_categories, hidden_size)
        self.learning_rate = 0.005
        self.criterion = torch.nn.NLLLoss()
        self.losses = []
        self.total_loss = 0
    
    def train(self):

        for i in range(self.max_iter):
            for inputs, targets in self.data.nextBatch():
                self.train_iter(inputs, targets)
            
            if i % 500000000 == 0:
                correct, total = self.evaluate()
                print("Loss: {0}, Total: {1}, Correct: {2}".format(self.total_loss, total, correct))
    
    def train_iter(self, input, target):
        self.rnn.zero_grad()
        hidden = self.rnn.initHidden()
        for i in range(0, len(input)):
            output, hidden = self.rnn(input[i], hidden)
        loss = self.criterion(output, target)
        loss.backward()
        for p in self.rnn.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)            
        self.total_loss += loss.data[0]
        self.losses.append(loss.data[0])

    def predict(self, s):
        hidden = self.rnn.initHidden()
        for i in range(0, len(s)):
            output, hidden = self.rnn(s[i], hidden)
        topv, topi = output.data.topk(1)
        return topv[0][0], topi[0][0]

    def evaluate(self):
        correct = 0
        total = 0
        for input, target in self.data.all():
            total += 1
            score, idx = self.predict(input)
            if idx == target.data[0]:
                correct += 1
        return correct, total




data = NameClassification(r"D:\Temp\Pytorch\data\names")

trainer = RNNTrainer(100000, data, 64)

trainer.train()


