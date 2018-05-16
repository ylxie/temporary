
import sys

import datasets
from sentiment_classification import RNNBinaryClassifierTrainer

import random
import numpy as np
import torch

import faulthandler

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

platform = sys.platform

#data_root_dir = r'D:\Temp\Pytorch\data\aclImdb'
if sys.platform == 'win32':
    # data_root_dir = r'D:\Temp\Pytorch\data\aclImdb-tiny'
    # data_root_dir = r'D:\Projects\Pytorch\Data\aclImdb-tiny'
    data_root_dir = r'D:\Projects\Pytorch\Data\aclImdb'
else:
    data_root_dir = r'/Data/aclImdb-tiny'

TOP_WORDS_COUNT = 1000
HIDDEN_SIZE = 32
MAX_ITERATIONS = 10
EVAL_EVERY_ITERATIONS = 1
BATCH_SIZE = 8
DEVICE = 'cuda'

print("Loading raw data collections ... ")
raw_data = datasets.SentenceClassificationDataCollection(data_root_dir)
print("Building data sets ... ")
data_set = datasets.SentenceClassificationDataSet(raw_data, TOP_WORDS_COUNT, device=DEVICE)

trainer = RNNBinaryClassifierTrainer(data_set, HIDDEN_SIZE, BATCH_SIZE, device=DEVICE)

trainer.train(MAX_ITERATIONS, EVAL_EVERY_ITERATIONS)

# faulthandler.enable()

# try:
#     trainer.train(MAX_ITERATIONS, EVAL_EVERY_ITERATIONS)
# except:
#     print("ERROR")
#     faulthandler.dump_traceback()


