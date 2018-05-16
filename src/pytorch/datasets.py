


import glob
import os.path
import random

import torch
import torch.autograd

import text_processors

class SentenceClassificationDataCollection():
    def __init__(self, data_root_dir):
        self._data_root_dir = data_root_dir
        self.train_samples, self.train_labels = self.__load_train()
        self.test_samples, self.test_labels = self.__load_test()

    def __load_data(self, train_or_test, pos_or_neg):
        data_path = os.path.join(self._data_root_dir, train_or_test, pos_or_neg, "*.txt")
        samples = [open(file, encoding='utf-8').read().strip() for file in glob.glob(data_path)]
        labels = [1 if pos_or_neg == 'pos' else 0 for i in range(len(samples))]
        return samples, labels

    def __load_test(self):
        pos_samples, pos_labels = self.__load_data('test', 'pos')
        neg_samples, neg_labels = self.__load_data('test', 'neg')
        pos_samples.extend(neg_samples)
        pos_labels.extend(neg_labels)
        return pos_samples, pos_labels

    def __load_train(self):
        pos_samples, pos_labels = self.__load_data('train', 'pos')
        neg_samples, neg_labels = self.__load_data('train', 'neg')
        pos_samples.extend(neg_samples)
        pos_labels.extend(neg_labels)
        return pos_samples, pos_labels

class SentenceClassificationDataSet():

    def __init__(self, raw_data_collection, top_words, device='cuda'):
        self.device = torch.device(device)
        self.raw_data_collection = raw_data_collection
        self.text_processor = text_processors.EnglishTextProcessor()
        self._build_vocab(top_words)
        self._build_data_sets()

    def _build_vocab(self, top_words):
        self._orignal_word_freq = dict()

        self._normalized_train_samples = [self.text_processor.process(sample) for sample in self.raw_data_collection.train_samples]
        for sent_tokens in self._normalized_train_samples:
            self._add_sent(sent_tokens)
        self._normalized_test_samples = [self.text_processor.process(sample) for sample in self.raw_data_collection.test_samples]
        for sent_tokens in self._normalized_test_samples:
            self._add_sent(sent_tokens)

        self._word2id = dict()
        self._word2id['__UNK__'] = 0
        for token in sorted(self._orignal_word_freq.items(), key=lambda x: x[1], reverse=True):
            self._word2id[token] = len(self._word2id)
            if (len(self._word2id) >= top_words):
                break
        self.n_words = len(self._word2id)

    def _build_data_sets(self):
        self.train_set = [(self.sent_2_tensor(self.raw_data_collection.train_samples[i]), \
                           self.target_2_tensor(self.raw_data_collection.train_labels[i])) \
                           for i in range(len(self.raw_data_collection.train_samples))]
        self.test_set = [(self.sent_2_tensor(self.raw_data_collection.test_samples[i]), \
                          self.target_2_tensor(self.raw_data_collection.test_labels[i])) \
                          for i in range(len(self.raw_data_collection.test_samples))]

        self.n_train_samples = len(self.train_set)
        self.n_test_samples = len(self.test_set)


    def _add_sent(self, sentence_tokens):
        for token in sentence_tokens:
            self._orignal_word_freq[token] = self._orignal_word_freq.get(token, 0) + 1

    def token_2_tensor(self, token):
        wordid = self._word2id.get(token, 0)
        tensor = torch.zeros(1, self.n_words, device=self.device)
        tensor[0][wordid] = 1
        # return torch.autograd.Variable(tensor)
        return tensor

    def tokens_2_tensor(self, tokens):
        return [self.token_2_tensor(token) for token in tokens]

    def sent_2_tensor(self, sentence):
        sent_tokens = self.text_processor.process(sentence)
        return self.tokens_2_tensor(sent_tokens)

    def target_2_tensor(self, target):
        # return torch.autograd.Variable(torch.LongTensor([target]).view(-1, 1).float())
        return torch.FloatTensor([target]).view(-1,1).cuda()

    def next_train_batch(self, batch_size = 3):
        # idx = random.randint(0, self.n_train_samples - 1)
        shuffled_idx = list(range(self.n_train_samples))
        random.shuffle(shuffled_idx)
        for i in range(0, self.n_train_samples, batch_size):
            samples = list(self.train_set[shuffled_idx[j]] for j in range(i, min(self.n_train_samples, i + batch_size)))
            # if i == 214:
            #     continue
            # print("{0}/{1}-{2}".format(i, self.n_train_samples, len(samples[0][0])))
            yield self.create_pad_sequence_batch(samples)
            # print("{0}/{1}-{2}".format(i, self.n_train_samples, len(samples[0][0])))
        #     samples = self.train_set[batch_idx]
        # # samples = list(x.data.view(-1, x.size()[0], x.size()[1]) for x in self.train_set[idx][0])
        #     input = torch.cat(samples, dim=0)
        #     target = self.train_set[idx][1]
        #     input = torch.autograd.Variable(input)
        #     yield input, target
    
    def next_test_batch(self, batch_size = 1):
        for i in range(0, self.n_test_samples, batch_size):
            samples = list(self.test_set[j] for j in range(i, min(self.n_test_samples, i + batch_size)))
            yield self.create_pad_sequence_batch(samples)
    
    def create_pad_sequence_batch(self, samples):
        # samples.sort()
        sorted_samples = sorted(samples, key=lambda x:len(x[0]), reverse=True)
        # sorted_samples = samples
        target_batch = torch.cat(list(x[1] for x in sorted_samples), dim=0)
        inputs = list(torch.cat(x[0], dim=0) for x in sorted_samples)
        inputs_length = list(len(x) for x in inputs)
        padded_inputs_batch = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        return padded_inputs_batch, inputs_length, target_batch
                









