# Stock Market Sentiment Analysis using Financial News.
# A toy demo inspired by Jason Brownlee's tutorial:
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
import numpy as np
import re
import torch
import csv

local_dictionary = {}
word_counter = 1 # Reserve 0 for padding
def one_hot(d, vsize):
  global word_counter
  global local_dictionary
  # d is a string.
  d = d.lower()
  word_list = re.split(' |!', d)
  # remove empty string.
  word_list = list(filter(None, word_list))
  for w in word_list:
    if w not in local_dictionary:
        local_dictionary[w] = word_counter
        word_counter += 1
  return [local_dictionary[w] for w in word_list]

def pad_sequences(seq, maxlen, padding='post'):
  for s in seq:
    s += [0,] * (maxlen - len(s))
  return seq

def read_data(FileName):
  _docs = []
  _labels = []
  with open(FileName, 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader)
    for x in csvreader:
        print(x)
        _docs.append(x[0])
        _labels.append(int(x[1]))

  # class labels: 0: 'positive sentiment'; 1: 'negative setiment'.
  # In Pytorch, CrossEntropyLoss expects the labels to be in the
  # range of [0, #classes).
  return _docs, np.array(_labels)

train_docs, train_labels = read_data('train.csv')
test_docs, test_labels = read_data('test.csv')

# Tokenizer and one-hot encoding.
vocab_size = 50
encoded_train_docs = [one_hot(d, vocab_size) for d in train_docs]
encoded_test_docs = [one_hot(d, vocab_size) for d in test_docs]
print(encoded_train_docs)

# Pad each sequence to a fixed length of 5 words
max_length = 5
padded_train_docs = pad_sequences(encoded_train_docs, maxlen=max_length, padding='post')
padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post')
print(padded_train_docs)

# Model Definition.
class SentimentEmbedding(torch.nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super(SentimentEmbedding, self).__init__()
    self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    self.flatten = torch.nn.Flatten()
    self.fc = torch.nn.Linear(embedding_dim * max_length, 2)
    self.softmax = torch.nn.Softmax()

  def forward(self, x):
    x = self.embedding(x)
    x = self.flatten(x)
    x = self.fc(x)
    output = self.softmax(x)
    return output

my_model = SentimentEmbedding(vocab_size, embedding_dim=8)

# Training
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.00001)
losses, accuracies = [], []
for i in (t := range(99999)):
  print(i)
  X = torch.tensor(padded_train_docs)
  Y = torch.tensor(train_labels)
  optimizer.zero_grad()
  output = my_model(X)
  loss = loss_function(output, Y)
  loss.backward()
  optimizer.step()

test_X = torch.tensor(padded_test_docs)
print(my_model(test_X))

