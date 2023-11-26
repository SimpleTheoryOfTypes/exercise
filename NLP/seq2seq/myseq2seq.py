#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import multi30k
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from typing import Tuple, List, Iterable, List
import random, math, time
import numpy as np


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
train, valid, test = multi30k.Multi30k("./", split=("train", "valid", "test"), language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
iter1 = iter(train)
print("Sample train sentence pair: ", next(iter1))
iter2 = iter(valid)
print("Sample Validation sentence pair: ", next(iter2))

for data_iter in train:
  print("Next Validation sentence pair: ", data_iter)
  break

print(f"Number of training examples = ", len(list(train)))
print(f"Number of validation examples = ", len(list(valid)))
#print(f"Number of test examples = ", len(list(test)))

token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data iterator
    train_iter = multi30k.Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. The index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

# Print some sample tokens in vocab_transform:
print("vocab_transform size (DE) = ", vocab_transform['de'].vocab.__len__())
print("vocab_transform size (EN) = ", vocab_transform['en'].vocab.__len__())
print("the 678th DE token is ", vocab_transform['de'].vocab.lookup_token(678))
print("the 678th EN token is ", vocab_transform['en'].vocab.lookup_token(678))


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        #print("Encoder embedded shape = ", embedded.shape)
        
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        #print("Encoder hidden shape = ", hidden.shape)
        #print("Encoder cell shape = ", cell.shape)
        
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        input = input.unsqueeze(0)
        #print("input shape = ", input.shape)
        
        #input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        #print("decoder embedded shape = ", embedded.shape)
        #embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, bath size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.fc_out(output.squeeze(0))
        
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
    
class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"
    
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        #print("Seq2seq src shape = ", src.shape)
        #print("Seq2seq trg shape = ", trg.shape)
        
        # teacher_forcing_ration is probability to use teacher forcing
        # e.g., if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        #print("Seq2seq hidden shape = ", hidden.shape)
        #print("Seq2seq cell shape = ", cell.shape)
        
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        #print("Seq2seq input shape = ", input.shape)
        
        for t in range(1, trg_len):
            # insert input token embdding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            #print("Seq2seq output shape = ", output.shape)
            
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # decide if we are going to use teach forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs


INPUT_DIM = len(vocab_transform[SRC_LANGUAGE].vocab)
OUTPUT_DIM = len(vocab_transform[TGT_LANGUAGE].vocab)
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
HID_DIM = 1024
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)


# top-like gpu utility check: nvidia-smi -l 1
#assert torch.cuda.is_available(), "No CUDA found on your machine."
if torch.cuda.is_available():
  print("Training on this device: ", torch.cuda.get_device_name(0))
else:
  print("Training on CPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


### Testing the model forward flow
# Test tokenizer: input should be a list of words
def sample_data(input_sentences: List[Tuple[str]]):
    print(input_sentences[0])
    sample_src = [x[0].strip().split() for x in input_sentences]
    sample_dst = [x[1].strip().split() for x in input_sentences]
    return sample_src, sample_dst

sample_de, sample_en = sample_data(list(train)[615:616])

x = []
x_toks = []
for sample in sample_de:
    t0 = [BOS_IDX] + vocab_transform['de'](sample) + [EOS_IDX]
    x.append(t0)
    x_toks.append([vocab_transform['de'].vocab.lookup_token(t) for t in t0])

y = []
y_toks = []
for sample in sample_en:
    t0 = [BOS_IDX] + vocab_transform['en'](sample) + [EOS_IDX]
    y.append(t0)
    y_toks.append([vocab_transform['en'].vocab.lookup_token(t) for t in t0])

print("x = ", x)
print("y = ", y)
print("x_toks = ", x_toks)
print("y_toks = ", y_toks)

tx_len = len(x[0]) # model assumes input is [seq len, batch size]
ty_len = len(y[0])
tx = torch.tensor(x).reshape(tx_len, 1)
ty = torch.tensor(y).reshape(ty_len, 1)
sample_output = model(tx.to(device), ty.to(device), 0.5)
predict_tokids = torch.argmax(sample_output, dim=-1).flatten()
predict_toks = []
for ptid in predict_tokids:
  predict_toks.append(vocab_transform['en'].vocab.lookup_token(ptid))
print("initial model predicted toks = ", predict_toks)

"""
Preparing data:
https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/transformer_tutorial.ipynb
"""
from torch.nn.utils.rnn import pad_sequence

#helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# Function to add BOS/ESOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization
                                               vocab_transform[ln], # Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
    
# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
        
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

from torch.utils.data import DataLoader
BATCH_SIZE = 128
#print(len(train_iter))
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
#print(train_dataloader.__str__())
#for src, tgt in train_dataloader:
#     print(src.shape)
#     print(tgt.shape)
#     print(src[:,123])
#     print(tgt[:,123])
#     break

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    numTrainDataPoints = 0
    for i, batch in enumerate(iterator):
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        # trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        numTrainDataPoints += 1 #iterator.batch_size
        # print("local loss = ", epoch_loss)
    
    return epoch_loss / numTrainDataPoints

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    numEvalDataPoints = 0
    with torch.no_grad():
        for i,batch in enumerate(iterator):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            
            output = model(src, trg, 0) # turn off teacher forcing
            
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            numEvalDataPoints += 1 #iterator.batch_size
            
    return epoch_loss / numEvalDataPoints

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 300
CLIP = 1
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, train_dataloader, criterion)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
        
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')

    # check the sample translation after each training epoch
    paul = model(tx.to(device), ty.to(device), 0.0)
    predict_tokids = torch.argmax(paul, dim=-1).flatten()
    predict_toks = []
    for ptid in predict_tokids:
      predict_toks.append(vocab_transform['en'].vocab.lookup_token(ptid))
    print("[GOLD TRANSLATION] = ", y_toks)
    print("model translation = ", predict_toks)

