from __future__ import unicode_literals, print_function, division

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import re

from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 10
ITERATIONS = 1000
SEQ_LENGTH = 50
EMBEDDING_SIZE = 100
LSTM_SIZE = 64

# TODO 1: put a text file of your choosing in the same directory and put its name here
TEXT_FILE = 'D:/Course/deep/huiy_assignment4/custom_text.txt'

string = open(TEXT_FILE).read()

# convert text into tekens
tokens = re.split('\W+', string)

# get vocabulary
vocabulary = sorted(set(tokens))

# get corresponding indx for each word in vocab
word_to_ix = {word: i for i, word in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
print('vocab size: {}'.format(VOCABULARY_SIZE))

#############################################
# TODO 2: create variable for embedding matrix. Hint: you can use nn.Embedding for this
#############################################
embedding_lookup_for_x = nn.Embedding(num_embeddings=VOCABULARY_SIZE, embedding_dim=EMBEDDING_SIZE)


#############################################
# TODO 3: define an lstm encoder function that takes the embedding lookup and produces a final state
# _, final_state = your_lstm_encoder(embedding_lookup_for_x, ...)
#############################################
def lstm_encoder(embedding_lookup_for_x, x, lstm_encode_layer):
    x = embedding_lookup_for_x(x)
    return lstm_encode_layer(x)


#############################################
# TODO 4: define an lstm decoder function that takes the final state from previous step and produces a sequence of outputs
# outs, _ = your_lstm_decoder(final_state, ...)
#############################################
def lstm_decoder(final_state, x, lstm_decode_layer, embedding_for_decode):
    decoder_input = torch.zeros(BATCH_SIZE, 1, EMBEDDING_SIZE)  # SOS token
    hidden = final_state

    outs = torch.zeros(BATCH_SIZE, SEQ_LENGTH, LSTM_SIZE)
    embedding_x = embedding_for_decode(x)
    # use teacher forcing
    for i in range(SEQ_LENGTH):
        decoder_output, hidden = lstm_decode_layer(decoder_input, hidden)
        outs[:, i, :] = decoder_output.squeeze()
        decoder_input = embedding_x[:, i, :].view(BATCH_SIZE, 1, EMBEDDING_SIZE)
    return outs


#############################################
# TODO: create loss/train ops
#############################################

lstm_encode_layer = nn.LSTM(input_size=EMBEDDING_SIZE, hidden_size=LSTM_SIZE, batch_first=True)
embedding_for_decode = nn.Embedding(num_embeddings=VOCABULARY_SIZE, embedding_dim=EMBEDDING_SIZE)
lstm_decode_layer = nn.LSTM(input_size=EMBEDDING_SIZE, hidden_size=LSTM_SIZE, batch_first=True)
output_layer = nn.Linear(in_features=LSTM_SIZE, out_features=VOCABULARY_SIZE)
loss_fn = nn.BCELoss()
parameters = list(embedding_lookup_for_x.parameters()) + \
             list(embedding_for_decode.parameters()) + \
             list(lstm_encode_layer.parameters()) + \
             list(lstm_decode_layer.parameters()) + \
             list(output_layer.parameters())
optimizer = optim.Adam(params=parameters, lr=0.01)


# helper function
def to_one_hot(y_tensor, c_dims):
    """converts a N-dimensional input to a NxC dimnensional one-hot encoding
    """
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y_tensor.shape, -1)
    return y_one_hot


# do training

i = 0
for num_iter in range(ITERATIONS):

    if num_iter % 10 == 0: print(num_iter)
    batch = [[vocabulary.index(v) for v in tokens[ii:ii + SEQ_LENGTH]] for ii in range(i, i + BATCH_SIZE)]
    batch = np.stack(batch, axis=0)
    batch = torch.tensor(batch, dtype=torch.long)
    i += BATCH_SIZE
    if i + BATCH_SIZE + SEQ_LENGTH > len(tokens): i = 0

    #############################################
    # TODO: create loss and update step
    #############################################

    # Hint:following steps will most likely follow the pattern
    _, final_state = lstm_encoder(embedding_lookup_for_x, batch, lstm_encode_layer)
    targets = to_one_hot(batch, c_dims=VOCABULARY_SIZE).view(BATCH_SIZE, SEQ_LENGTH, VOCABULARY_SIZE)
    outs = lstm_decoder(final_state, batch, lstm_decode_layer, embedding_for_decode)
    outs = F.softmax(output_layer(outs), dim=-1)
    loss = loss_fn(outs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# plot word embeddings
# assuming embeddingscalled "learned_embeddings",
learned_embeddings = embedding_lookup_for_x(
    torch.tensor([i for i in range(VOCABULARY_SIZE)], dtype=torch.long)).detach().numpy()
fig = plt.figure()
learned_embeddings_pca = sklearn.decomposition.PCA(2).fit_transform(learned_embeddings)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='w')
MIN_SEPARATION = .1 * min(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])

fig.clf()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='w')

#############################################
# TODO 5: run this multiple times
#############################################
num_plotted_tokens = 100
xy_plotted = set()
for i in np.random.choice(VOCABULARY_SIZE, num_plotted_tokens, replace=False):
    x_, y_ = learned_embeddings_pca[i]
    if any([(x_ - point[0]) ** 2 + (y_ - point[1]) ** 2 < MIN_SEPARATION for point in xy_plotted]): continue
    xy_plotted.add(tuple([learned_embeddings_pca[i, 0], learned_embeddings_pca[i, 1]]))
    ax.annotate(vocabulary[i], xy=learned_embeddings_pca[i])
plt.show()