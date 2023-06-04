import pandas as pd
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

medium_data = pd.read_csv('../full.csv')
medium_data.head()
print("Number of records: ", medium_data.shape[0])
print("Number of fields: ", medium_data.shape[1])

medium_data['SMILE']

token_regex = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|H|t_|h_|ad_|\(|\)|\.|=|#|−|\+|\\\\\/|:|∼|@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"
tokenizer.get_regex(token_regex,medium_data['SMILE'])
total_characters = len(tokenizer.word_index) + 1

input_sequences = []
for line in medium_data['SMILE']:
    token_list = tokenizer.regex_to_sequences([line])[0]
    #print(token_list)
    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# print(input_sequences)
print("Total input sequences: ", len(input_sequences))

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[1]

xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_characters)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))

query_value_attention_seq = tf.keras.layers.Attention(use_scale=False, score_mode='dot')([query_seq_encoding, value_seq_encoding])
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(query_value_attention_seq) 
input_layer = tf.keras.layers.Concatenate()([query_encoding, query_value_attention])
model.add(input_layer)

adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=200, verbose=1)
#print model.summary()
print(model)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

