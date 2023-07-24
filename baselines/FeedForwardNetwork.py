

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import layers
from functools import reduce

def unravel_df(df):
    seq_surf = df['surf.sequence'].tolist()
    seq_deep = df['deep.sequence'].tolist()

    seqs = seq_surf + seq_deep

    label = list(np.concatenate([np.zeros(len(seq_surf)), np.ones(len(seq_deep))]).astype(int))
    print(len(label))
    
    newdf = {'text': seqs, 'label': label}
    return newdf


dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
df = pd.read_csv(dpath)
df['lengths_deep'] = df['deep.sequence'].str.len()
df['lengths_surf'] = df['surf.sequence'].str.len()
padding_length = max(max(df['lengths_deep']),max(df['lengths_surf']))

df = df.convert_dtypes()

# Load one split:
split_no = 1
inds = pickle.load(open('/lustre/isaac/scratch/ababjac/DeepSurface/data/splits.pkl', 'rb'))

for train_inds, test_inds in inds[split_no]:
    train_set = df.iloc[train_inds,:]
    test_set = df.iloc[test_inds,:]
    train_seqs_list = train_set['surf.sequence'].tolist() + train_set['deep.sequence'].tolist()
    train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0]), np.ones(train_set.shape[0])])
    print('train list', len(train_seqs_list))
    print('train labels', train_seqs_labels.shape)

        
    test_seqs_list = test_set['surf.sequence'].tolist() + test_set['deep.sequence'].tolist()
    test_seqs_labels = np.concatenate([np.zeros(test_set.shape[0]), np.ones(test_set.shape[0])])
    print('test list', len(test_seqs_list))
    print('test labels', test_seqs_labels.shape)
    print('-' * 30)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


avg_prc = []
avg_roc = []

for i in range(5):
    print("-----FOLD ", i, "-----")
    print("starting training")

    train = df.iloc[inds[1][i][0]]
    test = df.iloc[inds[1][i][1]]

    train = unravel_df(train_set)
    x_train = train['text']
    y_train = train['label']

    test = unravel_df(test_set)
    x_test = test['text']
    y_test = test['label']

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(list(x_train))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    x_train_seq = pad_sequences(x_train, padding_length, padding='post', truncating='post')
    x_test_seq = pad_sequences(x_test, padding_length)

    x_train_seq = np.reshape(x_train_seq, (x_train_seq.shape[0], x_train_seq.shape[1], 1))
    x_test_seq = np.reshape(x_test_seq, (x_test_seq.shape[0], x_test_seq.shape[1], 1))

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(y_train.shape)

    model = ks.Sequential()
    model.add(layers.Dense(32, input_shape=(padding_length, ), activation='relu'))
    model.add(layers.Dense(64))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

    batch_size = 64
    model.fit(x_train_seq, y_train, epochs = 10, verbose = 3, batch_size=batch_size)

    predictions = model.predict(x_test_seq)
    roc = roc_auc_score(y_test, predictions)
    prc = average_precision_score(y_test, predictions)

    avg_roc.append(roc)
    avg_prc.append(prc)

    print("ROC:", roc)
    print("PRC:", prc)

def Average(lst):
    return sum(lst) / len(lst)


print("AVERAGE_ROC: " + Average(avg_roc))
print("AVERAGE_PRC: " + Average(avg_prc))
