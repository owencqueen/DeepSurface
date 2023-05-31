import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Get longest string length for padding
df = pd.read_csv( '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv')
df['lengths_deep'] = df['deep.sequence'].str.len()
df['lengths_surf'] = df['surf.sequence'].str.len()
padding_length = max(max(df['lengths_deep']),max(df['lengths_surf']))

def tokenize_data(data):
  tokenizer = Tokenizer(char_level=True)
  tokenizer.fit_on_texts(data)
  tokens = tokenizer.texts_to_sequences(data)

  padded_seqs = pad_sequences(tokens, maxlen=padding_length, padding='post')
  return padded_seqs

def unravel_df(df):
    seq_surf = df['surf.sequence'].tolist()
    seq_deep = df['deep.sequence'].tolist()

    seqs = seq_surf + seq_deep

    label = list(np.concatenate([np.zeros(len(seq_surf)), np.ones(len(seq_deep))]).astype(int))
    print(len(label))
    
    newdf = {'text': seqs, 'label': label}
    return newdf

# Load one split:
split_no = 1
inds = pickle.load(open('/lustre/isaac/scratch/shaebarh/DeepSurface/data/splits.pkl', 'rb'))

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

    
   
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Normalizer
from sklearn.metrics import average_precision_score
normalizer = Normalizer()

prc_avg = []
roc_avg = []

for i in range(5):
  print("-----FOLD ", i, "-----")
  print("starting training")

  train = df.iloc[inds[1][i][0]]
  test = df.iloc[inds[1][i][1]]

  train = unravel_df(train)
  test = unravel_df(test)
  print("data unraveled")

  x_train = tokenize_data(train['text'])
  x_test = tokenize_data(test['text'])
    
  y_train = train['label']
  y_test = test['label']

  x_train = normalizer.transform(x_train)
  x_test = normalizer.transform(x_test)
  print("data normalized")

  import xgboost as xgb

  xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric="auc", learning_rate=0.1, max_depth=25, )
  xgb_model.fit(x_train,y_train)
  predictions = xgb_model.predict(x_test)
  
  print("model trained")

  roc = roc_auc_score(y_test, predictions)
  prc = average_precision_score(y_test, predictions)

  roc_avg.append(roc)
  prc_avg.append(prc)

  print("ROC: ", roc)
  print("PRC: ", prc)
  

ROC = sum(roc_avg)/len(roc_avg)
PRC = sum(prc_avg)/len(prc_avg)

print("AVG ROC: ", ROC)
print("AVG PRC: ", PRC)
