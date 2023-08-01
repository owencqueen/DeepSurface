import torch
from torchdrug import datasets, transforms
from torchdrug import core, models, tasks, utils
import logging
import json
import os, argparse, sys

sys.path.append("/home/oqueen/DeepSurface/esm")
from torchdrug_esm import CustomModel

import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from sklearn.preprocessing import Normalizer
from scipy.stats import spearmanr

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

PATH = '/home/oqueen/DeepSurface'
D_PATH = PATH+'/data/'
ITERS = 10 # *10 = num_epochs
#modelname = 'ProtLSTM'
D_NAME = 'BetaLactamase'

parser = argparse.ArgumentParser()
parser.add_argument('--nparams', type=str, default = '8m')
parser.add_argument('--frozen', action='store_true', help = 'Run model with frozen encoder weights')
args = parser.parse_args()

name = args.nparams if (args.nparams[-1] == 'm') else "{}m".format(args.nparams)
modelname = "ESM-2-{}".format(name)

truncate_transform = transforms.TruncateProtein(max_length=1024, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = datasets.BinaryLocalization(D_PATH, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
train_set, valid_set, test_set = dataset.split()

# Unpack string sequences:
all_seqs, all_targets = dataset.sequences, dataset.targets["localization"]
train_seqs = [all_seqs[i] for i in train_set.indices]
val_seqs = [all_seqs[i] for i in valid_set.indices]
test_seqs = [all_seqs[i] for i in test_set.indices]

train_y = [all_targets[i] for i in train_set.indices]
val_y = [all_targets[i] for i in valid_set.indices]
test_y = [all_targets[i] for i in test_set.indices]

padding_length = max([max([len(s) for s in sL]) for sL in (train_seqs, val_seqs, test_seqs)])

def tokenize_data(data):
  tokenizer = Tokenizer(char_level=True)
  tokenizer.fit_on_texts(data)
  tokens = tokenizer.texts_to_sequences(data)

  padded_seqs = pad_sequences(tokens, maxlen=padding_length, padding='post')
  return padded_seqs

train_x, val_x, test_x = tokenize_data(train_seqs), tokenize_data(val_seqs), tokenize_data(test_seqs)

rfc = RandomForestClassifier(verbose=2, n_estimators=100, n_jobs=8, max_depth=None)

print('Start model fit')
rfc.fit(train_x, train_y)
print('fit model')

predictions = rfc.predict(test_x)

roc = roc_auc_score(test_y, predictions)
prc = average_precision_score(test_y, predictions)

print('AUROC = {:.4f}'.format(roc))
print('AUPRC = {:.4f}'.format(prc))

# r = spearmanr(test_y, predictions)
# rmse = mean_squared_error(test_y, predictions, squared = False)

# import ipdb; ipdb.set_trace()

# print('RMSE = {:.4f}'.format(rmse))
# print('Spearman = {:.4f}'.format(r.statistic))

# task = tasks.PropertyPrediction(
#                    model, 
#                    task=dataset.tasks,
#                    criterion='mse', 
#                    metric=('mae', 'rmse', 'spearmanr'),
#                    normalization=False,
#                    num_mlp_layer=2
#                    )


