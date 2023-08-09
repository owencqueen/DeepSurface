import torch
from torch.utils import data as torch_data
from torchdrug import datasets, transforms, data, utils
from torchdrug import core, models, tasks
from torchdrug.core import Registry as R

import pandas as pd
import numpy as np
import logging
import json
import os
import re
import pickle
from sklearn.model_selection import train_test_split

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

PATH = '/lustre/isaac/scratch/ababjac/DeepSurface/'
D_PATH = PATH+'/data/'
ITERS = 10 # *10 = num_epochs
START = 0
NUM_EPOCH = None
LOAD_EPOCH = False
M_NAME = 'ProtBERT'
D_NAME = 'YeastBio'
RUN = 0

@R.register('datasets.YeastBio')
@utils.copy_args(data.ProteinDataset.load_sequence)
class YeastBio(data.ProteinDataset):
    '''
    Paired homological sequences of reference and evolved proteins. Label is binary: 0 - reference and 1 - evolved.
    Statistics:
        - #Train: 8004
        - #Valid: 890
        - #Test: 2224
        - #Classification task: 1
    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    '''

    txt_file = '/lustre/isaac/scratch/ababjac/DeepSurface/data/yeast_bio.txt'
    pkl_file = '/lustre/isaac/scratch/ababjac/DeepSurface/data/splits2.pkl'

    split_no = 1
    run = RUN

    train_end = 8004         # first 70%
    val_end = train_end+890  # next 10%
    test_end = val_end+2224  # last 20%

    splits = ['train', 'valid', 'test']
    target_fields = ['ref_evolve']

    def __init__(self, path, verbose=1, **kwargs):

        path = os.path.expanduser(path)

        if not os.path.exists(path):
            os.makedirs(path)

        self.path = path

        df = pd.read_csv(self.csv_file, delimiter=',', header=0)

        inds = pickle.load(open(self.pkl_file, 'rb'))
        train_inds, test_inds = inds[self.split_no][self.run]
        train_set = df.iloc[train_inds,:]
        test_set = df.iloc[test_inds,:]

        train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=1234)

        train_seqs_list = train_set['reference.sequence'].tolist() + train_set['evolved.sequence'].tolist()
        train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0], dtype=int), np.ones(train_set.shape[0], dtype=int)])

        val_seqs_list = val_set['reference.sequence'].tolist() + val_set['evolved.sequence'].tolist()
        val_seqs_labels = np.concatenate([np.zeros(val_set.shape[0], dtype=int), np.ones(val_set.shape[0], dtype=int)])

        test_seqs_list = test_set['reference.sequence'].tolist() + test_set['evolved.sequence'].tolist()
        test_seqs_labels = np.concatenate([np.zeros(test_set.shape[0], dtype=int), np.ones(test_set.shape[0], dtype=int)])

        sequences = np.concatenate([train_seqs_list, val_seqs_list, test_seqs_list]).tolist()
        targets = {'ref_ecolve' : np.concatenate([train_seqs_labels, val_seqs_labels, test_seqs_labels]).tolist()}

        #not needed on small data
        '''
        del df
        del inds
        del train_inds
        del test_inds
        del train_set
        del val_set
        del test_set
        del train_seqs_list
        del val_seqs_list
        del test_seqs_list
        del train_seqs_labels
        del val_seqs_labels
        del test_seqs_labels

        gc.collect()
        '''

        self.load_sequence(sequences, targets, verbose=verbose, **kwargs)

    def split(self):
        train_set = torch_data.Subset(self, list(range(self.train_end)))
        valid_set = torch_data.Subset(self, list(range(self.train_end, self.val_end)))
        test_set = torch_data.Subset(self, list(range(self.val_end, self.test_end)))
        return train_set, valid_set, test_set

truncate_transform = transforms.TruncateProtein(max_length=1024, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = YeastBio(D_PATH, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
train_set, valid_set, test_set = dataset.split()

model = models.ProteinBERT(
                   input_dim=dataset.node_feature_dim,
                   )

task = tasks.PropertyPrediction(
                   model,
                   task=dataset.tasks,
                   criterion='bce',
                   metric=('auroc', 'auprc'),
                   normalization=False,
                   num_mlp_layer=2
                   )


#logging.basicConfig(filename=PATH+'/results/{}.log'.format(M_NAME), filemode='w')
#logger = logging.getLogger()

optimizer = torch.optim.Adam(task.parameters(), lr=5e-5)
solver = core.Engine(
                   task,
                   train_set,
                   valid_set,
                   test_set,
                   optimizer,
                   gpus=[0],
                   batch_size=16,
                   #logger='logging'
                   )

best_score = float("-inf")
best_epoch = -1

torch.cuda.empty_cache()

if not os.path.exists(PATH+'/models/{}/{}/run_{}'.format(D_NAME, M_NAME, RUN)):
    os.makedirs(PATH+'/models/{}/{}/run_{}'.format(D_NAME, M_NAME, RUN))

if LOAD_EPOCH:
    solver.load(PATH+'/models/{}/{}/run_{}/epoch_{}.pth'.format(D_NAME, M_NAME, RUN, NUM_EPOCH))

for i in range(START, ITERS+1):
    solver.model.split = 'train'
    solver.train(num_epoch=1)
    solver.save(PATH+'/models/{}/{}/run_{}/epoch_{}.pth'.format(D_NAME, M_NAME, RUN, (solver.epoch*i)))

    solver.model.split = 'valid'
    metric = solver.evaluate('valid', log=True)

    score = []
    for k, v in metric.items():
        if k.startswith('auroc'):
            score.append(v)
    
    score = sum(score) / len(score)
    if score > best_score:
        best_score = score
        best_epoch = (solver.epoch * i)

solver.load(PATH+'/models/{}/{}/run_{}/epoch_{}.pth'.format(D_NAME, M_NAME, RUN, best_epoch))

#with open(PATH+'/models/{}/best_epoch_{}.json'.format(M_NAME, best_epoch), 'w') as fout:
#    json.dump(solver.config_dict(), fout)

solver.save(PATH+'/models/{}/{}/run_{}/best.pth'.format(D_NAME, M_NAME, RUN))

if not os.path.exists(PATH+'/results/{}/'.format(D_NAME)):
    os.makedirs(PATH+'/results/{}/'.format(D_NAME))

solver.model.split = 'valid'
eval_metrics = solver.evaluate('valid', log=True)
with open(PATH+'/results/{}/{}_eval_metrics_{}.log.txt'.format(D_NAME, M_NAME, RUN), 'w') as f:
    f.write(str(eval_metrics))

solver.model.split = 'test'
test_metrics = solver.evaluate('test', log=True)
with open(PATH+'/results/{}/{}_test_metrics_{}.log.txt'.format(D_NAME, M_NAME, RUN), 'w') as f:
    f.write(str(test_metrics))


