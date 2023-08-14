import pandas as pd
import numpy as np
import torch
import pickle
import os
import re

from torch.utils import data as torch_data
from sklearn.model_selection import train_test_split
from torchdrug import datasets, transforms, data, utils
from torchdrug.core import Registry as R


RUN = 0

@R.register('datasets.YeastBio')
@utils.copy_args(data.ProteinDataset.load_sequence)
class YeastBio(data.ProteinDataset):
    '''
    Paired homological sequences of reference and "evolved" proteins. Label is binary: 1 - evolved and 0 - reference.
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

        reference = []
        evolved = []
        yeast_id = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()

            for i in range(0, len(lines), 3):
                yeast_id.append(re.sub('\W+', '', lines[i]))
                reference.append(re.sub('\W+', '', lines[i+1]))
                evolved.append(re.sub('\W+', '', lines[i+2]))

        df = pd.DataFrame({'yeast_id' : yeast_id, 'reference.sequence' : reference, 'evolved.sequence' : evolved})
        
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

        sequences = list(np.concatenate([train_seqs_list, val_seqs_list, test_seqs_list]))
        targets = {'ref_evolve' : np.concatenate([train_seqs_labels, val_seqs_labels, test_seqs_labels]).tolist()}

        print(len(train_seqs_list), len(val_seqs_list), len(test_seqs_list))
#        print(type(targets['ref_evolve'][0]))
        print(len(sequences))
        print(len(targets))

        self.load_sequence(sequences, targets, verbose=verbose, **kwargs)

    def split(self):
        train_set = torch_data.Subset(self, list(range(self.train_end)))
        valid_set = torch_data.Subset(self, list(range(self.train_end, self.val_end)))
        test_set = torch_data.Subset(self, list(range(self.val_end, self.test_end)))
        return train_set, valid_set, test_set


# Load one split:
data_path = '/lustre/isaac/scratch/ababjac/DeepSurface/data/'

truncate_transform = transforms.TruncateProtein(max_length=1024, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = YeastBio(data_path, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
print(dataset.tasks)
train_split, valid_split, test_split = dataset.split()
print(train_split)
print(valid_split)
print(test_split)

