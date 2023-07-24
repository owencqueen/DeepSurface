import pandas as pd
import numpy as np
import torch
import pickle
import os

from torch.utils import data as torch_data
from sklearn.model_selection import train_test_split
from torchdrug import datasets, transforms, data, utils
from torchdrug.core import Registry as R



@R.register('datasets.DeepSurface')
@utils.copy_args(data.ProteinDataset.load_sequence)
class DeepSurface(data.ProteinDataset):
    '''
    Paired homological sequences of deep and surface proteins. Label is binary: 1 - deep and 0 - surface.
    Statistics:
        - #Train: 331854
        - #Valid: 36874
        - #Test: 92184
        - #Classification task: 1
    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    '''

    csv_file = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
    pkl_file = '/lustre/isaac/scratch/ababjac/deep-surface-protein-NLP/splits/splits.pkl'

    split_no = 1

    train_end = 331854        # first 70%
    val_end = train_end+36874 # next 10%
    test_end = val_end+92184  # last 20%

    splits = ['train', 'valid', 'test']
    target_fields = ['deep_surface']

    def __init__(self, path, verbose=1, **kwargs):
        
        path = os.path.expanduser(path)
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.path = path

        df = pd.read_csv(self.csv_file, delimiter=',', header=0)
        
        inds = pickle.load(open(self.pkl_file, 'rb'))
        train_inds, test_inds = inds[self.split_no][0]
        train_set = df.iloc[train_inds,:]
        test_set = df.iloc[test_inds,:]

        train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=1234)

        train_seqs_list = train_set['surf.sequence'].tolist() + train_set['deep.sequence'].tolist()
        train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0], dtype=int), np.ones(train_set.shape[0], dtype=int)])

        val_seqs_list = val_set['surf.sequence'].tolist() + val_set['deep.sequence'].tolist()
        val_seqs_labels = np.concatenate([np.zeros(val_set.shape[0], dtype=int), np.ones(val_set.shape[0], dtype=int)])

        test_seqs_list = test_set['surf.sequence'].tolist() + test_set['deep.sequence'].tolist()
        test_seqs_labels = np.concatenate([np.zeros(test_set.shape[0], dtype=int), np.ones(test_set.shape[0], dtype=int)])

        sequences = list(np.concatenate([train_seqs_list, val_seqs_list, test_seqs_list]))
        targets = {'deep_surface' : np.concatenate([train_seqs_labels, val_seqs_labels, test_seqs_labels]).tolist()}

        print(type(targets['deep_surface'][0]))
'''
#        print(sequences)
#        print(targets)
        self.load_sequence(sequences, targets, verbose=verbose, **kwargs)

    def split(self):
        train_set = torch_data.Subset(self, list(range(self.train_end)))
        valid_set = torch_data.Subset(self, list(range(self.train_end, self.val_end)))
        test_set = torch_data.Subset(self, list(range(self.val_end, self.test_end)))
        return train_set, valid_set, test_set
'''

# Load one split:
data_path = '/lustre/isaac/scratch/ababjac/DeepSurface/data/'

truncate_transform = transforms.TruncateProtein(max_length=1024, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = DeepSurface(data_path, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
#print(dataset.tasks)
#train_split, valid_split, test_split = dataset.split()
#print(train_split)
#print(valid_split)
#print(test_split)

