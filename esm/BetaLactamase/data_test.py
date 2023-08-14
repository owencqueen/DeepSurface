import torch
from torchdrug import datasets, transforms
from torchdrug import core, models, tasks, utils
import logging
import json
import os, argparse, sys

sys.path.append("/om2/user/oqueen/DeepSurface/esm")
from torchdrug_esm import CustomModel

PATH = '/om2/user/oqueen/DeepSurface'
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

dataset = datasets.BetaLactamase(D_PATH, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
train_set, valid_set, test_set = dataset.split()

import ipdb; ipdb.set_trace()


# task = tasks.PropertyPrediction(
#                    model, 
#                    task=dataset.tasks,
#                    criterion='mse', 
#                    metric=('mae', 'rmse', 'spearmanr'),
#                    normalization=False,
#                    num_mlp_layer=2
#                    )


