import torch
from torchdrug import datasets, transforms
from torchdrug import core, models, tasks, utils
import logging
import json
import os, argparse, sys

sys.path.append("/om2/user/oqueen/DeepSurface/esm")
sys.path.append("/om2/user/oqueen/DeepSurface/baselines")
from torchdrug_esm import CustomModel
from yeast import YeastBio

PATH = '/om2/user/oqueen/DeepSurface'
D_PATH = PATH+'/data/'
ITERS = 10 # *10 = num_epochs
#modelname = 'ProtLSTM'
D_NAME = 'Yeast'

parser = argparse.ArgumentParser()
parser.add_argument('--nparams', type=str, default = '8m')
parser.add_argument('--frozen', action='store_true', help = 'Run model with frozen encoder weights')
args = parser.parse_args()

name = args.nparams if (args.nparams[-1] == 'm') else "{}m".format(args.nparams)
modelname = "ESM-2-{}".format(name)

truncate_transform = transforms.TruncateProtein(max_length=1024, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = YeastBio(D_PATH, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
train_set, valid_set, test_set = dataset.split()

model = CustomModel(path="/om2/user/oqueen/DeepSurface/esm/Yeast/", 
    model=modelname)

task = tasks.PropertyPrediction(
                   model, 
                   task=dataset.tasks,
                   criterion='bce', 
                   metric=('auroc', 'auprc'),
                   normalization=False,
                   num_mlp_layer=2
                   )

if args.frozen:
    for param in task.model.parameters():
        param.requires_grad = False


#logging.basicConfig(filename='/om2/user/oqueen/DeepSurface/esm/Yeast/results/{}.log'.format(modelname), filemode='w')
#logger = logging.getLogger()

optimizer = torch.optim.Adam(task.parameters(), lr=2e-4)
solver = core.Engine(
                   task, 
                   train_set, 
                   valid_set, 
                   test_set, 
                   optimizer,
                   gpus=[0], 
                   batch_size=32,
                   #logger='logging'
                   )

best_score = float("-inf")
best_epoch = -1

if not os.path.exists('/om2/user/oqueen/DeepSurface/esm/Yeast/models/{}/{}/'.format(D_NAME, modelname)):
    os.makedirs('/om2/user/oqueen/DeepSurface/esm/Yeast/models/{}/{}/'.format(D_NAME, modelname))


for i in range(1, ITERS+1):
    solver.model.split = 'train'
    solver.train(num_epoch=10)
    solver.save('/om2/user/oqueen/DeepSurface/esm/Yeast/models/{}/{}/epoch_{}.pth'.format(D_NAME, modelname, (solver.epoch*i)))

    solver.model.split = 'valid'
    metric = solver.evaluate('valid', log=True)

    score = []
    for k, v in metric.items():
        if k.startswith('auprc'):
            score.append(v)
    
    score = sum(score) / len(score)
    if score > best_score:
        best_score = score
        best_epoch = (solver.epoch * i)

solver.load('/om2/user/oqueen/DeepSurface/esm/Yeast/models/{}/{}/epoch_{}.pth'.format(D_NAME, modelname, best_epoch))

#with open('/om2/user/oqueen/DeepSurface/esm/Yeast/models/{}/best_epoch_{}.json'.format(modelname, best_epoch), 'w') as fout:
#    json.dump(solver.config_dict(), fout)

solver.save('/om2/user/oqueen/DeepSurface/esm/Yeast/models/{}/{}/best.pth'.format(D_NAME, modelname))

if not os.path.exists('/om2/user/oqueen/DeepSurface/esm/Yeast/results/{}/'.format(D_NAME)):
    os.makedirs('/om2/user/oqueen/DeepSurface/esm/Yeast/results/{}/'.format(D_NAME))

solver.model.split = 'valid'
eval_metrics = solver.evaluate('valid', log=True)
with open('/om2/user/oqueen/DeepSurface/esm/Yeast/results/{}/{}_eval_metrics.log.txt'.format(D_NAME, modelname), 'w') as f:
    f.write(str(eval_metrics))

solver.model.split = 'test'
test_metrics = solver.evaluate('test', log=True)
with open('/om2/user/oqueen/DeepSurface/esm/Yeast/results/{}/{}_test_metrics.log.txt'.format(D_NAME, modelname), 'w') as f:
    f.write(str(test_metrics))


