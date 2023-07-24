import torch
from torchdrug import datasets, transforms
from torchdrug import core, models, tasks, utils
import logging
import json
import os

PATH = '/lustre/isaac/scratch/ababjac/DeepSurface/'
D_PATH = PATH+'/data/'
ITERS = 10 # *10 = num_epochs
M_NAME = 'ProtLSTM'
D_NAME = 'BetaLactamase'

truncate_transform = transforms.TruncateProtein(max_length=1024, random=False)
protein_view_transform = transforms.ProteinView(view='residue')
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = datasets.BetaLactamase(D_PATH, atom_feature=None, bond_feature=None, residue_feature='default', transform=transform)
train_set, valid_set, test_set = dataset.split()

model = models.ProteinLSTM(
                   input_dim=dataset.node_feature_dim,
                   hidden_dim=640,
                   num_layers=3,
                   )

task = tasks.PropertyPrediction(
                   model, 
                   task=dataset.tasks,
                   criterion='mse', 
                   metric=('mae', 'rmse', 'spearmanr'),
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
                   batch_size=32,
                   #logger='logging'
                   )

best_score = float("-inf")
best_epoch = -1

if not os.path.exists(PATH+'/models/{}/{}/'.format(D_NAME, M_NAME)):
    os.makedirs(PATH+'/models/{}/{}/'.format(D_NAME, M_NAME))


for i in range(1, ITERS+1):
    solver.model.split = 'train'
    solver.train(num_epoch=10)
    solver.save(PATH+'/models/{}/{}/epoch_{}.pth'.format(D_NAME, M_NAME, (solver.epoch*i)))

    solver.model.split = 'valid'
    metric = solver.evaluate('valid', log=True)

    score = []
    for k, v in metric.items():
        if k.startswith('spearmanr'):
            score.append(v)
    
    score = sum(score) / len(score)
    if score > best_score:
        best_score = score
        best_epoch = (solver.epoch * i)

solver.load(PATH+'/models/{}/{}/epoch_{}.pth'.format(D_NAME, M_NAME, best_epoch))

#with open(PATH+'/models/{}/best_epoch_{}.json'.format(M_NAME, best_epoch), 'w') as fout:
#    json.dump(solver.config_dict(), fout)

solver.save(PATH+'/models/{}/{}/best.pth'.format(D_NAME, M_NAME))

if not os.path.exists(PATH+'/results/{}/'.format(D_NAME)):
    os.makedirs(PATH+'/results/{}/'.format(D_NAME))

solver.model.split = 'valid'
eval_metrics = solver.evaluate('valid', log=True)
with open(PATH+'/results/{}/{}_eval_metrics.log.txt'.format(D_NAME, M_NAME), 'w') as f:
    f.write(str(eval_metrics))

solver.model.split = 'test'
test_metrics = solver.evaluate('test', log=True)
with open(PATH+'/results/{}/{}_test_metrics.log.txt'.format(D_NAME, M_NAME), 'w') as f:
    f.write(str(test_metrics))


