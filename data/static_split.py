import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
df = pd.read_csv(dpath)

SEED = 1234
val_size = 0.025
test_size = 0.25

inds = np.arange(df.shape[0])
train_inds, test_inds = train_test_split(inds, test_size = val_size + test_size, random_state = SEED)
test_inds, val_inds = train_test_split(test_inds, test_size = val_size / test_size, random_state = SEED)

print('Train inds', train_inds.shape)
print('Test inds', test_inds.shape)
print('Val inds', val_inds.shape)

np.save('train_split.npy', train_inds)
np.save('val_split.npy', val_inds)
np.save('test_split.npy', test_inds)