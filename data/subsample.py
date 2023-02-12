import numpy as np
import pandas as pd

dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
df = pd.read_csv(dpath)

# Randomly sample indices:
randints = np.random.choice(np.arange(df.shape[0]), size = (1000,), replace = False)

dfsub = df.iloc[randints,:]

dfsub.to_csv('deepsurface_1000.csv', index = False)