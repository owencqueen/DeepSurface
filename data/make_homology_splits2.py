import pickle
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold


def main():
    df = read_data()
    # Make 5 10-fold cross validations
    print(df.shape)

    inds = np.arange(df.shape[0])

    ind_lists = []

    for _ in range(5):
        all_inds = list(KFold(n_splits = 5, shuffle = True, random_state = 1234).split(inds))
        ind_lists.append(all_inds)

    pickle.dump(ind_lists, open('splits2.pkl', 'wb'))

def read_data():
    dpath = './yeast_bio.txt'

    reference = []
    evolved = []
    yeast_id = []
    with open(dpath, 'r') as f:
        lines = f.readlines()
        
        for i in range(0, len(lines), 3):
            yeast_id.append(re.sub('\W+', '', lines[i]))
            reference.append(re.sub('\W+', '', lines[i+1]))
            evolved.append(re.sub('\W+', '', lines[i+2]))

    df = pd.DataFrame({'yeast_id': yeast_id, 'reference.sequence' : reference, 'evolved.sequence' : evolved})
    return df

def test_load():
    inds = pickle.load(open('splits2.pkl', 'rb'))
    print('len', len(inds))

def example_load_data():
    df = read_data()

    # Load one split:
    split_no = 0
    inds = pickle.load(open('splits2.pkl', 'rb'))

    for train_inds, test_inds in inds[split_no]:
        train_set = df.iloc[train_inds,:]
        test_set = df.iloc[test_inds,:]
        train_seqs_list = train_set['reference.sequence'].tolist() + train_set['evolved.sequence'].tolist()
        train_seqs_labels = np.concatenate([np.zeros(train_set.shape[0]), np.ones(train_set.shape[0])])
        print('train list', len(train_seqs_list))
        print('train labels', train_seqs_labels.shape)

        test_seqs_list = test_set['reference.sequence'].tolist() + test_set['evolved.sequence'].tolist()
        test_seqs_labels = np.concatenate([np.zeros(test_set.shape[0]), np.ones(test_set.shape[0])])
        print('test list', len(test_seqs_list))
        print('test labels', test_seqs_labels.shape)
        print('-' * 30)
        


if __name__ == '__main__':
    #main()
    test_load()
    example_load_data()
