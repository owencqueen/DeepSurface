import torch
import argparse, os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Huggingface:
from datasets import Dataset
from transformers import AutoTokenizer, EsmForSequenceClassification as ESMClf, EsmModel as ESM
from transformers import TrainingArguments, Trainer

DEBUG = False # Set to use only validation dataset - saves loading time
SEED = 1234

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unravel_df(df):
    seq_surf = df['surf.sequence'].tolist()
    seq_deep = df['deep.sequence'].tolist()

    seqs = seq_surf + seq_deep

    label = list(np.concatenate([np.zeros(len(seq_surf)), np.ones(len(seq_deep))]).astype(int))
    print(len(label))
    
    newdf = {'text': seqs, 'label': label}
    return newdf

def compute_metrics(epred):
    # Computes metrics from specialized output from huggingface

    preds = np.exp(epred[0][:,1]) / np.sum(np.exp(epred[0]), axis = 1)
    labels = epred[1]

    metrics = {}
    metrics['auprc'] = average_precision_score(labels, preds)
    metrics['auroc'] = roc_auc_score(labels, preds)

    return metrics

def get_dataset(tokenizer, args, val_size = 0.02, test_size = 0.2, use_fixed_split = True, only_test = False):
    dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'

    df = pd.read_csv(dpath)
    inds = np.arange(df.shape[0]) # All indices of df

    # Splitting training, validation, testing
    if (args.fold_num is not None) and (args.split_path is not None):
        # INDEXING 1 AT FIXED VALUE FOR NOW
        train_inds, test_inds = pickle.load(open(args.split_path, 'rb'))[1][args.fold_num]
        train_inds, val_inds = train_test_split(train_inds, test_size = 0.05, random_state = SEED)
        print('Loading split path')
    elif use_fixed_split:
        split_path = '/lustre/isaac/scratch/oqueen/DeepSurface/data/fixed_split'
        train_inds = np.load(os.path.join(split_path, 'train_split.npy'))
        test_inds = np.load(os.path.join(split_path, 'test_split.npy'))
        val_inds = np.load(os.path.join(split_path, 'val_split.npy'))
    else:
        train_inds, test_inds = train_test_split(inds, test_size = val_size + test_size, random_state = SEED)
        test_inds, val_inds = train_test_split(test_inds, test_size = val_size / test_size, random_state = SEED)
    #exit()

    # "Unravel" deep and surface columns into one dataset
    train_dict = unravel_df(df.iloc[train_inds,:])
    val_dict = unravel_df(df.iloc[val_inds,:])
    test_dict = unravel_df(df.iloc[test_inds,:])

    def tokenize_fn(examples): # Function to tokenize the sequences
        t = tokenizer(examples['text'], truncation = True, padding = 'max_length')
        return t
    
    # Convert all to Huggingface datsets:
    # These operations take a min:
    
    test_dataset = Dataset.from_dict(test_dict)
    if not DEBUG:
        test_dataset = test_dataset.map(tokenize_fn, batched = False)
    else:
        test_dataset = None

    if not only_test:
        train_dataset = Dataset.from_dict(train_dict)
        if not DEBUG:
            train_dataset = train_dataset.map(tokenize_fn, batched = False)
        else:
            train_dataset = None

        val_dataset = Dataset.from_dict(val_dict)
        val_dataset = val_dataset.map(tokenize_fn, batched = False)
        print(val_dataset)
        tens = torch.tensor(val_dataset['input_ids'])
        print('val tensors', tens.shape)
        #print('val', type(val_dataset['input_ids'][0]))

    else:

        return test_dataset

    return train_dataset, val_dataset, test_dataset

def get_training_args(args):
    out_path = '/lustre/isaac/scratch/oqueen/DeepSurface/esm/esm_8m_out' if  args.out_path is None else args.out_path
    rname = 'split_{}_fixed'.format(args.fold_num) if args.fix_encoder else 'split_{}'.format(args.fold_num)
    targs = TrainingArguments(
        output_dir = out_path,
        overwrite_output_dir = True,
        num_train_epochs = args.epochs,
        learning_rate = args.lr,
        evaluation_strategy = 'steps',
        eval_steps = args.eval_steps,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 100,
        fp16=True if torch.cuda.is_available() else False,
        report_to = 'wandb',
        run_name = rname
    )
    return targs

def train_esm(model, datasets, tokenizer, args = None):

    targs = get_training_args(args)
    
    trainer = Trainer(
        model = model,
        args = targs,
        train_dataset = datasets[0],
        eval_dataset = datasets[1],
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    trainer.train()

if __name__ == '__main__':
    # Can parse args up here
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--lr', type=float, default = 1e-4)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--num_params', type=str, default = '8m', help = 'options = 8m, 35m, 650m')
    parser.add_argument('--eval_steps', type = int, default = 500, help = 'number of steps in between evaluations')
    parser.add_argument('--fix_encoder', action = 'store_true')
    parser.add_argument('--use_fixed_split', action = 'store_true')
    parser.add_argument('--split_path', type = str, default = None)
    parser.add_argument('--fold_num', type = int)
    parser.add_argument('--out_path', type = str, help = 'Directory path to output model results')


    args = parser.parse_args()

    assert args.num_params.lower() in ['8m', '35m', '650m'], "N params must be in ['8m', '35m', '650m']"

    # Assign model name for checkpoint to load from huggingface:
    if args.num_params.lower() == '8m':
        modelpath = "facebook/esm2_t6_8M_UR50D"
    elif args.num_params.lower() == '35m':
        modelpath = "facebook/esm2_t12_35M_UR50D"
    elif args.num_params.lower() == '650m':
        modelpath = "facebook/esm2_t33_650M_UR50D"
    
    # Get pretrained model weights
    
        # model = ESM.from_pretrained(modelpath).to(device)
        # for param in model.parameters(): # Freezes the encoder
        #     param.requires_grad = False

    model = ESMClf.from_pretrained(modelpath).to(device)
    if args.fix_encoder:
        # Fix all ESM parameters, leave only ESM prediction head:
        for param in model.esm.parameters():
            param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    data_tup = get_dataset(tokenizer, args, use_fixed_split =  args.use_fixed_split)
    if DEBUG:
        data_tup = list(data_tup)
        tmp = data_tup[0]
        data_tup[0] = data_tup[1]
        data_tup[1] = tmp
    train_esm(model, data_tup, tokenizer, args)