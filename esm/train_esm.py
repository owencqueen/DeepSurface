import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Huggingface:
from datasets import Dataset
from transformers import AutoTokenizer, EsmForSequenceClassification as ESMClf, EsmModel as ESM
from transformers import TrainingArguments, Trainer

DEBUG = False

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

    preds = epred[0]
    labels = epred[1]

    print('preds', preds)
    print('labels', labels)

    metrics = {}

    metrics['auprc'] =  0.5

    return metrics

def get_dataset(tokenizer, val_size = 0.02, test_size = 0.2):
    dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'

    df = pd.read_csv(dpath)
    inds = np.arange(df.shape[0]) # All indices of df

    # Splitting training, validation, testing
    train_inds, test_inds = train_test_split(inds, test_size = val_size + test_size)
    test_inds, val_inds = train_test_split(test_inds, test_size = val_size / test_size)

    # "Unravel" deep and surface columns into one dataset
    train_dict = unravel_df(df.iloc[train_inds,:])
    val_dict = unravel_df(df.iloc[val_inds,:])
    test_dict = unravel_df(df.iloc[test_inds,:])

    def tokenize_fn(examples): # Function to tokenize the sequences
        t = tokenizer(examples['text'], truncation = True, padding = 'max_length')
        return t
    
    # Convert all to Huggingface datsets:
    # These operations take a min:
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

    test_dataset = Dataset.from_dict(test_dict)
    if not DEBUG:
        test_dataset = test_dataset.map(tokenize_fn, batched = False)
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset

def get_training_args(args):
    targs = TrainingArguments(
        output_dir = '/lustre/isaac/scratch/oqueen/DeepSurface/esm/esm_out',
        overwrite_output_dir = True,
        num_train_epochs = args.epochs,
        learning_rate = args.lr,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = 50,
        fp16=True if torch.cuda.is_available() else False,
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
    parser.add_argument('--num_params', default = '8m', help = 'options = 8m, 35m, 650m')

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
    model = ESMClf.from_pretrained(modelpath).to(device)
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    data_tup = get_dataset(tokenizer)
    if DEBUG:
        data_tup = list(data_tup)
        tmp = data_tup[0]
        data_tup[0] = data_tup[1]
        data_tup[1] = tmp
    train_esm(model, data_tup, tokenizer, args)