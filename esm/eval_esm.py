import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Huggingface:
from datasets import Dataset
from transformers import AutoTokenizer, EsmForSequenceClassification as ESMClf, EsmModel as ESM
from transformers import TrainingArguments, Trainer

from train_esm import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def c_metrics(epred):
    preds = np.exp(epred[0][:,1]) / np.sum(np.exp(epred[0]), axis = 1)
    labels = epred[1]

    metrics = {}
    metrics['auprc'] = average_precision_score(labels, preds)
    metrics['auroc'] = roc_auc_score(labels, preds)

    return metrics

def main(args):

    model = ESMClf.from_pretrained(args.ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    args.fold_num = None
    args.split_path = None

    test = get_dataset(tokenizer, args = args, use_fixed_split = False, only_test = True)

    trainer = Trainer(
        model = model,
        eval_dataset = test,
        tokenizer = tokenizer,
        compute_metrics = c_metrics, # From train_esm
    )

    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    if args.out_path is not None:
        df = pd.DataFrame(eval_metrics)
        df.to_csv(args.out_path, index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type = str, default = None)
    parser.add_argument('--out_path', type = str, default = None)
    parser.add_argument('--split_no', type = int, required = True)
    parser.add_argument('--split_path', default = '/lustre/isaac/scratch/oqueen/DeepSurface/data/splits.pkl')
    args = parser.parse_args()

    print('Model', args.ckpt_path)

    main(args)