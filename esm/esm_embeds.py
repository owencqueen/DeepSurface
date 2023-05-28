import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Huggingface:
from datasets import Dataset
from transformers import AutoTokenizer, EsmForSequenceClassification as ESMClf, EsmModel as ESM
from transformers import TrainingArguments, Trainer

from train_esm import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def manual_batch_forwards(model, dataset, batch_size = 8):

    iters = np.arange(0, len(dataset), step = batch_size)

    labels = []
    z_list = []

    for i in trange(len(iters) - 1):
        ids = torch.tensor(dataset[iters[i]:iters[i+1]]['input_ids'], device = device)
        attn_masks = torch.tensor(dataset[iters[i]:iters[i+1]]['attention_mask'], device = device)
        out = model.esm(input_ids = ids, attention_mask = attn_masks, return_dict = True)
        labels += dataset[iters[i]:iters[i+1]]['label']

        # if i % 10:
        #     torch.cuda.empty_cache()

        z_list.append(out.last_hidden_state[:,0,:].detach().clone().cpu()) # Gets CLS token

    Z = torch.cat(z_list, dim = 0)
    return Z, labels

def main(args):

    model = ESMClf.from_pretrained(args.ckpt_path)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    args.fold_num = None
    args.split_path = None

    test = get_dataset(tokenizer, args = args, use_fixed_split = False, only_test = True)
    # def tokenize_fn(examples): # Function to tokenize the sequences
    #         t = tokenizer(examples['text'], truncation = True, padding = 'max_length')
    #         return t
    # test = test.map(tokenize_fn, batched = True)

    # ids = torch.tensor(test[5:9]['input_ids'], device = device)
    # attn_masks = torch.tensor(test[5:9]['attention_mask'], device = device)
    # out = model(input_ids = ids, attention_mask = attn_masks)

    Z, labels = manual_batch_forwards(model, test, batch_size = 4)
    torch.save((Z, labels), args.save_path)

    # Make manual dataloader:

    # trainer = Trainer(
    #     model = model,
    #     eval_dataset = test,
    #     tokenizer = tokenizer,
    #     compute_metrics = c_metrics, # From train_esm
    # )

    #eval_metrics = trainer.evaluate()
    # print(eval_metrics)

    # if args.out_path is not None:
    #     df = pd.DataFrame(eval_metrics)
    #     df.to_csv(args.out_path, index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type = str, default = None)
    parser.add_argument('--split_no', type = int, default = 0)
    parser.add_argument('--save_path', type = str)
    args = parser.parse_args()

    print('Model', args.ckpt_path)

    main(args)