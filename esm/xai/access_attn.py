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

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, local_files_only=True)
    model = ESMClf.from_pretrained(args.ckpt_path, local_files_only=True)

    pstr = "MKEEEISNKKDKGISRRKFLGGAAATAAAFTIVPRHVLGGAGYTPPSEKLNVAGVGVGGMGGENIINVAGMERDKDRNLIKKREGENIVALCDVDEKFASDIFNAFHKAKKYKDFRKMLEKQ"
    print('len', len(pstr))
    t_pstr = tokenizer(pstr, return_tensors='pt')
    print(t_pstr)
    print('inp', t_pstr['input_ids'].shape[1])
    print('attn', t_pstr['attention_mask'].shape[1])

    #out = model(**t_pstr, output_attentions = True)
    out = model(**t_pstr, output_attentions = False)
    #attns = out.attentions

    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id

    print('cls', cls_token_id)
    print('sep', sep_token_id)
    print('pad', ref_token_id)
    print('eos', tokenizer.eos_token_id)

    print(out)
    #exit()

    # How to access:
    print('len pstr', len(pstr))
    #print('len attns', len(attns))
    print('attn', attns[0].shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type = str)
    args = parser.parse_args()

    main(args)