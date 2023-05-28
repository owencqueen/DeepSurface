import torch
import os, argparse
import numpy as np
import pandas as pd
from Bio import SeqIO

from transformers import AutoTokenizer, EsmForSequenceClassification as ESMClf, EsmModel as ESM
from transformers import TrainingArguments, Trainer

from captum.attr import LayerIntegratedGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# use this page: https://captum.ai/tutorials/Bert_SQUAD_Interpret

def get_all_proteins():
    fasta_dir_path = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/protein_analysis/fasta_files'
    id_to_seq = {}
    for f in os.listdir(fasta_dir_path):
        fpath = os.path.join(fasta_dir_path, f)

        s = list(SeqIO.parse(fpath, 'fasta'))[0]
        id_to_seq[s.id] = str(s.seq)
        
    return id_to_seq

def get_attr_values():
    
    q = get_all_proteins()
    ckpt_path = '../esm/model_outputs/esm_sp=2/checkpoint-2500'

    gt_df = pd.read_csv('true_labels.csv')

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = ESMClf.from_pretrained(ckpt_path)
    model.to(device)

    print('Model Loaded')

    eos_token_id = tokenizer.eos_token_id
    cls_token_id = tokenizer.cls_token_id
    ref_token_id = tokenizer.pad_token_id

    def gen_baseline(input_ids):
        L = [cls_token_id] + ([ref_token_id] * (input_ids.shape[1] - 2)) + [eos_token_id]
        return torch.tensor(L, device = device).unsqueeze(0)

    def forward_esm(input_tok, attention_mask): # Custom forward func for Captum
        #attn_mask = torch.ones_like(input_tok).to(input_tok.device)
        out = model(input_ids = input_tok, attention_mask = attention_mask).logits
        return out.max(1)[0]

    #out = model(**t_pstr, output_attentions = True)

    lig = LayerIntegratedGradients(forward_esm, model.esm.embeddings)

    pred_list = []
    flist = []
    attr_dict = {}

    for i, (k, v) in enumerate(q.items()):
        print('File:', k)

        t_pstr = tokenizer(v, return_tensors='pt').to(device)

        baseline = gen_baseline(t_pstr['input_ids'])
        # print('T_pstr', t_pstr['input_ids'].shape)
        # print('baseline', baseline.shape)

        with torch.no_grad():
            pred = model(**t_pstr).logits.argmax(dim=-1)
        print('Pred', pred)

        # Attr has to be accessed here:
        attr = lig.attribute(t_pstr['input_ids'], 
            baselines = baseline,
            additional_forward_args = (t_pstr['attention_mask']))
        #print('attr', attr.shape)

        pred_list.append(pred[0].item())
        flist.append(k)
        attr_dict[v] = attr.detach().clone().cpu().numpy()

    df = pd.DataFrame({'id':flist, 'pred':pred_list})
    modelname = ckpt_path.split('/')[-2]
    df.to_csv('pred_{}.csv'.format(modelname), index = False)

    np.save('attr_dict_{}.npy'.format(modelname), attr_dict)
    

if __name__ == '__main__':
    base_model = ''

    get_attr_values()
    #get_all_proteins()
