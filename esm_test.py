import torch
from datasets import load_dataset
from transformers import AutoTokenizer, EsmForSequenceClassification as ESMClf, EsmModel as ESM

def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    #model = ESMClf.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = ESM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    inputs = tokenizer("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", return_tensors="pt")
    
    out = model(**inputs)
    lhs = out.last_hidden_state # Per-token output
    po = out.pooler_output # Per-token
    print(lhs.shape)
    print(po.shape)

def load_data():
    
    dpath = '/lustre/isaac/proj/UTK0196/deep-surface-protein-data/M0059E_training_set.tsv'
    dataset = load_dataset("csv", data_files = dpath)

    print(dataset)

if __name__ == '__main__':
    load_data()