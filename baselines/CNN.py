import torch
from torch import nn
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from skorch import NeuralNetworkClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_score. roc_auc_score

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

curr_dir = os.getcwd()


class Conv1DNN(nn.Module):
    def __init__(self, vocab_size=20, embed_len=32, target_classes=2):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_len)
        self.conv1 = nn.Conv1d(embed_len, 64, kernel_size=7, padding='valid')
        self.linear = nn.Linear(64, len(target_classes))
    
    def forward(self, x, embed_len=32, max_tokens=1024):
        embed = self.embedding_layer(x)
        embed = embed.reshape(len(embed), embed_len, max_tokens) ## Embedding Length needs to be treated as channel dimension
        x1 = nn.functional.relu(self.conv1(embed))
        x2, _ = x1.max(dim=-1)
        logits = self.linear(x2)

        return logits


if name == '__main__':
    data = pd.read_csv(curr_dir+'/data/deepsurface_1000.csv')
    print(data)

    MAX = 1024 #matching ESM

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=62835) #save 20% for testing

    X_train = df_train['codons_cleaned'].values
    y_train = df_train['sentiment'].values

    X_test = df_test['codons_cleaned'].values
    y_test = df_test['sentiment'].values

    print('Tokenizing...')
    tokenizer = Tokenizer(num_words=helpers.VOCAB_SIZE)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_seq = pad_sequences(X_train_seq, MAX)
    X_test_seq = pad_sequences(X_full_seq, MAX)

    model = NeuralNetBinaryClassifier(
        module=Conv1DNN,
        max_epochs=50,
        batch_size=8,
    )

    logits = model(X_test)
    y_prob = nn.Softmax(dim=1)(logits)
