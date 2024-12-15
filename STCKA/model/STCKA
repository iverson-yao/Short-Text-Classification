import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math


class STCK_Atten(nn.Module):
    def __init__(self, text_vocab_size, concept_vocab_size, embedding_dim, hidden_size, output_size, gama=0.5, num_layer=1, finetuning=True):
        super(STCK_Atten, self).__init__()

        self.gama = gama
        da = hidden_size
        db = int(da / 2)

        # Initialize word embeddings randomly
        self.txt_word_embed = nn.Embedding(text_vocab_size, embedding_dim)
        self.cpt_word_embed = nn.Embedding(concept_vocab_size, embedding_dim)

        # If you want to enable fine-tuning, set requires_grad accordingly
        self.txt_word_embed.weight.requires_grad = finetuning
        self.cpt_word_embed.weight.requires_grad = finetuning

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layer, batch_first=True, bidirectional=True)
        self.W1 = nn.Linear(2 * hidden_size + embedding_dim, da)
        self.w1 = nn.Linear(da, 1, bias=False)
        self.W2 = nn.Linear(embedding_dim, db)
        self.w2 = nn.Linear(db, 1, bias=False)
        self.output = nn.Linear(2 * hidden_size + embedding_dim, output_size)
