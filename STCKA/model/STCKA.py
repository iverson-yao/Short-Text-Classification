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


    def self_attention(self, H):
        # H: batch_size, seq_len, 2*hidden_size
        hidden_size = H.size()[-1]
        Q = H
        K = H
        V = H
        # batch_size, seq_len, seq_len
        atten_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(hidden_size), -1)
        A = torch.bmm(atten_weight, V)  # batch_size, seq_len, 2*hidden_size
        A = A.permute(0, 2, 1)
        # q: short text representation
        q = F.max_pool1d(A, A.size()[2]).squeeze(-1)  # batch_size, 2*hidden_size

        return q

    def cst_attention(self, c, q):
        # c: batch_size, concept_seq_len, embedding_dim
        # q: batch_size, 2*hidden_size
        q = q.unsqueeze(1)
        q = q.expand(q.size(0), c.size(1), q.size(2))
        c_q = torch.cat((c, q), -1)  # batch_size, concept_seq_len, embedding_dim+2*hidden_size
        c_q = self.w1(F.tanh(self.W1(c_q)))  # batch_size, concept_seq_len, 1
        alpha = F.softmax(c_q.squeeze(-1), -1)  # batch_size, concept_seq_len

        return alpha

    def ccs_attention(self, c):
        # c: batch_size, concept_seq_len, embedding_dim
        c = self.w2(F.tanh(self.W2(c)))  # batch_size, concept_seq_len, 1
        beta = F.softmax(c.squeeze(-1), -1)  # batch_size, concept_seq_len

        return beta
