import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse

from model.STCKA import STCK_Atten

from utils import metrics
import copy
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def config():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument("--epoch", default=100, type=int,
                        help="the number of epochs needed to train")
    parser.add_argument("--lr", default=2e-5, type=float,
                        help="the learning rate")
    parser.add_argument("--train_data_path", default="dataset/aclImdb/train", type=str,
                        help="train dataset path (IMDB dataset train folder containing 'pos' and 'neg' subfolders)")
    parser.add_argument("--dev_data_path", default="dataset/aclImdb/test", type=str,
                        help="dev dataset path (IMDB dataset test folder containing 'pos' and 'neg' subfolders)")
    parser.add_argument("--test_data_path", default="dataset/aclImdb/test", type=str,
                        help="test dataset path (IMDB dataset test folder containing 'pos' and 'neg' subfolders)")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--dev_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="the batch size")
    # parser.add_argument("--txt_embedding_path", default="dataset/glove.6B.300d.txt", type=str,
    #                     help="pre-trained word embeddings path (e.g., GloVe embeddings)")
    # parser.add_argument("--cpt_embedding_path", default="dataset/glove.6B.300d.txt", type=str,
    #                     help="pre-trained word embeddings path (e.g., GloVe embeddings for concepts)")
    parser.add_argument("--txt_embedding_path", default=None, type=str,
                        help="pre-trained word embeddings path (e.g., GloVe embeddings)")
    parser.add_argument("--cpt_embedding_path", default=None, type=str,
                        help="pre-trained word embeddings path (e.g., GloVe embeddings for concepts)")
    parser.add_argument("--embedding_dim", default=300, type=int,
                        help="the text/concept word embedding size")
    parser.add_argument("--hidden_size", default=128, type=int,
                        help="the hidden size")
    parser.add_argument("--output_size", default=2, type=int,
                        help="the output size (for binary classification: positive/negative)")
    parser.add_argument("--fine_tuning", default=True, type=bool,
                        help="whether fine-tune word embeddings")
    parser.add_argument("--early_stopping", default=15, type=int,
                        help="Tolerance for early stopping (# of epochs).")
    parser.add_argument("--load_model", default=None,
                        help="load pretrained model for testing")
    args = parser.parse_args()

    return args

import os
import torch
from torchtext.legacy import data
from torchtext.vocab import Vectors
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import glob



class TextDataset(Dataset):
    def __init__(self, text_files, label=None, tokenize=lambda x: x.split()):
        """
        初始化数据集，文本文件路径和标签
        :param text_files: 文本文件路径列表
        :param label: 标签 (正面 1，负面 0)，未标注数据可以传入 None
        :param tokenize: 分词函数
        """
        self.text_files = text_files
        self.labels = [label] * len(text_files) if label is not None else [None] * len(text_files)
        self.tokenize = tokenize

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        with open(self.text_files[idx], 'r', encoding='utf-8') as f:
            text = f.read().strip()  # 读取文本并去除首尾空白
            print("done")
        return {'text': self.tokenize(text), 'label': self.labels[idx]}

