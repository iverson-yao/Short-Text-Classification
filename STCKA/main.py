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




def main():
    args = config()

    if not args.train_data_path:
        logger.info("Please input train dataset path")
        exit()
    print(11111111111111111111111)
    # Load the IMDB dataset
    all_ = load_dataset(args.train_data_path, args.dev_data_path, args.test_data_path,
                                args.txt_embedding_path, args.cpt_embedding_path, args.train_batch_size,
                                args.dev_batch_size, args.test_batch_size)
    print(11111111111111111111111)
    txt_TEXT, cpt_TEXT, txt_vocab_size, cpt_vocab_size, txt_word_embeddings, cpt_word_embeddings, \
        train_iter, dev_iter, test_iter, label_size = all_
    print(11111111111111111111111)
    # Create the model
    model = STCK_Atten(txt_vocab_size, cpt_vocab_size, args.embedding_dim, txt_word_embeddings,
                       cpt_word_embeddings, args.hidden_size, label_size)

    # Move the model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("true")

    # Split train and dev data (IMDB data is already split in folders)
    train_data, test_data = train_test_split(train_iter, 0.8)
    train_data, dev_data = train_dev_split(train_data, 0.8)

    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # Load pre-trained model if specified
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        test_loss, acc, p, r, f1 = eval_model(model, test_data, loss_func)
        logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', test_loss, acc, p, r, f1)
        return

    # Training and evaluation loop
    best_score = 0.0
    test_loss, test_acc, test_p, test_r, test_f1 = 0, 0, 0, 0, 0
    for epoch in range(args.epoch):
        train_loss, eval_loss, acc, p, r, f1 = train_model(model, train_data, dev_data, epoch, args.lr, loss_func)

        logger.info('Epoch:%d, Training Loss:%.4f', epoch, train_loss)
        logger.info('Epoch:%d, Eval Loss:%.4f, Eval Acc:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f', epoch, eval_loss,
                    acc, p, r, f1)

        # Save the best model based on F1 score
        if f1 > best_score:
            best_score = f1
            torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score)))
            test_loss, test_acc, test_p, test_r, test_f1 = eval_model(model, test_data, loss_func)

        logger.info('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f', test_loss, test_acc,
                    test_p, test_r, test_f1)


if __name__ == "__main__":
    main()

