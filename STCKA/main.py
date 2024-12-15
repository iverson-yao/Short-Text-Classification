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


def load_dataset(train_data_path, dev_data_path, test_data_path, txt_wordVectors_path,
                 cpt_wordVectors_path, train_batch_size, dev_batch_size, test_batch_size):
    """
    加载数据集，构建词汇表和数据迭代器
    """
    print(22222)
    tokenize = lambda x: x.split()

    # 定义文本字段 (text)，用<pad>进行填充
    txt_TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token='<pad>',
                          lower=True, include_lengths=True, batch_first=True)

    # 定义标签字段 (label)，没有unk_token
    LABEL = data.Field(sequential=False, batch_first=True, unk_token=None)

    # 加载文本文件
    def get_text_files(path, label=None):
        text_files = glob.glob(os.path.join(path, '*.txt'))
        return TextDataset(text_files, label=label)

    # 加载训练集，正面评论和负面评论
    train_pos_files = get_text_files(os.path.join(train_data_path, 'pos'), label=1)
    train_neg_files = get_text_files(os.path.join(train_data_path, 'neg'), label=0)
    train_data = train_pos_files + train_neg_files

    # 加载测试集
    test_pos_files = get_text_files(os.path.join(test_data_path, 'pos'), label=1)
    test_neg_files = get_text_files(os.path.join(test_data_path, 'neg'), label=0)
    test_data = test_pos_files + test_neg_files

    # 如果有未标注的训练数据
    unsup_data = []
    if os.path.exists(os.path.join(train_data_path, 'unsup')):
        unsup_files = get_text_files(os.path.join(train_data_path, 'unsup'))
        unsup_data = unsup_files  # 未标注数据没有标签
        print("have unsup_data")

    # 构建词汇表
    txt_TEXT.build_vocab(train_data)
    print(3333)
    # 构建标签词汇表
    LABEL.build_vocab(train_data)

    # 创建数据迭代器
    train_iter = data.Iterator(train_data, batch_size=train_batch_size,
                               train=True, sort=False, repeat=False, shuffle=True)
    dev_iter = None  # 如果需要验证集数据，可以加上处理代码
    test_iter = data.Iterator(test_data, batch_size=test_batch_size,
                              train=False, sort=False, repeat=False, shuffle=False)

    # 获取词汇表的大小
    txt_vocab_size = len(txt_TEXT.vocab)
    label_size = len(LABEL.vocab)

    return txt_TEXT, txt_vocab_size, train_iter, test_iter, label_size


def train_test_split(all_iter, ratio):
    """
    根据给定比例划分数据集为训练集和测试集。
    """
    length = len(all_iter)
    train_data = []
    test_data = []
    train_end = int(length * ratio)

    for ind, batch in enumerate(all_iter):
        if ind < train_end:
            train_data.append(batch)
        else:
            test_data.append(batch)

    return train_data, test_data

def train_dev_split(train_iter, ratio):
    length = len(train_iter)
    train_data = []
    dev_data = []
    train_start = 0
    train_end = int(length*ratio)
    ind = 0
    for batch in train_iter:
        if ind < train_end:
            train_data.append(batch)
        else:
            dev_data.append(batch)
        ind += 1
    return train_data, dev_data



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, dev_iter, epoch, lr, loss_func):
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    all_loss = 0.0
    model.train()
    ind = 0.0
    for idx, batch in enumerate(train_iter):
        txt_text = batch.text[0]
        cpt_text = batch.concept[0]
        # batch_size = text.size()[0]
        target = batch.label

        if torch.cuda.is_available():
            txt_text = txt_text.cuda()
            cpt_text = cpt_text.cuda()
            target = target.cuda()

        optim.zero_grad()
        # pred: batch_size, output_size
        logit = model(txt_text, cpt_text)

        loss = loss_func(logit, target)

        loss.backward()
        # clip_gradient(model, 1e-1)
        optim.step()

        if idx % 10 == 0:
            logger.info('Epoch:%d, Idx:%d, Training Loss:%.4f', epoch, idx, loss.item())
            # dev_iter_ = copy.deepcopy(dev_iter)
            # p, r, f1, eval_loss = eval_model(model, dev_iter, id_label)
        all_loss += loss.item()
        ind += 1

    eval_loss, acc, p, r, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    eval_loss, acc, p, r, f1 = eval_model(model, dev_iter, loss_func)
    # return all_loss/ind
    return all_loss / ind, eval_loss, acc, p, r, f1


def eval_model(model, val_iter, loss_func):
    eval_loss = 0.0
    ind = 0.0
    score = 0.0
    pred_label = None
    target_label = None
    # flag = True
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_iter)):
            txt_text = batch.text[0]
            cpt_text = batch.concept[0]
            # batch_size = text.size()[0]
            target = batch.label

            if torch.cuda.is_available():
                txt_text = txt_text.cuda()
                cpt_text = cpt_text.cuda()
                target = target.cuda()
            logit = model(txt_text, cpt_text)

            loss = loss_func(logit, target)
            eval_loss += loss.item()
            if ind > 0:
                pred_label = torch.cat((pred_label, logit), 0)
                target_label = torch.cat((target_label, target))
            else:
                pred_label = logit
                target_label = target

            ind += 1

    acc, p, r, f1 = metrics.assess(pred_label, target_label)
    return eval_loss / ind, acc, p, r, f1

# Set up logger
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)



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

