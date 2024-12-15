# python学习日记
import torch
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader

# 使用torchtext 0.12+ 新的 API
# tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
#
# # 加载数据集
# train_data, test_data = IMDB(split=('train', 'test'))
#
# # 创建一个迭代器（示例）
# def collate_batch(batch):
#     text = [item[0] for item in batch]
#     label = [item[1] for item in batch]
#     return text, label
#
# train_iter = DataLoader(train_data, batch_size=32, collate_fn=collate_batch)
#
# # 打印部分数据
# for batch in train_iter:
#     print(batch)
#     break

# from torchtext.legacy.datasets import IMDB
# from torchtext.legacy.data import Field, LabelField
# import spacy
#
# # 显式加载 en_core_web_sm 模型
# spacy_en = spacy.load("en_core_web_sm")
#
# # 创建 Field 和 LabelField
# TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=spacy_en, include_lengths=True)
# LABEL = torchtext.legacy.data.LabelField(dtype=torch.float)
#
# # 加载数据
# train_data, test_data = IMDB.splits(TEXT, LABEL)
# print(len(train_data), len(test_data))
#
# # 打印数据的一部分
# print(vars(train_data.examples[0]))
import torch
#
from torchtext.datasets import IMDB
from torchtext.legacy.data import Field
import pickle
import spacy

# # 显式加载 en_core_web_sm 模型
# spacy_en = spacy.load("en_core_web_sm")
# tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
# # 定义文本和标签的处理方式
# TEXT = Field(sequential=True, tokenize=spacy_en, include_lengths=True)
# LABEL = Field(sequential=False, use_vocab=True, is_target=True)
# print(TEXT,LABEL)
# # 加载IMDB数据集
# train_data, test_data = IMDB(split=('train', 'test'))
# print("build")
# # 构建词汇表
# TEXT.build_vocab(train_data, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
# LABEL.build_vocab(train_data)
#
# print(f'Number of training examples: {len(train_data)}')
# print(f'Number of testing examples: {len(test_data)}')
#
# # 使用DataLoader批量加载数据
#
# # 保存处理后的数据集（可以使用pickle或torch.save）
# def save_data(loader, filename):
#     with open(filename, 'wb') as f:
#         pickle.dump(loader, f)
#
# save_data(train_data, 'train_data.pkl')
# save_data(test_data, 'test_data.pkl')
#
# # 在下次使用时重新加载
# def load_data(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)
#
# train_data = load_data('train_data.pkl')
# test_data = load_data('test_data.pkl')
#
# # 在下一次使用时加载保存的文件
# from torch.utils.data import DataLoader
#
# train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# 现在可以直接使用train_loader和test_loader进行训练和测试
# 显式加载 en_core_web_sm 模型
spacy_en = spacy.load("en_core_web_sm")
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

# 定义文本和标签的处理方式
TEXT = Field(sequential=True, tokenize=spacy_en, include_lengths=True)
LABEL = Field(sequential=False, use_vocab=True, is_target=True)

# 加载IMDB数据集
train_data, test_data = IMDB(split=('train', 'test'))

# 构建词汇表
TEXT.build_vocab(train_data, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# 保存词汇表和数据集
def save_vocab(vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)

def save_data(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

# 保存数据集和词汇表
save_data(train_data.examples, 'train_data_examples.pkl')  # 保存数据集的文本和标签
save_data(test_data.examples, 'test_data_examples.pkl')

save_vocab(TEXT.vocab, 'TEXT_vocab.pkl')  # 保存TEXT的词汇表
save_vocab(LABEL.vocab, 'LABEL_vocab.pkl')  # 保存LABEL的词汇表

# 加载数据集和词汇表
def load_vocab(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 重新加载数据集和词汇表
train_examples = load_data('train_data_examples.pkl')
test_examples = load_data('test_data_examples.pkl')
TEXT.vocab = load_vocab('TEXT_vocab.pkl')
LABEL.vocab = load_vocab('LABEL_vocab.pkl')

# 使用文本数据和标签重建 Dataset
from torchtext.legacy.data import Example, Dataset

# 创建自定义的 Dataset 对象
def create_dataset(examples, text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]
    return Dataset([Example.fromlist([example.text, example.label], fields) for example in examples], fields)

train_data = create_dataset(train_examples, TEXT, LABEL)
test_data = create_dataset(test_examples, TEXT, LABEL)

# 创建DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 验证
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')