import torch
import torchtext
from torchtext.vocab import Vectors
import torch.nn as nn
import time
import json
import os
import logging
import argparse
import random
from utils import build_optimizer, init_configs
from models.dense_rnn_net import DenseRNNNet
# from dataset.ag_news_corpus import AGNewsCorpus
from torchtext.data.utils import get_tokenizer
from torchtext.legacy.data import Field, LabelField
from torchtext.legacy.datasets import AG_NEWS, SST, IMDB
from torchtext.legacy.data import BucketIterator
import spacy
spacy_en = spacy.load('en_core_web_sm')
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
print(1)
# import nltk
# nltk.download('punkt_tab')  # 下载punkt模型

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calc_acc(preds, target, num_classes=1):
    if num_classes == 1:
        rounded_preds = torch.round(torch.sigmoid(preds))
    else:
        _, rounded_preds = torch.max(preds, 1)
        rounded_preds = rounded_preds.squeeze(-1)
    correct = (rounded_preds == target).float()  # convert into float for division
    acc = correct.sum() / correct.size(0)
    return acc


def evaluate(config_args, data_iter, model, criterion, show_progress=False, model_name="hmlstm"):
    model.eval()  # turn on the evaluation mode which disables dropout
    total_loss = 0

    running_correct = 0
    with torch.no_grad():
        count = 0
        for batch in data_iter:
            if model_name in ["dense_rnn", "dense_gru", "dense_lstm"]:
                batch_text, batch_lengths = batch.text
                preds, _ = model(batch_text.cuda(), hidden=None, lengths=batch_lengths)
            else:
                batch_text, batch_lengths = batch.text
                preds, _ = model(batch_text.cuda(), batch_lengths)
            if config_args["num_classes"] == 1:
                preds = preds.squeeze(1)
            y_tensor = batch.label.long()
            if config_args["cuda"]:
                y_tensor = y_tensor.cuda()
            loss = criterion(preds, y_tensor)
            total_loss += loss.data
            running_correct += calc_acc(preds, y_tensor, config_args["num_classes"])

            count += 1

            if show_progress:
                if count % (len(data_iter) // 10) == 0:
                    logger.info("Testing, {} / {}".format(count, len(data_iter)))

    model.train()
    return total_loss / len(data_iter), running_correct / len(data_iter)

import spacy

# 使用正确的模型名称



# def load_data(data_name, fix_length, batch_first, include_lengths, split_ratio=0.85):
#     logger.info("prepare data...")
#     if fix_length > 0:
#         text_field = torchtext.data.Field(
#             batch_first=batch_first, include_lengths=include_lengths,
#             fix_length=fix_length, tokenize="spacy"
#         )
#     else:
#         print(torchtext.data.Field)
#         text_field = torchtext.data.Field(
#             batch_first=batch_first, include_lengths=include_lengths,
#             tokenize="spacy"
#         )
#     label_field = torchtext.data.LabelField(dtype=torch.float)
#     if data_name == "ag_news":
#         train_data, test_data = AGNewsCorpus.splits(text_field, label_field)
#         train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state=random.seed(1234))
#     elif data_name == "sst":
#         train_data, valid_data, test_data = torchtext.datasets.SST.splits(text_field, label_field, fine_grained=True)
#     else:
#         train_data, test_data = torchtext.datasets.IMDB.splits(text_field, label_field)
#         train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state=random.seed(1234))
#     logger.info("train: {}, valid: {}, test: {}".format(len(train_data), len(valid_data), len(test_data)))
#     return text_field, label_field, train_data, valid_data, test_data

def load_data(data_name, fix_length, batch_first, include_lengths, split_ratio=0.85):
    logger.info("prepare data...")
    print(2)
    # 设置text_field
    if fix_length > 0:
        text_field = Field(
            batch_first=batch_first, include_lengths=include_lengths,
            fix_length=fix_length, tokenize="spacy"
        )
    else:
        print(3)
        text_field = Field(
            batch_first=batch_first, include_lengths=include_lengths,
            tokenize="spacy"
        )
    print(4)
    # 设置label_field
    label_field = LabelField(dtype=torch.float)
    print(f"text_field attributes: \n- tokenize: {text_field.tokenize}\n- batch_first: {text_field.batch_first}\n- include_lengths: {text_field.include_lengths}")
    print(f"label_field attributes: \n- dtype: {label_field.dtype}")
    # 加载不同数据集
    # if data_name == "ag_news":
    #         # train_data, test_data = AG_NEWS.splits(text_field, label_field)
    #         # train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state=random.seed(1234))
    # elif data_name == "sst":
    #     train_data, valid_data, test_data = SST.splits(text_field, label_field, fine_grained=True)
    # else:  # 默认加载IMDB数据集
    if data_name == "imdb":

        train_data, test_data = IMDB.splits(text_field, label_field)
        print("done")
        print(f"Train data length: {len(train_data)}")
        if len(train_data) == 0:
            raise ValueError("train_data is empty! Please check your dataset.")

    # 将训练数据集分割成训练集和验证集
    train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state=random.seed(1234))
    print("done")
    # 输出数据集大小
    logger.info("train: {}, valid: {}, test: {}".format(len(train_data), len(valid_data), len(test_data)))

    return text_field, label_field, train_data, valid_data, test_data


# def load_data(data_name, fix_length, batch_first, include_lengths, split_ratio=0.85):
#     logger.info("prepare data...")
#     def nltk_tokenizer(text):
#         return nltk.word_tokenize(text)
#
#     if fix_length > 0:
#         text_field = torchtext.data.Field(
#             batch_first=batch_first, include_lengths=include_lengths,
#             fix_length=fix_length, tokenize=nltk_tokenizer
#         )
#     else:
#         text_field = torchtext.data.Field(
#             batch_first=batch_first, include_lengths=include_lengths,
#             tokenize=nltk_tokenizer
#         )
#     label_field = torchtext.data.LabelField(dtype=torch.float)
#     if data_name == "ag_news":
#         train_data, test_data = AGNewsCorpus.splits(text_field, label_field)
#         train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state=random.seed(1234))
#     elif data_name == "sst":
#         train_data, valid_data, test_data = torchtext.datasets.SST.splits(text_field, label_field, fine_grained=True)
#     else:
#         train_data, test_data = torchtext.datasets.IMDB.splits(text_field, label_field)
#         print(f"训练数据样本数: {len(train_data)}")
#         print(split_ratio)
#         train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state=random.seed(1234))
#     logger.info("train: {}, valid: {}, test: {}".format(len(train_data), len(valid_data), len(test_data)))
#     return text_field, label_field, train_data, valid_data, test_data


def build_vocab(text_field, label_field, train_data, embed_size, user_predefined_vector_name):
    logger.info("Build vocabulary...")
    if not user_predefined_vector_name:
        vector_name = "glove.6B.100d" if embed_size == 100 else "glove.42B.300d"
        text_field.build_vocab(train_data, vectors=vector_name, max_size=25000)
        label_field.build_vocab(train_data)
    else:
        vectors = Vectors(name=user_predefined_vector_name, cache=".vector_cache/")
        text_field.build_vocab(train_data, vectors=vectors, max_size=25000)
        label_field.build_vocab(train_data, vectors=vectors)


def load_model(model_name, config_args, text_field):
    if model_name in ["dense_rnn", "dense_gru", "dense_lstm"]:
        model = DenseRNNNet(
            hidden_size_list=config_args["hidden_size_list"],
            dict_size=config_args["dict_size"],
            embed_size=config_args["embed_size"],
            dropout_ratio=config_args["dropout_ratio"],
            model_name=model_name,
            max_depth=config_args["max_depth"],
            out_hidden_size=config_args["output_hidden_size"],
            num_classes=config_args["num_classes"],
            batch_first=config_args["batch_first"],
            bidirectional=config_args["bidirectional"],
            hierarchical=config_args["hierarchical"],
            add_dense_block=config_args["add_dense_block"],
            add_transition_function=config_args["add_transition_function"] if "add_transition_function" in config_args else False
        )
    else:
        raise ValueError("choose model from 'dense_rnn', 'dense_gru', 'dense_lstm'")

    model.embed_in.weight.data.copy_(text_field.vocab.vectors)
    model.embed_in.weight.requires_grad = config_args["train_word_embeddings"]
    if config_args["cuda"]:
        model = model.cuda()
    return model


def add_new_parameters(optimizer, model, checked_parameters):
    new_params_group = []
    for name, param in model.named_parameters():
        if "dense_weight_hh" in name and name not in checked_parameters:
            checked_parameters.add(name)
            new_params_group.append(param)
    optimizer.add_param_group({"params": new_params_group})


def epoch_train(epoch, train_iter, optimizer, config_args, model, criterion, checked_parameters):
    total_loss = 0
    display_iters = config_args["display_iters"]

    time_it_start = time.time()
    running_correct = 0
    it = 0
    for batch in train_iter:
        optimizer.zero_grad()
        batch_text, batch_lengths = batch.text
        if config_args["cuda"]:
            batch_text = batch_text.cuda()
        if config_args["model_name"] in ["dense_rnn", "dense_gru", "dense_lstm"]:
            preds, hidden_list = model(batch_text, hidden=None, lengths=batch_lengths)
        else:
            preds, hidden_list = model(batch_text, batch_lengths)

        if config_args["num_classes"] == 1:
            preds = preds.squeeze(1)
        # logger.info("preds: {}, batch_label: {}".format(preds.size(), batch.label.size()))
        y_tensor = batch.label.long()
        if config_args["cuda"]:
            y_tensor = y_tensor.cuda()
        train_loss = criterion(preds, y_tensor)
        running_correct += calc_acc(preds, y_tensor, config_args["num_classes"]).data

        train_loss.backward()
        if config_args["clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config_args["clip"])
        optimizer.step()

        total_loss += train_loss.data
        it += 1
        add_new_parameters(optimizer, model, checked_parameters)

        if it > 0 and it % display_iters == 0:
            cur_loss = total_loss / display_iters
            logger.info(
                "Epoch: {}, iters: {} / {}, training loss: {:.4f}, "
                "training accuracy: {:.4f}, "
                "using {:.2f} seconds\n".format(
                    epoch, it, len(train_iter), cur_loss,
                    running_correct / display_iters,
                    time.time() - time_it_start
                )
            )

            total_loss = 0
            running_correct = 0
            if not config_args["debug"]:
                time_it_start = time.time()


def save_model_and_configs(config_args, model):
    if not os.path.exists(config_args["model_save_name"]):
        os.makedirs(config_args["model_save_name"])
    model_file = os.path.join(config_args["model_save_name"], config_args["model_name"] + '.bin')
    logger.info("Save model to [{}]".format(model_file))
    torch.save(model.state_dict(), model_file)

    model_config_file = os.path.join(
        config_args["model_save_name"], config_args["model_name"] + '_config.json'
    )
    if not os.path.exists(model_config_file):
        with open(model_config_file, "w") as outfile:
            json.dump(config_args, outfile, indent=4, sort_keys=True)


def train(config_args):
    time_init = time.time()
    text_field, label_field, train_data, valid_data, test_data = load_data(
        data_name=config_args["data_name"],
        fix_length=config_args["fix_length"],
        batch_first=config_args["batch_first"],
        include_lengths=config_args["include_lengths"],
        split_ratio=config_args["split_ratio"]
    )

    build_vocab(
        text_field, label_field, train_data,
        config_args["embed_size"],
        config_args["user_predefined_vector_name"]
    )

    train_iter, valid_iter, test_iter = torchtext.legacy.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config_args["batch_size"],
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
    )
    dict_size = len(text_field.vocab)
    # model_params["dict_size"] = dict_size
    config_args["dict_size"] = dict_size

    model = load_model(config_args["model_name"], config_args, text_field)
    if config_args["num_classes"] == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if config_args["cuda"]:
        criterion = criterion.cuda()

    optimizer = build_optimizer(
        config_args["optimizer_name"],
        model,
        config_args["learning_rate"],
        config_args["weight_decay"]
    )
    checked_parameters = set()

    if config_args["lr_scheduler"]:
        if "lr_scheduler_milestones" in config_args and config_args["lr_scheduler_milestones"]:
            lr_scheduler_milestones = config_args["lr_scheduler_milestones"]
        else:
            lr_scheduler_milestones = [11, 21]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_scheduler_milestones, gamma=config_args["lr_scheduler_gamma"]
        )
    else:
        scheduler = None

    start_time = time.time()
    best_valid_loss = 1000.0 * 1000.0
    best_valid_acc = 0.0

    for epoch in range(config_args["epochs"]):
        epoch_begin_time = time.time()
        if config_args["lr_scheduler"]:
            scheduler.step(epoch)

        epoch_train(epoch, train_iter, optimizer, config_args, model, criterion, checked_parameters)
        valid_loss, valid_acc = evaluate(
            config_args, valid_iter, model, criterion, model_name=config_args["model_name"]
        )
        logger.info(
            "Epoch: {}, valid loss: {:.4f}, best valid_loss: {:.4f}, "
            "valid accuracy: {:.4f}, best_valid_accuracy: {:.4f}, "
            "using {:.2f} seconds\n".format(
                epoch, valid_loss, best_valid_loss,
                valid_acc, best_valid_acc,
                time.time() - epoch_begin_time
            )
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            # save_model(model_save_path, model)
            if config_args["save_model"]:
                save_model_and_configs(config_args, model)

    logger.info(
        "Training done, best_valid_loss: {:.4f}, best_valid_acc: {:.4f}, using {:.2} seconds\n".format(
            best_valid_loss, best_valid_acc, time.time() - start_time
        )
    )
    # raise RuntimeError("Do not test when debugging")
    logger.info("Testing...")
    # model.load_state_dict(torch.load(config_args["load_model_path"]))
    # model = torch.load(model_save_path)
    test_loss, test_acc = evaluate(
        config_args,
        test_iter, model, criterion, show_progress=True, model_name=config_args["model_name"]
    )
    logger.info("test_loss: {:.4f}, test_acc: {:.4f}, using {:.2f} seconds\n".format(
        test_loss, test_acc, time.time() - time_init))


def test(config_args):
    text_field, label_field, train_data, valid_data, test_data = load_data(
        data_name=config_args["data_name"],
        fix_length=config_args["fix_length"],
        batch_first=config_args["batch_first"],
        include_lengths=config_args["include_lengths"],
        split_ratio=config_args["split_ratio"]
    )

    build_vocab(
        text_field, label_field, train_data,
        config_args["embed_size"],
        config_args["user_predefined_vector_name"]
    )

    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config_args["batch_size"],
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
    )
    dict_size = len(text_field.vocab)
    # model_params["dict_size"] = dict_size
    config_args["dict_size"] = dict_size

    model = load_model(config_args["model_name"], config_args, text_field)
    model.load_state_dict(torch.load(config_args["load_model_path"]))
    model.eval()

    if config_args["num_classes"] == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if config_args["cuda"]:
        criterion = criterion.cuda()

    test_loss, test_acc = evaluate(
        config_args,
        test_iter, model, criterion, show_progress=True, model_name=config_args["model_name"]
    )
    logger.info("test_loss: {:.2f}, test_acc: {:.2f}".format(test_loss, test_acc))
    logger.info("Test done!")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file', default="", type=str, required=False, help="configuration parameters")
    arg_parser.add_argument('--mode', default="train", type=str, required=False, help="train or test")
    arg_parser.add_argument('--data_name', default="imdb", type=str, required=False, help="imdb, sst, ag")
    arg_parser.add_argument('--load_model_path', default="", type=str, required=False, help="saved model")
    arg_parser.add_argument('--timestamp', default=str(time.time() * 1000), type=str, required=False, help="timestamp")
    arg_parser.add_argument('--model_name', default="dense_gru", type=str, required=False, help="model name")
    arg_parser.add_argument("--seed", default=0, type=int, required=False, help="seed")
    args = arg_parser.parse_args()

    if args.mode == "test":
        saved_model_config_file = os.path.join(
            args.load_model_path, args.timestamp,
            args.model_name + "_" + args.data_name + "_config.json"
        )
        logger.info("Reading configuration from [{}]".format(saved_model_config_file))
        configs = init_configs(saved_model_config_file)
    else:
        configs = init_configs(args.config_file)
        configs["model_save_name"] = os.path.join(configs["save_dir"], args.data_name, args.timestamp)
        configs["model_timestamp"] = args.timestamp
        configs["data_name"] = args.data_name
        if args.seed != 0:
            configs["seed"] = args.seed

    configs["mode"] = args.mode
    configs["load_model_path"] = os.path.join(configs["model_save_name"], configs["model_name"] + '.bin')

    if args.mode == 'train':
        print(configs)
        train(configs)
    elif args.mode == 'test':
        test(configs)
    else:
        raise RuntimeError("Unknown mode!")
