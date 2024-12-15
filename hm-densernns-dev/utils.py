import random
import numpy as np
import string
import torch
import os
import datetime
import json
import logging
import torchtext


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def save_path(data_name, model_timestamp):
    # model_save_path
    model_save_dir = "../checkpoints/" + data_name + "/"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, "model_" + str(model_timestamp) + ".pt")
    return model_save_path


def save_model(model_save_path, model):
    with open(model_save_path, 'wb') as f:
        torch.save(model, f)


def save_params(data_name, model_params, model_timestamp):
    logger.info("Saving model parameters...")
    model_params_dir = "../params/" + data_name + "/"
    if not os.path.exists(model_params_dir):
        os.makedirs(model_params_dir)
    model_params_json_path = os.path.join(model_params_dir, "model_params_" + str(model_timestamp) + ".json")
    for key, item in model_params.items():
        if isinstance(item, datetime.datetime):
            model_params[key] = item.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(item, datetime.date):
            model_params[key] = item.strftime('%Y-%m-%d')
    with open(model_params_json_path, 'w') as fp:
        json.dump(model_params, fp)


def load_params(data_name, model_timestamp):
    logger.info("Loading model parameters...")
    model_params_dir = "../params/" + data_name + "/"
    if not os.path.exists(model_params_dir):
        os.makedirs(model_params_dir)
    model_params_json_path = os.path.join(model_params_dir, "model_params_" + str(model_timestamp) + ".json")
    with open(model_params_json_path, 'r') as fp:
        model_params = json.load(fp)
    return model_params


def build_model(model_init_path, model, reload_=False):
    # load or build the model
    if reload_ and os.path.exists(model_init_path):
        logger.info("Reloading model parameters from {}".format(model_init_path))
        with open(model_init_path, 'rb') as f:
            model = torch.load(f)
        logger.info("Successfully init model from {}.".format(model_init_path))
    else:
        logger.info(
            "Random Initialization Because:, reload_={}, path exists={}".format(
                reload_, os.path.exists(model_init_path))
        )
        logger.info("Build model...")
        # model = model.cuda()
        logger.info("Init model by randomization.")
    return model


def build_optimizer(optimizer_name, model, learning_rate, weight_decay):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("{} is not among ['sgd', 'adam']".format(optimizer_name))

    return optimizer


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def count_parameters(model):
    logger.info("\n\n")

    logger.info("Parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info("{}: {}".format(name, param.data.size()))

    num_vars = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Size of model parameters: {:,d}".format(num_vars))
    return num_vars


def init_configs(config_file):
    """Parse configurations from file."""
    with open(config_file) as json_file:
        config_args = json.load(json_file)

    torch.manual_seed(config_args["seed"])
    if config_args["cuda"]:
        torch.cuda.manual_seed(config_args["seed"])

    np.random.seed(int(config_args["seed"] * 13 / 7))
    return config_args


def train_validation_split(dataset, train_size=None, validation_size=None, shuffle=True):
    """Splits dataset into train and validation set.
    Args:
        dataset (Dataset): The dataset to be split.
        train_size (float): Fraction of dataset to be added to the train set, in range (0, 1).
        validation_size (float): Fraction of dataset treated as the validation set, in range (0, 1).
        Mutual exclusive with train_size.
        shuffle (bool, optional): If true, shuffle dataset before splitting.
    Returns:
        tuple: training dataset, validation dataset.
    """

    if train_size is None and validation_size is None:
        raise ValueError('Either train_size or validation_size must be given')

    examples = list(dataset.examples)
    if shuffle:
        random.shuffle(examples)

    train_size = train_size or (1. - validation_size)
    split_idx = int(train_size * len(examples))

    train_examples, val_examples = examples[:split_idx], examples[split_idx:]

    train_dataset = torchtext.data.Dataset(train_examples, dataset.fields)
    val_dataset = torchtext.data.Dataset(val_examples, dataset.fields)

    train_dataset.sort_key, val_dataset.sort_key = dataset.sort_key, dataset.sort_key

    return train_dataset, val_dataset
