# HM-DenseRNNs
Code for our IJCAI-2019 paper "Recurrent Neural Network for Text Classification with Hierarchical Multiscale Dense Connections"


## Training
Generally, we feed a configuration file, which specifies the model and various other settings. 
Next, we will take HM-DenseGRU as an example to show its configurations. 


### HM-DenseGRU
The configuration is as follows:
```json
{
    "add_dense_block": true,
    "add_transition_function": false,
    "batch_first": true,
    "batch_size": 32,
    "bidirectional": false,
    "clip": 5.0,
    "cuda": true,
    "data_name": "sst",
    "debug": false,
    "dict_size": 17199,
    "display_iters": 100,
    "dropout_ratio": 0.3,
    "embed_size": 300,
    "epochs": 30,
    "fix_length": 0,
    "hidden_size_list": [64, 64, 64],
    "hierarchical": true,
    "include_lengths": true,
    "learning_rate": 0.001,
    "lr_scheduler": true,
    "lr_scheduler_gamma": 0.1,
    "lr_scheduler_milestones": [11, 21],
    "max_depth": 1,
    "max_len": 100,
    "mode": "train",
    "model_name": "dense_gru",
    "num_classes": 5,
    "optimizer_name": "adam",
    "output_hidden_size": 128,
    "save_dir": "saved_models/",
    "save_model": true,
    "seed": 1111,
    "split_ratio": 0.85,
    "train_word_embeddings": false,
    "user_predefined_vector_name": "",
    "weight_decay": 0.0
}
```
Running this configuration of HM-DenseGRU on SST-5 for five times would yield 47.11% (0.9382).
