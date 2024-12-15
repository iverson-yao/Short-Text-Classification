import torch.nn as nn
from torch.nn import Module
from torch.autograd import Variable
import torch
import logging
from models.dense_rnn import DenseRNNBase


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DenseRNNNet(Module):
    def __init__(self, hidden_size_list, dict_size, embed_size,
                 dropout_ratio, model_name, max_depth, out_hidden_size,
                 num_classes=2, batch_first=True,
                 loss_func=nn.CrossEntropyLoss,
                 bidirectional=True,
                 bias=True,
                 hierarchical=True, add_dense_block=True,
                 add_transition_function=False):
        super(DenseRNNNet, self).__init__()
        self.dict_size = dict_size
        self.hidden_size_list = hidden_size_list
        self.drop = nn.Dropout(p=dropout_ratio)
        self.embed_in = nn.Embedding(dict_size, embed_size)
        self.dropout_ratio = dropout_ratio
        self.use_all_layers = False
        self.model = DenseRNNBase(
            mode=model_name,
            input_size=embed_size,
            hidden_size=hidden_size_list[0],
            num_layers=len(hidden_size_list),
            batch_first=batch_first,
            bias=bias,
            dropout=dropout_ratio,
            start_dense_depth=max_depth,
            dense_depth_base=2,
            add_transition_function=add_transition_function,
            hierarchical=hierarchical,
            add_dense_block=add_dense_block
        )

        self.num_classes = num_classes
        self.out_hidden_size = out_hidden_size
        self.batch_first = batch_first
        print(num_classes)
        print(out_hidden_size)
        print(batch_first)

        if self.use_all_layers:
            self.rnn_hidden_size = 2 * sum(self.hidden_size_list) if bidirectional else sum(self.hidden_size_list)
        else:
            self.rnn_hidden_size = self.hidden_size_list[-1] * 2 if bidirectional else self.hidden_size_list[-1]
        self.model_name = model_name
        print("RNN Hidden Size:", self.rnn_hidden_size)


        if bidirectional:
            self.output_layer = nn.Linear(self.rnn_hidden_size, num_classes)
        else:
            self.output_layer = nn.Linear(self.rnn_hidden_size, num_classes)

        self.output_layer = nn.Linear(self.rnn_hidden_size, num_classes)
        # self.relu = nn.ReLU()
        print("Input dimension of output layer:", self.output_layer.weight.shape[1])

        self.loss = loss_func()

    def forward(self, inputs, hidden, lengths=None):
        emb = self.drop(self.embed_in(inputs))
        # embedded = [batch_size, time_steps, embed_size]
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
            packed_input, batch_sizes, _, _ = packed_emb
        else:
            packed_input = emb
            batch_sizes = None

        hidden_list, hidden_new, all_history_states = self.model(packed_input, hidden, batch_sizes)

        h = hidden_list[-1, :, :]

        return self.output_layer(self.drop(h)), hidden_new

    def init_hidden_states(self, batch_size):
        hidden = []
        for l in range(len(self.hidden_size_list)):
            h_l = Variable(torch.zeros(batch_size, self.hidden_size_list[l]).float().cuda(), requires_grad=False)
            c_l = Variable(torch.zeros(batch_size, self.hidden_size_list[l]).float().cuda(), requires_grad=False)
            hidden.append([h_l, c_l])
        return hidden
