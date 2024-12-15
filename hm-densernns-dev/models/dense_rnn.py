import math
import torch
from torch.nn import Module, Parameter


class DenseRNNCellBase(Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias, num_chunks,
                 dense_depth=1, dense_depth_base=2, add_transition_function=True,
                 layer=0, hierarchical=True, add_dense_block=True):
        super(DenseRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dense_depth = dense_depth
        self.dense_depth_base = dense_depth_base
        self.add_transition_function = add_transition_function
        self.layer = layer
        self.hierarchical = hierarchical
        self.add_dense_block = add_dense_block

        self.weight_ih = Parameter(torch.Tensor(input_size, num_chunks * hidden_size))
        # self.weight_hh = torch.nn.ParameterList([Parameter(torch.Tensor(hidden_size, num_chunks * hidden_size))])
        self.weight_hh = Parameter(torch.Tensor(hidden_size, num_chunks * hidden_size))

        # self.dense_weight_hh = torch.nn.ParameterDict([
        #     [
        #         "dense_weight_hh_{}".format(dense_depth_i),
        #         None
        #     ] for dense_depth_i in range(self.dense_depth)
        # ])
        for dense_depth_i in range(self.dense_depth):
            self.register_parameter("dense_weight_hh_{}".format(dense_depth_i), None)

        if self.add_transition_function:
            self.weight_hth = Parameter(torch.Tensor(hidden_size, num_chunks * hidden_size))
            if self.bias:
                self.bias_hth = Parameter(torch.Tensor(num_chunks * hidden_size))
            else:
                self.register_parameter('bias_hth', None)
        else:
            self.register_parameter('weight_hth', None)

        if bias:
            self.bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
            # weight.data.fill_(0.001)

    def transition_function(self, hx):
        if self.bias:
            dense_h = torch.mm(hx, self.weight_hth) + self.bias_hth
        else:
            dense_h = torch.mm(hx, self.weight_hth)
        return dense_h

    def concat_states_hierarchical(self, hx, time_step, hx_list, cell_state_pos=0):
        """Concatenate previous hidden states in a hierarchical sparse order."""
        total_depth = time_step % self.dense_depth

        if total_depth <= 1:
            return 0, [hx]

        depth = 1
        valid_depth = 0

        dense_hx_list = [hx]
        while depth < total_depth:
            if self.layer == 0:
                dense_hx_list += [hx_list[time_step - 1 - depth][self.layer][cell_state_pos][:hx.size(0)]]
                valid_depth += 1
            elif depth % (self.dense_depth // self.dense_depth_base) == (self.dense_depth // self.dense_depth_base) - 1:
                # This is to say, the previous step would be overlooked if it is not the end of the below dense block
                # say, the current dense block is 8, and the dense_depth_base is 4, then only step 3 would be considered
                # at step 7. This would introduce hierarchical sparse updating utility
                dense_hx_list += [hx_list[time_step - 1 - depth][self.layer][cell_state_pos][:hx.size(0)]]
                valid_depth += 1
            depth += 1

        return valid_depth, dense_hx_list

    def concat_states_high_order_or_stack(self, hx, time_step, hx_list, cell_state_pos=0):
        """Concatenate previous hidden states in high or stack order."""
        total_depth = time_step % self.dense_depth

        if total_depth <= 1:
            return 0, [hx]

        depth = 1
        valid_depth = 0
        dense_hx_list = [hx]

        while depth < total_depth and time_step - 1 - depth >= 0:
            dense_hx_list += [hx_list[time_step - 1 - depth][self.layer][cell_state_pos][:hx.size(0)]]
            valid_depth += 1
            depth += 1
        return valid_depth, dense_hx_list

    def concat_prev_states(self, hx, time_step, hx_list, cell_state_pos=0):
        if self.hierarchical and self.add_dense_block:
            return self.concat_states_hierarchical(hx, time_step, hx_list, cell_state_pos)
        else:
            return self.concat_states_high_order_or_stack(hx, time_step, hx_list, cell_state_pos)


class DenseRNNCell(DenseRNNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", dense_depth=1,
                 dense_depth_base=2, add_transition_function=True,
                 layer=0, hierarchical=True, add_dense_block=True):
        super(DenseRNNCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=1, dense_depth=dense_depth,
            dense_depth_base=dense_depth_base,
            add_transition_function=add_transition_function,
            layer=layer, hierarchical=hierarchical,
            add_dense_block=add_dense_block
        )
        self.nonlinearity = nonlinearity
        self.layer = layer

    def rnn_step(self, cell_state_tensor, hidden_state_tensor,
                 input_x, outputs_tensor, time_step):
        """
        B: max_batch_size, T: time_steps, L: layers, H: hidden_size
        b: batch_size for current time step
        :param cell_state_tensor: [B, L, H]
        :param hidden_state_tensor: [B, L, H]
        :param input_x: [b, H]
        :param outputs_tensor: [B, T, L, H]
        :param time_step: t
        :param: batch_size: b
        :return:
        """
        batch_size = input_x.size(0)
        total_depth = time_step % self.dense_depth
        depth = 0
        valid_depth = 0
        hx = hidden_state_tensor[self.layer][:batch_size, :]
        s_recurrent = torch.mm(hx, self.weight_hh)

        while depth < total_depth:
            last_valid_depth = valid_depth
            if (self.layer == 0) or (self.layer > 0 and depth % (self.dense_depth // self.dense_depth_base) == (self.dense_depth // self.dense_depth_base) - 1):
                valid_depth += 1

            if valid_depth > last_valid_depth:
                if getattr(self, "dense_weight_hh_{}".format(last_valid_depth)) is None:
                    new_added_parameter = Parameter(
                        torch.Tensor(self.hidden_size, self.hidden_size).to(input_x.device)
                    )
                    torch.nn.init.uniform_(new_added_parameter, -self.stdv, self.stdv)
                    setattr(self, "dense_weight_hh_{}".format(last_valid_depth), new_added_parameter)

                dense_h = outputs_tensor[time_step - 1 - last_valid_depth][:batch_size, self.layer, :].squeeze(1)
                s_dense = torch.mm(dense_h, getattr(self, "dense_weight_hh_{}".format(last_valid_depth)))
                s_recurrent = s_recurrent + s_dense

            depth += 1

        s_below = torch.mm(input_x, self.weight_ih)
        if self.bias:
            s_recurrent = s_recurrent + self.bias_hh
            s_below = s_below + self.bias_ih

        f_s = s_recurrent + s_below
        h_new = torch.tanh(f_s)

        hidden_state_tensor[self.layer] = torch.cat(
            [h_new, hidden_state_tensor[self.layer][batch_size:]],
            dim=0
        )
        cell_state_tensor[self.layer] = torch.cat(
            [h_new, hidden_state_tensor[self.layer][batch_size:]],
            dim=0
        )
        return s_recurrent


class DenseGRUCell(DenseRNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", dense_depth=1,
                 dense_depth_base=2, add_transition_function=True,
                 layer=0, hierarchical=True, add_dense_block=True):
        super(DenseGRUCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=3, dense_depth=dense_depth,
            dense_depth_base=dense_depth_base,
            add_transition_function=add_transition_function,
            layer=layer, hierarchical=hierarchical,
            add_dense_block=add_dense_block
        )
        self.nonlinearity = nonlinearity
        self.layer = layer

    def rnn_step(self, cell_state_tensor, hidden_state_tensor,
                 input_x, outputs_tensor, time_step):
        """
        B: max_batch_size, T: time_steps, L: layers, H: hidden_size
        b: batch_size for current time step
        :param cell_state_tensor: [B, L, H]
        :param hidden_state_tensor: [B, L, H]
        :param input_x: [b, H]
        :param outputs_tensor: [B, T, L, H]
        :param time_step: t
        :param: batch_size: b
        :return:
        """
        batch_size = input_x.size(0)
        total_depth = time_step % self.dense_depth
        depth = 0
        valid_depth = 0
        hx = hidden_state_tensor[self.layer][:batch_size, :]
        s_recurrent = torch.mm(hx, self.weight_hh)

        while depth < total_depth:
            last_valid_depth = valid_depth
            if (self.layer == 0) or (self.layer > 0 and depth % (self.dense_depth // self.dense_depth_base) == (self.dense_depth // self.dense_depth_base) - 1):
                valid_depth += 1

            if valid_depth > last_valid_depth:
                if getattr(self, "dense_weight_hh_{}".format(last_valid_depth)) is None:
                    new_added_parameter = Parameter(
                        torch.Tensor(self.hidden_size, 3 * self.hidden_size).to(input_x.device)
                    )
                    torch.nn.init.uniform_(new_added_parameter, -self.stdv, self.stdv)
                    setattr(self, "dense_weight_hh_{}".format(last_valid_depth), new_added_parameter)

                dense_h = outputs_tensor[time_step - 1 - last_valid_depth][:batch_size, self.layer, :].squeeze(1)
                s_dense = torch.mm(dense_h, getattr(self, "dense_weight_hh_{}".format(last_valid_depth)))
                s_recurrent = s_recurrent + s_dense

            depth += 1

        s_below = torch.mm(input_x, self.weight_ih)
        if self.bias:
            s_recurrent = s_recurrent + self.bias_hh
            s_below = s_below + self.bias_ih

        i_r, i_i, i_n = s_below.chunk(3, 1)
        h_r, h_i, h_n = s_recurrent.chunk(3, 1)

        r_gate = torch.sigmoid(i_r + h_r)
        i_gate = torch.sigmoid(i_i + h_i)
        n_gate = torch.tanh(i_n + r_gate * h_n)
        hy = n_gate + i_gate * (hx - n_gate)

        hidden_state_tensor[self.layer] = torch.cat(
            [hy, hidden_state_tensor[self.layer][batch_size:]],
            dim=0
        )
        cell_state_tensor[self.layer] = torch.cat(
            [hy, hidden_state_tensor[self.layer][batch_size:]],
            dim=0
        )
        return s_recurrent


class DenseLSTMCell(DenseRNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", dense_depth=1,
                 dense_depth_base=2, add_transition_function=True,
                 layer=0, hierarchical=True, add_dense_block=True):
        super(DenseLSTMCell, self).__init__(
            input_size, hidden_size, bias, num_chunks=4, dense_depth=dense_depth,
            dense_depth_base=dense_depth_base,
            add_transition_function=add_transition_function,
            layer=layer, hierarchical=hierarchical,
            add_dense_block=add_dense_block
        )
        self.nonlinearity = nonlinearity
        self.layer = layer

    def rnn_step(self, cell_state_tensor, hidden_state_tensor,
                 input_x, outputs_tensor, time_step):
        """
        B: max_batch_size, T: time_steps, L: layers, H: hidden_size
        b: batch_size for current time step
        :param cell_state_tensor: [B, L, H]
        :param hidden_state_tensor: [B, L, H]
        :param input_x: [b, H]
        :param outputs_tensor: [B, T, L, H]
        :param time_step: t
        :param: batch_size: b
        :return:
        """
        batch_size = input_x.size(0)
        total_depth = time_step % self.dense_depth
        depth = 0
        valid_depth = 0
        hx = hidden_state_tensor[self.layer][:batch_size, :]
        cx = cell_state_tensor[self.layer][:batch_size, :]
        s_recurrent = torch.mm(hx, self.weight_hh)

        while depth < total_depth:
            last_valid_depth = valid_depth
            if (self.layer == 0) or (self.layer > 0 and depth % (self.dense_depth // self.dense_depth_base) == (self.dense_depth // self.dense_depth_base) - 1):
                valid_depth += 1

            if valid_depth > last_valid_depth:
                if getattr(self, "dense_weight_hh_{}".format(last_valid_depth)) is None:
                    new_added_parameter = Parameter(
                        torch.Tensor(self.hidden_size, 4 * self.hidden_size).to(input_x.device)
                    )
                    torch.nn.init.uniform_(new_added_parameter, -self.stdv, self.stdv)
                    setattr(self, "dense_weight_hh_{}".format(last_valid_depth), new_added_parameter)

                dense_h = outputs_tensor[time_step - 1 - last_valid_depth][:batch_size, self.layer, :].squeeze(1)
                s_dense = torch.mm(dense_h, getattr(self, "dense_weight_hh_{}".format(last_valid_depth)))
                s_recurrent = s_recurrent + s_dense

            depth += 1

        s_below = torch.mm(input_x, self.weight_ih)
        if self.bias:
            s_recurrent = s_recurrent + self.bias_hh
            s_below = s_below + self.bias_ih

        f_s = s_recurrent + s_below
        f = torch.sigmoid(f_s[:, 0:self.hidden_size])
        i = torch.sigmoid(f_s[:, self.hidden_size:self.hidden_size * 2])
        o = torch.sigmoid(f_s[:, self.hidden_size * 2:self.hidden_size * 3])
        g = torch.tanh(f_s[:, self.hidden_size * 3:self.hidden_size * 4])

        c_new = f * cx + i * g
        h_new = o * torch.tanh(c_new)

        hidden_state_tensor[self.layer] = torch.cat(
            [h_new, hidden_state_tensor[self.layer][batch_size:]],
            dim=0
        )
        cell_state_tensor[self.layer] = torch.cat(
            [c_new, hidden_state_tensor[self.layer][batch_size:]],
            dim=0
        )
        return s_recurrent


class DenseRNNBase(Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']

    def __init__(self, mode, input_size, hidden_size, num_layers,
                 batch_first=False, bias=True,
                 dropout=0.0,
                 start_dense_depth=1, dense_depth_base=2,
                 add_transition_function=True,
                 hierarchical=True, add_dense_block=True):
        super(DenseRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.dropout = float(dropout)
        self.start_dense_depth = start_dense_depth
        self.dense_depth_base = dense_depth_base
        self.add_transition_function = add_transition_function
        self.hierarchical = hierarchical
        self.add_dense_block = add_dense_block

        self.drop = torch.nn.Dropout(p=self.dropout)

        if mode == 'dense_lstm':
            self.rnn_cell = DenseLSTMCell
        elif mode == "dense_gru":
            self.rnn_cell = DenseGRUCell
        elif mode == "dense_rnn":
            self.rnn_cell = DenseRNNCell
        else:
            raise ValueError("Unrecognized DenseRNN mode: " + mode)

        self.cells = torch.nn.ModuleList(self.create_multicell())

    def create_multicell(self):
        cells = []
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size

            cells.append(self.rnn_cell(
                input_size=layer_input_size,
                hidden_size=self.hidden_size,
                bias=self.bias,
                dense_depth=2**(self.start_dense_depth + layer),
                dense_depth_base=self.dense_depth_base,
                add_transition_function=self.add_transition_function,
                layer=layer,
                hierarchical=self.hierarchical,
                add_dense_block=self.add_dense_block
            ))
        return cells

    def init_hidden_states(self, batch_size, inputs):
        hidden = []
        for layer in range(self.num_layers):
            h_l = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            c_l = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            hidden.append([h_l, c_l])
        return hidden

    def forward(self, inputs, hidden=None, batch_sizes=None):
        batch_size, _ = self._calc_batch_size(inputs.size())
        if batch_sizes is not None:
            batch_size = batch_sizes[0]
        if hidden is None:
            hidden = self.init_hidden_states(batch_size, inputs)
        if batch_sizes is None:
            func = self._fixed_forward
        else:
            func = self._var_forward_tf

        return func(inputs, hidden, batch_sizes)

    def _var_forward_tf(self, inputs, hidden, batch_sizes):
        """Follow rnn practice in tensorflow.
        outputs_tensor:
            a placeholder for the outputs_tensor with shape [B, T, H],
            where B stands for batch_size, T for time steps and H for hidden size.
        hidden_state_tensor:
            a placeholder for the final hidden state with shape [B, H].
        """
        max_batch_size = batch_sizes[0]
        time_steps = len(batch_sizes)
        outputs_tensor = [torch.zeros(
            (max_batch_size, len(self.cells), self.hidden_size),
            dtype=inputs.dtype, device=inputs.device
        ) for _ in range(time_steps)]
        hidden_state_tensor = [torch.zeros(
            (max_batch_size, self.hidden_size),
            dtype=inputs.dtype, device=inputs.device
        ) for _ in range(len(self.cells))]
        cell_state_tensor = [torch.zeros(
            (max_batch_size, self.hidden_size),
            dtype=inputs.dtype, device=inputs.device
        ) for _ in range(len(self.cells))]
        # for layer_i in range(len(self.cells)):
        #     hidden_state_tensor[layer_i] = hidden[layer_i][0]
        #     cell_state_tensor[layer_i] = hidden[layer_i][1]

        input_offset = 0
        # last_batch_size = batch_sizes[0]  # the biggest time step
        for t, batch_size in enumerate(batch_sizes):
            step_input = inputs[input_offset:input_offset + batch_size]
            input_offset += batch_size

            # dec = last_batch_size - batch_size

            for layer in range(len(self.cells)):
                if layer == 0:
                    h_below = step_input
                else:
                    # h_below = hidden_state_tensor[layer - 1].index_select(
                    #     0, torch.tensor(list(range(batch_size))).to(inputs.device)
                    # )
                    h_below = hidden_state_tensor[layer - 1][:batch_size, :]

                if self.dropout != 0 and self.training and layer < self.num_layers - 1:
                    h_below = self.drop(h_below)  # add dropout to each layer's input except the top layer

                if self.hierarchical:
                    if layer == 0 or t % self.cells[layer].dense_depth == 0 \
                            or t % (self.cells[layer].dense_depth // self.cells[layer].dense_depth_base) == \
                            (self.cells[layer].dense_depth // self.cells[layer].dense_depth_base - 1):
                        hy = self.cells[layer].rnn_step(
                            cell_state_tensor=cell_state_tensor,
                            hidden_state_tensor=hidden_state_tensor,
                            input_x=h_below,
                            outputs_tensor=outputs_tensor,
                            time_step=t
                        )
                    else:
                        # to change
                        # we have to copy, remember this, not doing anything is not right
                        pass
                else:
                    hy = self.cells[layer].rnn_step(
                        cell_state_tensor=cell_state_tensor,
                        hidden_state_tensor=hidden_state_tensor,
                        input_x=h_below,
                        outputs_tensor=outputs_tensor,
                        time_step=t
                    )
            # outputs_tensor[t].index_copy_(
            #     0, torch.tensor(list(range(max_batch_size))).to(inputs.device),
            #     torch.stack(hidden_state_tensor, dim=1)
            # )
            outputs_tensor[t] = torch.stack(hidden_state_tensor, dim=1)  # .clone()

        hidden_state_tensor = torch.stack(hidden_state_tensor)
        return hidden_state_tensor, outputs_tensor, hy

    @staticmethod
    def _calc_batch_size(input_size):
        time_steps = input_size[1]
        batch_size = input_size[0]
        return batch_size, time_steps

    @staticmethod
    def concat_along_steps(hiddens_list, hiddens, only_h=False):
        """Steps:
        1. 6:10.reverse() -> 10:6
        2. 10:6 + 3:5.reverse() -> 10:3
        3: 10:3 + 1:2.reverse() -> 10:1 (finally)
        """
        if len(hiddens_list) == 0:
            new_hidden_list = []
            for hiddens_layer in hiddens:
                hiddens_layer.reverse()
                new_hidden_list.append(hiddens_layer)
            return new_hidden_list
        elif len(hiddens) == 0:
            return hiddens_list
        new_hiddens_list = []
        for hiddens_list_layer, hiddens_layer in zip(hiddens_list, hiddens):
            hiddens_layer.reverse()
            if only_h:
                new_hiddens_list.append(hiddens_list_layer + hiddens_layer)
            else:
                pass
        return new_hiddens_list

    def _fixed_forward(self, inputs, hidden, batch_sizes=None):
        batch_size, time_steps = self._calc_batch_size(inputs.size())

        hidden_list = []
        hidden_new = hidden
        hidden_new_list = []

        for t in range(time_steps):
            hidden_new = []
            current_hidden = []
            for layer in range(len(self.cells)):
                if layer == 0:
                    h_below = inputs[:, t, :]
                else:
                    h_below = hidden[layer - 1][0]

                if self.dropout != 0 and self.training and layer < self.num_layers - 1:
                    h_below = self.drop(h_below)  # add dropout to each layer's input except the top layer

                # time_call_begin = time.time()
                h_tl, c_tl = self.cells[layer](
                    cx=hidden[layer][1],
                    hx=hidden[layer][0],
                    input_x=h_below,
                    hx_list=hidden_new_list,
                    time_step=t
                )
                hidden_new.append([h_tl, c_tl])
                # h_tl: batch_size * hidden_size
                # current_hidden: L [batch_size * hidden_size]
                current_hidden.append(h_tl)
            hidden = hidden_new
            hidden_list.append(current_hidden)
            hidden_new_list.append(hidden_new)

        return hidden_list, hidden_new, hidden_new_list
