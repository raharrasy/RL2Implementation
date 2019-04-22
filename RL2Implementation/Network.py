import torch
import torch.nn as nn
import torch.nn.functional as F


class RL2LSTM(nn.Module):
    def __init__(self, input_dim, lstm_output_dim, hidden_dim, action_dim, prep=False, hidden_prep_dim = None, prep_out_dim=None):
        super(RL2LSTM, self).__init__()
        self.input_dim = input_dim
        self.lstm_output_dim = lstm_output_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.prep_flag = prep
        self.prep = lambda inp : inp
        self.prep_out_dim = prep_out_dim
        self.hidden_prep_dim = hidden_prep_dim
        if not prep:
            self.prep_out_dim = input_dim
        else:
            self.prep_linears = nn.ModuleList([])
            prep_inp = self.input_dim
            for layer_dim in self.hidden_prep_dim:
                self.prep_linears.append(nn.Linear(prep_inp, layer_dim))
                prep_inp = layer_dim
            self.final_prep_layer = nn.Linear(layer_dim[-1], self.prep_out_dim)

        self.LSTMNet = nn.LSTM(self.prep_out_dim, self.lstm_output_dim)
        input_dim = self.lstm_output_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 20)

        self.feed_forward_act = nn.Linear(20, action_dim)
        self.feed_forward_val = nn.Linear(20, 1)

    def forward(self, input, hidden_state):
        if not self.prep_flag:
            input = self.prep(input)
        else:
            for layers in self.prep_linears:
                input = F.relu(layers(input))
            input = F.relu(self.final_prep_layer(input))

        lstm_output, lstm_hidden = self.LSTMNet(input.view(1,1,-1), hidden_state)
        forward_res = F.relu(lstm_output[0])
        forward_res = F.relu(self.linear(forward_res))
        forward_res = F.relu(self.linear2(forward_res))
        act_layer_res = self.feed_forward_act(forward_res)
        val_layer_res = self.feed_forward_val(forward_res)
        return act_layer_res, val_layer_res, lstm_hidden