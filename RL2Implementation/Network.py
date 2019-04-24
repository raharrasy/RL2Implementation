import torch
import torch.nn as nn
import torch.nn.functional as F

class RL2LSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super(RL2LSTM, self).__init__()
        self.LSTM = nn.LSTM(input_dim, hidden_dim, batch_first = True)
        self.fullyconnected1 = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, net_input, hidden):
        out, hidden = self.LSTM(net_input, hidden)
        out = F.relu(self.fullyconnected1(out))
        pol_out = F.softmax(self.policy(out), dim = -1)
        val_out = self.value(out)
        return pol_out, val_out, hidden

