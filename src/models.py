import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, units=(64, 32), num_inputs=8, num_actions=4, seq_len=3):
        super(LinearModel, self).__init__()
        layers = [
            nn.Linear(num_inputs * seq_len, units[0]),
            nn.ReLU()
        ]
        if len(units) == 2:
            layers.append(nn.Linear(units[0], units[1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(units[-1], num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.model(x)


class MemoryModel(nn.Module):
    def __init__(self, rnn: str = "GRU", units=(64, 32), num_inputs=8, num_actions=4, seq_len=3):
        super(MemoryModel, self).__init__()
        self.seq_len = seq_len
        if rnn == 'GRU':
            rnn = nn.GRU(num_inputs, units[0], batch_first=True)
        else:
            rnn = nn.LSTM(num_inputs, units[0], batch_first=True)
        self.rnn = rnn
        self.linear1 = nn.Linear(units[0] * seq_len, units[1])
        self.linear2 = nn.Linear(units[1], num_actions)

    def forward(self, x):
        self.rnn.flatten_parameters()
        # for i in range(self.seq_len)
        x, hidden = self.rnn(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
