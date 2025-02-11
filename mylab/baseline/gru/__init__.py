import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):

        hn = Variable(torch.zeros(x.size(0), self.hidden_dim).to(x))
       
        outs = []
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn) 
            outs.append(hn.unsqueeze(1))

        out = torch.cat(outs, dim = 1)
        out = self.fc(out) 
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 3, batch_first = True, res = True):
        super().__init__()

        self.gru = nn.ModuleList([
            GRUModel(input_dim, hidden_dim) if _ == 0 else GRUModel(hidden_dim, hidden_dim)
            for _ in range(num_layers) 
        ])

        self.res = res

    def forward(self, x):
        for grulayer in self.gru:
            if self.res:
                x = grulayer(x) + x
            else:
                x = grulayer(x)
        return x[:, -1]