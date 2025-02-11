import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        hx, cx = hidden
        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, F.tanh(cy))
        
        return (hy, cy)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim 
        self.lstm = LSTMCell(input_dim, hidden_dim)  
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):

        hn = Variable(torch.zeros(x.size(0), self.hidden_dim).to(x))
        cn = Variable(torch.zeros(x.size(0), self.hidden_dim).to(x))            
       
        outs = []
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:,seq,:], (hn,cn)) 
            outs.append(hn.unsqueeze(1))
            
        out = torch.cat(outs, dim = 1)
        out = self.fc(out) 
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 3, batch_first = True, res = True):
        super().__init__()
        
        self.lstm = nn.ModuleList([
            LSTMModel(input_dim, hidden_dim) if _ == 0 else LSTMModel(hidden_dim, hidden_dim)
            for _ in range(num_layers) 
        ])
        self.res = res

    def forward(self, x):
        for lstmlayer in self.lstm:
            if self.res:
                x = lstmlayer(x) + x
            else:
                x = lstmlayer(x)
        return x[:, -1]