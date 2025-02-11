import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import numpy as np


class T2V(nn.Module):
    '''
    transform time_step to vectors 
    '''
    def __init__(self, hidden_size):
        super(T2V,self).__init__()
        self.f1 = nn.Linear(1, hidden_size-1)
        self.f1.weight = torch.nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, hidden_size)).float().view(hidden_size, 1))
        self.f1.bias = torch.nn.Parameter(torch.zeros(hidden_size).float().view(hidden_size))
        self.f2 = nn.Linear(1, 1)
        self.f2.weight = torch.nn.Parameter(torch.ones([1,1]))
        self.f2.bias = torch.nn.Parameter(torch.zeros(1).float().view(1))

    def forward(self, t):
        tt = t.to(torch.float32)
        out1 = torch.cos(self.f1(tt))
        out2 = self.f2(tt)
        return torch.cat([out2, out1], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        assert len(t.shape) == 2
        d_model = self.dim
        batch = t.shape[0]
        N = t.shape[1]
        result = torch.ones(N, self.dim)  # seq, dim
        pos = torch.arange(N).unsqueeze(1)   # seq, 1
        i = torch.arange(0, d_model, 2)  # dim / 2
        div = 10000 ** (i / d_model)  # dim / 2
        term = pos / div  # seq, dim / 2
        result[:, 0::2] = torch.sin(term)
        result[:, 1::2] = torch.cos(term)

        result = result.unsqueeze(0).repeat(batch, 1, 1).to(t)
        return result

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.nn = nn.Linear(1, dim, bias = False)

        print('use time encoding')

    def forward(self, t):
        assert t.shape[-1] != 1 # batch, seq
        t = t.unsqueeze(-1)
        result = torch.ones(*[*t.shape[:-1], self.dim]).to(t)  # seq, dim
        tw = self.nn(t)  # batch, seq, dim
        result[..., 0::2] = torch.sin(tw[..., 0::2])
        result[..., 1::2] = torch.cos(tw[..., 1::2])
        return result
