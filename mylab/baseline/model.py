import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from ..transformer import TransformerEncoder as _TransformerEncoder_
import math
from torch.autograd import Variable
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, heads, num_layers = 3):
        super().__init__()
        
        self.trf = _TransformerEncoder_(hidden_dim, heads, num_layers = num_layers)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        encoded_x = self.trf(x)   
        encoded_x = self.linear(encoded_x.permute(0, 2, 1))
        return encoded_x[:, :, -1]