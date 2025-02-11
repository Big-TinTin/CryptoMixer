import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time


class FadeDelta(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gamma_h_l = nn.Linear(1, hidden_dim)
        self.hidden_size = hidden_dim
        
    def forward(self, h, delta_t):
        delta_h = torch.exp(-torch.max(torch.zeros(self.hidden_size).to(h), self.gamma_h_l(delta_t))) # batch * dim
        return h * delta_h  # batch * dim

class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, X_mean, device = 'cuda:0', output_last = False):
        """
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.
        
        Implemented based on the paper: 
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """
        
        super(GRUD, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.mask_size = input_size
        
        self.identity = torch.eye(input_size).to(device)
        self.zeros = Variable(torch.zeros(input_size).to(device))
        self.X_mean = Variable(torch.Tensor(X_mean).to(device))
        
        self.zl = nn.Linear(input_size + input_size + hidden_size, hidden_size)
        self.rl = nn.Linear(input_size + input_size + hidden_size, hidden_size)
        self.hl = nn.Linear(input_size + input_size + hidden_size, hidden_size)
        
        self.gamma_x_l = nn.Linear(1, input_size)
        self.gamma_h_l = nn.Linear(1, hidden_size)
        
        self.output_last = output_last
        self.init_device = False
        self.device = device
        
    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        # x batch * dim
        # x_last_obsv batch * dim
        # mask batch * dim
        # delta batch * 1
        
        delta = delta.unsqueeze(-1) # batch * 1 
        
        delta_x = torch.exp(-torch.max(torch.zeros(self.input_size).to(delta), self.gamma_x_l(delta))) # batch * dim
        delta_h = torch.exp(-torch.max(torch.zeros(self.hidden_size).to(delta), self.gamma_h_l(delta))) # batch * dim
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)  # batch * dim
        h = delta_h * h   # batch * dim
        
        combined = torch.cat((x, h, mask), 1)   # batch * (dim*3)
        z = F.sigmoid(self.zl(combined))    # batch * dim
        r = F.sigmoid(self.rl(combined))    # batch * dim
        combined_r = torch.cat((x, r * h, mask), 1)   # batch * (dim*3)
        h_tilde = F.tanh(self.hl(combined_r))   # batch * dim
        h = (1 - z) * h + z * h_tilde   # batch * dim
        
        return h
    
    def forward(self, x, mask, delta_t):
        batch_size = x.size(0)
        input_dim = x.size(2)
        
        Hidden_State = self.initHidden(batch_size)
        now_x = x[:, 1:]
        last_x = x[:, :-1]
        mask = mask[:, 1:]
        delta_t = delta_t[:, 1:]
        
        outputs = None
        step_size = now_x.size(1)
        for i in range(step_size):
            Hidden_State = self.step(now_x[:, i], 
                                     last_x[:, i], 
                                     self.X_mean, 
                                     Hidden_State, 
                                     mask[:, i], 
                                     delta_t[:, i])
            # batch * dim
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1) # batch * 1 * dim
            else: 
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)  # batch * seq * dim
                
        if self.output_last:
            return outputs[:,-1]
        else:
            return outputs
    
    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size)).to(self.device)
        return Hidden_State

