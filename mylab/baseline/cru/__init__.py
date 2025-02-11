import torch.nn as nn
from torch.nn.parameter import Parameter
from ..iteract_att import InteractATT
import torch
import numpy as np
import torch.nn.functional as F
from ..transformer import TransformerEncoder
from ..time2vec import PositionalEncoding, T2V
from .CRU import CRU
from .data import CollateFunc


class CRUClassifier(nn.Module):
    def __init__(self, 
                 input_xtime_index, 
                 input_x_index, 
                 input_xntime_index, 
                 input_x_n_index, 
                 input_x_action_index,
                 input_x_market_info_index, 
                 hidden_dim = 32, 
                 time_dim = 8,
                 output_dim = 2, 
                 heads = 4, 
                 final_hidden = 128, 
                 mean = None,
                 interact = False,
                 use_cru = False,
                 dropout = 0.0):
        super().__init__()
        
        self.input_x_action_index = input_x_action_index
        self.input_x_market_info_index = input_x_market_info_index

        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)

        # self.predict_lookup = nn.Embedding(2, hidden_dim)

        self.interact = interact
        self.use_cru = use_cru

        self.iteractatt = InteractATT(hidden_dim, hidden_dim, dropout = dropout)
        #self.mtand = Mtand(hidden_dim, nhidden=hidden_dim, embed_time=hidden_dim, num_heads=4, interact = interact, dropout = dropout)
        #self.grud3 = GRUD(hidden_dim, hidden_dim, np.zeros(hidden_dim), heads)
        #self.trf = TransformerEncoder(hidden_dim, 8, num_layers = 1, dropout = 0)
        self.cru = CRU(32, 64, args)

        #self.fadedelta = FadeDelta(hidden_dim)

        self.nn2 = nn.Linear(hidden_dim * 2, final_hidden)
        self.nn3 = nn.Linear(final_hidden, 64)
        self.nn4 = nn.Linear(64, output_dim)
    
        self.input_x_index = input_x_index
        self.input_xtime_index = input_xtime_index
        self.input_x_n_index = input_x_n_index
        self.input_xntime_index = input_xntime_index
        self.output_dim = output_dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        
    def forward(self, ref_t, t, x, mask, xn, sim_mask = None):

        batch, p, seq, _ = x.shape
        
        t_n = xn[:, self.input_xntime_index]  # (batch, 1)
        
        x_time = x[:, :, :, self.input_xtime_index]
        x = x[:, :, :, self.input_x_index]
        xn_time = xn[:, self.input_xntime_index]
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)

        x = self.init_x_nn(x)
        encoded_xn = self.init_xn_nn(xn)

        if self.interact:
            h = self.iteract_encoder(x, sim_mask = sim_mask)  # [batch, seq, dim] 
        else:
            h = x[:, -1]   #(batch, seq, dim)

        if self.use_cru:
            h = self.cru_encoder(h, ref_t, t, t_n, mask[:, -1])  # [batch, seq, dim] # [batch, dim]
        else:
            h = h
        
        h = self.time_series_encoder(h) # [batch, dim]
        
        out = self.final_nn(h, encoded_xn)
        #if self.output_dim > 1:
            #out = self.softmax(out)
        #else:
            #out = torch.sigmoid(out)

        return out

    def iteract_encoder(self, h, mask = None, sim_mask = None):
        
        #(batch, n_people, seq, dim)
        h = self.iteractatt(h, mask = mask, sim_mask = sim_mask)  # [batch, seq, dim] # [batch, dim]

        return h

    def cru_encoder(self, h, ref_t, t, t_n, mask):
        
        h = self.cru(h, ref_t, t, t_n, mask)  # [batch, seq, dim] # [batch, dim]

        return h 

    def final_nn(self, h, encoded_xn):
        final = torch.cat([h, encoded_xn], axis = 1) #(batch, dim * 2)
        # final = self.nn2(final)
        final = F.leaky_relu(self.nn2(final))
        #final = self.dropout(final)
        final = self.nn3(final)
        final = F.leaky_relu(final)
        #final = self.dropout(final)
        out = self.nn4(final)
        return out