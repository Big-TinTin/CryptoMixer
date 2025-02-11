import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import numpy as np


def attention(Q, K, V, attn_mask = None):                             
    scores = (Q @ K.transpose(-1, -2)) / np.sqrt(K.shape[-1])   
    if attn_mask is not None:
        scores.masked_fill_(attn_mask, -1e9)           
    attn_weight = F.softmax(scores, dim = -1)  # B * seq * seq
    attn_out = attn_weight @ V     # B * seq * dim             
    return attn_out, attn_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.wq = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.wk = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.wv = nn.Linear(d_model, d_model * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_model, d_model, bias=False)

    def forward(self, x, attn_mask = None):   
        batch_size = x.shape[0]
        Q = self.wq(x).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2) 
        K = self.wk(x).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2)
        V = self.wv(x).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2) 
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)             
        attn_out, attn_weight = attention(Q, K, V, attn_mask = attn_mask)          
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_model)
        output = self.fc(attn_out) 
        return output


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                             
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x   


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1,
                 activation = F.relu, layer_norm_eps = 1e-5):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FFN(d_model, dim_feedforward, dropout = dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, bias=False)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask = None):   
        x = self.norm1(x + self._sa_block(x, attn_mask = attn_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask = None):
        return self.dropout1(self.self_attn(x, attn_mask = attn_mask))

    def _ff_block(self, x):
        return self.dropout2(self.ffn(x))


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1,
                 activation = F.relu, layer_norm_eps = 1e-5, num_layers = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, 
                                     nhead, 
                                     dim_feedforward = dim_feedforward, 
                                     dropout = dropout,
                                     activation = activation, 
                                     layer_norm_eps = layer_norm_eps) 
             for _ in range(num_layers)]
        )

    def forward(self, x, attn_mask = None, res = False):
        for layer in self.layers:
            if res:
                x = layer(x, attn_mask = attn_mask) + x
            else:
                x = layer(x, attn_mask = attn_mask)
        return x