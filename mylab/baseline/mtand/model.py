import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1, dropout = 0.0) :
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.dropout = nn.Dropout(dropout)
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time, bias=False), 
                                      nn.Linear(embed_time, embed_time, bias=False),
                                      nn.Linear(input_dim*num_heads, nhidden, bias=False)])
        
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        # mask [batch, 1, seq]
        dim = value.size(-1)    # [batch, 1, seq， dim]
        d_k = query.size(-1)   # [batch, head, seq, dim_t_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)   # [batch, p, head, seq, seq]
        scores = scores  # [batch, p, head, seq, seq]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # mask [batch, p, 1, 1, seq]
        p_attn = F.softmax(scores, dim = -1)  # [batch, p, head, seq, seq]
        p_attn = self.dropout(p_attn)  # [batch, p, head, seq, seq]
        
        return p_attn @ value, p_attn   # [batch, head, seq, dim]
    
    
    def forward(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        value = value.unsqueeze(1)   # [batch, 1, seq， dim]
        query, key = [l(x).view(batch, -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]   # [batch, head, seq, dim_t_k]

        x, _ = self.attention(query, key, value, mask)  # [batch, head, seq, dim]
        # [32, 109, 4, 165, 32]
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)   # [batch, p, seq, dim]
        
        return self.linears[-1](x)  # [batch, p, seq, dim]


class enc_mtan_classif_activity(nn.Module):
 
    def __init__(self, input_dim, nhidden=32, 
                 embed_time=16, num_heads=1, device='cuda', dropout = 0.0, num_layers = 1, batch_first = True):
        super().__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        
        self.att = multiTimeAttention(input_dim, nhidden, embed_time, num_heads, dropout = dropout)
        self.gru = nn.GRU(nhidden, nhidden, num_layers = num_layers, batch_first = batch_first)

        self.periodic = nn.Linear(1, embed_time-1)
        self.linear = nn.Linear(1, 1)
        self.ref_t = torch.linspace(0.,1.,64)

    def norm_t(self, t):
        t_min = torch.min(t, dim = 1)[0].unsqueeze(-1) # b, seq
        t_max = torch.max(t, dim = 1)[0].unsqueeze(-1) # b, seq
        return (t - t_min) / (t_max - t_min)
    
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)  
        out2 = torch.sin(self.periodic(tt))  
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)  
       
    def forward(self, x, t):

        t = self.norm_t(t)
        t = t[:, :-1]
        k = self.learn_time_embedding(t)
        q = self.learn_time_embedding(self.ref_t.to(t)).unsqueeze(0).repeat(t.shape[0], 1, 1)
        x = self.att(q, k, x) 
        
        encoded_x, _  = self.gru(x) 
        return encoded_x[:, -1]

