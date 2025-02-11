import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TemporalPositionalEncoding(nn.Module):

    def __init__(self, 
                 dim
        ):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        assert len(t.shape) == 2
        
        batch_size, seq_len = t.shape
        d_model = self.dim
        
        result = torch.zeros(batch_size, seq_len, d_model).to(t)  # (batch_size, seq_len, dim)
        i = torch.arange(0, d_model, 2, dtype=torch.float32).to(t)

        div = 10000 ** (i / d_model)

        term = t.unsqueeze(-1) / div  # Broadcasting: (batch, seq_len, 1) / (1, dim/2)
        result[:, :, 0::2] = torch.sin(term)  # sin for even indices
        result[:, :, 1::2] = torch.cos(term)  # cos for odd indices
        
        return result
