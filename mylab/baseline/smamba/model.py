import torch
import torch.nn as nn
from .layers import Encoder, EncoderLayer, DataEmbedding_inverted
import torch.nn.functional as F

from mamba_ssm import Mamba
class SMamba(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, input_dim, hidden_dim, num_layers = 2, dropout = 0.):
        super().__init__()
        self.seq_len = 64
        self.pred_len = 1
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(input_dim, hidden_dim, dropout = dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=hidden_dim,  # Model dimension d_model
                            d_state=32,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=hidden_dim,  # Model dimension d_model
                            d_state=32,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    hidden_dim,
                    128,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(hidden_dim)
        )
        self.projector = nn.Linear(hidden_dim, self.pred_len, bias=True)
    # a = self.get_parameter_number()
    #
    # def get_parameter_number(self):
    #     """
    #     Number of model parameters (without stable diffusion)
    #     """
    #     total_num = sum(p.numel() for p in self.parameters())
    #     trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     trainable_ratio = trainable_num / total_num
    #
    #     print('total_num:', total_num)
    #     print('trainable_num:', total_num)
    #     print('trainable_ratio:', trainable_ratio)

    def forecast(self, x_enc, x_mark_enc = None):

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out = self.norm(enc_out)
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer) 
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1) # filter the covariates

        return dec_out


    def forward(self, x_enc, x_mark_enc = None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, 0]