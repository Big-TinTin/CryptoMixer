import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from .layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .layers.SelfAttention_Family import FullAttention, ProbAttention
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, input_x_dim, input_xn_dim):
        super().__init__()
        self.enc_in = input_x_dim
        self.dec_in = 56
        self.ab = 0
        self.modes = 32
        self.mode_select = 'random'
        # version = 'Fourier'
        self.version = 'Wavelets'
        self.moving_avg = [12, 24]
        self.L = 1
        self.base = 'legendre'
        self.cross_activation = 'tanh'
        self.seq_len = 64
        self.label_len = 1
        self.pred_len = 1
        self.output_attention = True
        self.d_model = 64
        self.embed = 'timeF'
        self.dropout = 0.05
        self.freq = 'h'
        self.factor = 1
        self.n_heads = 8
        self.d_ff = 128
        self.e_layers = 2
        self.d_layers = 1
        self.c_out = 2
        self.activation = 'gelu'
        self.wavelet = 0

        # Decomp
        kernel_size = [12, 24]
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, 'timeF', 'h',
                                                  0.05)
        self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, 'timeF', 'h',
                                                  0.05)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=self.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=self.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=self.d_model,
                                                  out_channels=self.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=self.d_model,
                                                  base='legendre',
                                                  activation=self.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=self.d_model,
                                            out_channels=self.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=self.d_model,
                                                      out_channels=self.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        enc_modes = int(min(self.modes, self.seq_len//2))
        dec_modes = int(min(self.modes, (self.seq_len//2+self.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model, self.n_heads),

                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_dec, x_mark_dec = None, x_mark_enc = None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -1, :], attns
        else:
            return dec_out[:, -1, :]  # [B, L, D]


