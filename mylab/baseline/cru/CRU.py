import torch
import numpy as np
import time as t
from datetime import datetime
import os
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from .utils import TimeDistributed, log_to_tensorboard, make_dir
from .encoder import Encoder
from .decoder import SplitDiagGaussianDecoder, BernoulliDecoder
from .model import CRULayer, var_activation, var_activation_inverse
# from lib.losses import rmse, mse, GaussianNegLogLik, bernoulli_nll
# from lib.data_utils import  align_output_and_target, adjust_obs_for_extrapolation

optim = torch.optim
nn = torch.nn


# taken from https://github.com/ALRhub/rkn_share/ and modified
class CRU(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def __init__(self, target_dim: int, lsd: int, args, use_cuda_if_available: bool = True, bernoulli_output: bool = False):
        """
        :param target_dim: output dimension
        :param lsd: latent state dimension
        :param args: parsed arguments
        :param use_cuda_if_available: if to use cuda or cpu
        :param use_bernoulli_output: if to use a convolutional decoder (for image data)
        """
        super().__init__()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        self._lsd = lsd
        if self._lsd % 2 == 0:
            self._lod = int(self._lsd / 2) 
        else:
            raise Exception('Latent state dimension must be even number.')
        self.args = args

        # parameters TODO: Make configurable
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self.bernoulli_output = bernoulli_output
        # main model

        self._cru_layer = CRULayer(
            latent_obs_dim=self._lod, args=args).to(self._device)

        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(self._lod, output_normalization=self._enc_out_normalization,
                      enc_var_activation=args.enc_var_activation).to(dtype=torch.float64)

        if bernoulli_output:
            BernoulliDecoder._build_hidden_layers = self._build_dec_hidden_layers
            self._dec = TimeDistributed(BernoulliDecoder(self._lod, out_dim=target_dim, args=args).to(
                self._device, dtype=torch.float64), num_outputs=1, low_mem=True)
            self._enc = TimeDistributed(
                enc, num_outputs=2, low_mem=True).to(self._device)

        else:
            SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
            SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
            self._dec = TimeDistributed(SplitDiagGaussianDecoder(self._lod, out_dim=target_dim, dec_var_activation=args.dec_var_activation).to(
                dtype=torch.float64), num_outputs=2).to(self._device)
            self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        # build (default) initial state
        self._initial_mean = torch.zeros(1, self._lsd).to(
            self._device, dtype=torch.float64)
        log_ic_init = var_activation_inverse(self._initial_state_variance)
        self._log_icu = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float64))
        self._log_icl = torch.nn.Parameter(
            log_ic_init * torch.ones(1, self._lod).to(self._device, dtype=torch.float64))
        self._ics = torch.zeros(1, self._lod).to(
            self._device, dtype=torch.float64)

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError
    
    # taken from https://github.com/ALRhub/rkn_share/ and modified
    def forward(self, obs_batch: torch.Tensor, time_points: torch.Tensor = None, obs_valid: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass on a batch
        :param obs_batch: batch of observation sequences
        :param time_points: timestamps of observations
        :param obs_valid: boolean if timestamp contains valid observation 
        """
        y, y_var = self._enc(obs_batch)
        post_mean, post_cov, prior_mean, prior_cov, kalman_gain = self._cru_layer(y, y_var, self._initial_mean,
                                                                                    [var_activation(self._log_icu), var_activation(
                                                                                        self._log_icl), self._ics],
                                                                                    obs_valid=obs_valid, time_points=time_points)
        
        out_mean, out_var = self._dec(
            prior_mean, torch.cat(prior_cov, dim=-1))

        return out_mean
