import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class GRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h))
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        g = torch.tanh(self.lin_xn(x) + self.lin_hn(r * h))
        return z * h + (1 - z) * g


class GRUODECell(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        g = torch.tanh(x + self.lin_hn(r * h))
        dh = (1 - z) * (g - h)
        return dh


class NODE(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, in_features, hidden_features, dropout_p=0.1, solver='rk4', step_size=0.25):
        super().__init__()
        self.hidden_dim = hidden_features
        if self.hidden_dim == 0:
            return
        self.gru = GRUCell(in_features, hidden_features)
        self.output_gru = GRUCell(in_features, hidden_features)
        self.odefun = GRUODECell(hidden_features)
        self.dropout = nn.Dropout(dropout_p)
        self.solver = solver
        if self.solver == 'euler' or self.solver == 'rk4':
            self.step_size = step_size

    def integrate(self, delta_t, X, Xn, mask=None):
        # X [batch, seq, dim]
        # Xn [batch, dim]
        # delta_t [batch, seq]

        batch, seq_len, feat_dim = X.shape
        delta_t = delta_t.unsqueeze(-1)  # batch, seq_len, 1
        h = torch.zeros(batch, self.hidden_dim).type_as(X)
        for i in range(seq_len):
            now_x = X[:, i, :]  # [batch, dim]
            if mask is not None:
                now_mask = mask[:, i].unsqueeze(-1)  # [batch, 1]
                now_x = now_x * now_mask
            h = self.gru(now_x, h)  # [batch, dim]
            now_delta_t = delta_t[:, i]   # [batch, 1]
            now_delta_t = torch.log10(torch.abs(now_delta_t) + 1.0) + 0.01   # [batch, 1]
            h = (torch.zeros(batch, 1).to(now_delta_t), now_delta_t, h)
            if self.solver == 'euler' or self.solver == 'rk4':
                solution = odeint(self,
                                  h,
                                  torch.tensor([self.start_time, self.end_time]).type_as(X),
                                  method=self.solver,
                                  options=dict(step_size=self.step_size))
            elif self.solver == 'dopri5':
                solution = odeint(self,
                                  h,
                                  torch.tensor([self.start_time, self.end_time]).type_as(X),
                                  method=self.solver)
            else:
                raise NotImplementedError('{} solver is not implemented.'.format(self.solver))
            _, _, h = tuple(s[-1] for s in solution)

        encoded_features = self.output_gru(Xn, h)
        encoded_features = self.dropout(encoded_features)
        return encoded_features

    def forward(self, s, state):
        t0, t1, x = state
        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0
        dx = self.odefun(t, x)
        dx = dx * ratio
        return torch.zeros_like(t0), torch.zeros_like(t1), dx
