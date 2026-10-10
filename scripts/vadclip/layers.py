"""Graph-convolution layers for the LGT-Adapter (device-agnostic port).

Faithful to ``utils/layers.py`` from the released VadCLIP repo, with one deliberate
change: the original ``DistanceAdj.forward`` hard-codes ``.to('cuda')``; here it takes
the target device as an argument so the whole model runs on CPU (this project is a
CPU-only benchmark). Everything else -- the GCN support/adjacency matmul, the
identity-vs-Conv1d residual rule, and the fixed cityblock distance kernel -- matches.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """Simple GCN layer (arXiv:1609.02907) with a residual connection."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, residual: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # residual rule (matches the original): identity when width is unchanged, a
        # channel-mixing Conv1d when it changes (512 -> 256), zero if disabled.
        self._conv_residual = (in_features != out_features) and residual
        if not residual:
            self.residual = lambda x: 0
        elif in_features == out_features:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = input.matmul(self.weight)          # [B, T, out]
        output = adj.matmul(support)                 # [B, T, out] (padded rows stay 0)

        if self.bias is not None:
            output = output + self.bias
        if self._conv_residual:
            inp = input.permute(0, 2, 1)             # [B, in, T]
            res = self.residual(inp).permute(0, 2, 1)
            output = output + res
        else:
            output = output + self.residual(input)
        return output


class DistanceAdj(nn.Module):
    """Fixed temporal-distance adjacency kernel (device-agnostic).

    For frame indices ``i, j`` the entry is ``exp(-|i-j| / e)`` -- a smooth decay with
    temporal separation. The original hard-codes CUDA; here the device is passed in.
    """

    def __init__(self):
        super().__init__()
        self.sigma = Parameter(torch.empty(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size: int, max_seqlen: int, device: torch.device) -> torch.Tensor:
        # cityblock distance over frame indices == |i - j| (pure numpy; no scipy needed)
        idx = np.arange(max_seqlen)
        dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)   # [L, L]
        dist = torch.from_numpy(dist).to(device)
        dist = torch.exp(-dist / torch.exp(torch.tensor(1.0)))          # exp(-|i-j|/e)
        return dist.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
