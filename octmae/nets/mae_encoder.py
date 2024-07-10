from typing import Sequence

import torch as th
from torch import nn
import lightning.pytorch as pl
import xformers.ops as xops

from octmae.nets.blocks import MAEBlock


class MAEEncoder(pl.LightningModule):

    def __init__(self, config) -> None:
        super(MAEEncoder, self).__init__()

        self.num_layers = config.num_enc_layers
        encoders = []
        for _ in range(self.num_layers):
            block = MAEBlock(config, attn_type='self')
            encoders.append(block)
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x: th.Tensor, x_index: th.Tensor, x_length: Sequence[int]):
        x, x_index = x.unsqueeze(0), x_index.unsqueeze(0)
        attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(x_length, x_length)
        for encoder in self.encoders:
            x = encoder(x, x_index, attn_bias)
        return x[0]
