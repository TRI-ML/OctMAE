from typing import Sequence, Optional

import torch as th
from torch import nn
import lightning.pytorch as pl
import xformers.ops as xops

from octmae.nets.blocks import MAEBlock


class MAEDecoder(pl.LightningModule):

    def __init__(self, config) -> None:
        super(MAEDecoder, self).__init__()

        self.num_layers = config.num_dec_layers
        decoders = []
        for _ in range(self.num_layers):
            block = MAEBlock(config, attn_type='cross')
            decoders.append(block)
        self.decoders = nn.Sequential(*decoders)

    def forward(
            self, x: th.Tensor, x_index: th.Tensor, x_length: Sequence[int],
            y: Optional[th.Tensor] = None, y_index: Optional[th.Tensor] = None, y_length: Optional[Sequence[int]] = None):

        x, x_index = x.unsqueeze(0), x_index.unsqueeze(0)
        y, y_index = y.unsqueeze(0), y_index.unsqueeze(0)
        attn_bias = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(x_length, y_length)

        for decoder in self.decoders:
            x = decoder(x, x_index, attn_bias, y, y_index)
        return x[0]
