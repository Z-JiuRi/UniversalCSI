from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

__all__ = ["transnet"]

Tensor = torch.Tensor


class Transformer(nn.Module):

    def __init__(self, d_model: int = 64, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: Optional[int] = None, dropout: float = 0.1,
                 activation=F.relu, custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 reduction=64, channel: int = 2, nt: int = 32,
                 nc: int = 32) -> None:
        super(Transformer, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        layer_args = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": "relu" if activation is F.relu else activation,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
        }

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = nn.TransformerEncoderLayer(**layer_args)
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = nn.TransformerDecoderLayer(**layer_args)
            decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.input_dim = channel * nt * nc
        self.d_model = d_model

        assert self.input_dim % self.d_model == 0, (
            f"d_model needs to divide the flattened CSI size ({self.input_dim})")
        assert self.input_dim % reduction == 0, (
            f"reduction needs to divide the flattened CSI size ({self.input_dim})")
        self.feature_shape = (self.input_dim // self.d_model, self.d_model)

        self.nhead = nhead
        self.batch_first = batch_first
        self.fc_encoder = nn.Linear(self.input_dim, self.input_dim // reduction)
        self.fc_decoder = nn.Linear(self.input_dim // reduction, self.input_dim)
        self._reset_parameters()

    def _to_transformer_input(self, src: Tensor) -> Tensor:
        return src.view(src.size(0), self.feature_shape[0], self.feature_shape[1])

    def forward(self, src: Tensor, tgt: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        batch_size = src.size(0)
        src = self._to_transformer_input(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory_encoder = self.fc_encoder(memory.reshape(batch_size, -1))
        memory_decoder = self.fc_decoder(memory_encoder).view(batch_size, self.feature_shape[0], self.feature_shape[1])
        output = self.decoder(
            memory_decoder,
            memory_decoder,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = output.reshape(batch_size, self.channel, self.nt, self.nc)
        return output

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        batch_size = src.size(0)
        src = self._to_transformer_input(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory_encoder = self.fc_encoder(memory.reshape(batch_size, -1))
        return memory_encoder

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def transnet(reduction=64, d_model=64, channel=2, nt=32, nc=32,
             dim_feedforward=None):
    r""" Create a proposed TransNet.

        :param reduction: the reciprocal of compression ratio
        :return: an instance of TransNet
    """
    model = Transformer(d_model=d_model, num_encoder_layers=2,
                        num_decoder_layers=2, nhead=2, reduction=reduction,
                        dropout=0., channel=channel, nt=nt, nc=nc,
                        dim_feedforward=dim_feedforward)
    return model
