'''
TransNet Transformer 解码器：把 (N, code_dim) 码字重建为 (N, channel, nt, nc)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
seq_len = input_dim // d_model

模型架构：
.
├── Linear(code_dim, input_dim)
├── View(N, seq_len, d_model)
├── TransformerDecoder
│   ├── TransformerDecoderLayer(d_model, nhead=2, dim_feedforward, batch_first=True)
│   │   ├── MultiheadAttention(d_model, 2, batch_first=True)
│   │   ├── Add()
│   │   ├── LayerNorm(d_model)
│   │   ├── MultiheadAttention(d_model, 2, batch_first=True)
│   │   ├── Add()
│   │   ├── LayerNorm(d_model)
│   │   ├── Linear(d_model, dim_feedforward)
│   │   ├── ReLU()
│   │   ├── Linear(dim_feedforward, d_model)
│   │   ├── Add()
│   │   └── LayerNorm(d_model)
│   └── TransformerDecoderLayer(d_model, nhead=2, dim_feedforward, batch_first=True)
├── LayerNorm(d_model)
└── View(N, channel, nt, nc)

使用技术：
先用 Linear 全局扩展码字，再用 Transformer 解码层混合 token；
当前实现以同一个 token 张量作为 tgt 和 memory，保持 TransNet 自编码重建范式。

保存模型权重时的参数维度：
fc_decoder.weight:                                       (input_dim, code_dim)
fc_decoder.bias:                                                  (input_dim,)
每个 decoder.layers.*.self_attn.in_proj_weight:           (3 * d_model, d_model)
每个 decoder.layers.*.self_attn.in_proj_bias:                     (3 * d_model,)
每个 decoder.layers.*.self_attn.out_proj.weight:              (d_model, d_model)
每个 decoder.layers.*.self_attn.out_proj.bias:                        (d_model,)
每个 decoder.layers.*.multihead_attn.in_proj_weight:      (3 * d_model, d_model)
每个 decoder.layers.*.multihead_attn.in_proj_bias:                (3 * d_model,)
每个 decoder.layers.*.multihead_attn.out_proj.weight:         (d_model, d_model)
每个 decoder.layers.*.multihead_attn.out_proj.bias:                   (d_model,)
每个 decoder.layers.*.linear1.weight:                 (dim_feedforward, d_model)
每个 decoder.layers.*.linear1.bias:                           (dim_feedforward,)
每个 decoder.layers.*.linear2.weight:                 (d_model, dim_feedforward)
每个 decoder.layers.*.linear2.bias:                                   (d_model,)
每个 decoder.layers.*.norm1/norm2/norm3.weight/bias:                  (d_model,)
decoder.norm.weight:                                                (d_model,)
decoder.norm.bias:                                                  (d_model,)
'''


import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class TransNetDecoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.feature_shape = (input_dim // d_model, d_model)
        self.fc_decoder = nn.Linear(input_dim // reduction, input_dim)
        decoder_layer = TransformerDecoderLayer(d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2, norm=nn.LayerNorm(d_model))

    def forward(self, code):
        batch_size = code.size(0)
        memory = self.fc_decoder(code).view(batch_size, self.feature_shape[0],
                                            self.feature_shape[1])
        out = self.decoder(memory, memory)
        out = out.view(batch_size, self.channel, self.nt, self.nc)
        return out
