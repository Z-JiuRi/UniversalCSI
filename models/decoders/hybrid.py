'''
混合 Transformer-CNN 解码器：全局 token 重建加局部 CNN 精修，输出 CSI 张量。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
seq_len = input_dim // d_model

模型架构：
.
├── semantic_projector
│   ├── LayerNorm(code_dim)
│   └── Linear(code_dim, code_dim)
├── Linear(code_dim, input_dim)
├── View(N, seq_len, d_model)
├── TransformerEncoder
│   ├── TransformerEncoderLayer(d_model, nhead=2, dim_feedforward, batch_first=True)
│   └── TransformerEncoderLayer(d_model, nhead=2, dim_feedforward, batch_first=True)
├── LayerNorm(d_model)
├── View(N, channel, nt, nc)
├── CNNRefinementHead
│   ├── Conv2d(channel, hidden, 3, padding=1, bias=False)
│   ├── LeakyReLU(0.3)
│   ├── ConvResidualBlock(hidden) * num_blocks
│   │   ├── Conv2d(hidden, hidden, 3, padding=1, bias=False)
│   │   ├── BatchNorm2d(hidden)
│   │   ├── LeakyReLU(0.3)
│   │   ├── Conv2d(hidden, hidden, 3, padding=1, bias=False)
│   │   ├── BatchNorm2d(hidden)
│   │   ├── Add()
│   │   └── LeakyReLU(0.3)
│   └── Conv2d(hidden, channel, 3, padding=1)
└── Add(coarse, residual_scale * refine(coarse))

使用技术：
码字语义投影后用 TransformerEncoder 全局混合 token；CNN 残差头补偿局部重建误差；
residual_scale 控制精修幅度，精修输出层可零初始化以稳定训练初期。

保存模型权重时的参数维度：
semantic_projector.norm.weight/bias:                                                   (code_dim,)
semantic_projector.linear.weight:                                             (code_dim, code_dim)
semantic_projector.linear.bias:                                                        (code_dim,)
token_projection.weight:                                                     (input_dim, code_dim)
token_projection.bias:                                                                (input_dim,)
每个 token_mixer.layers.*.self_attn.in_proj_weight:                           (3 * d_model, d_model)
每个 token_mixer.layers.*.self_attn.in_proj_bias:                                     (3 * d_model,)
每个 token_mixer.layers.*.self_attn.out_proj.weight:                              (d_model, d_model)
每个 token_mixer.layers.*.self_attn.out_proj.bias:                                        (d_model,)
每个 token_mixer.layers.*.linear1.weight:                                 (dim_feedforward, d_model)
每个 token_mixer.layers.*.linear1.bias:                                           (dim_feedforward,)
每个 token_mixer.layers.*.linear2.weight:                                 (d_model, dim_feedforward)
每个 token_mixer.layers.*.linear2.bias:                                                   (d_model,)
每个 token_mixer.layers.*.norm1/norm2.weight/bias:                                        (d_model,)
token_mixer.norm.weight:                                                                (d_model,)
token_mixer.norm.bias:                                                                  (d_model,)
refine.net.conv_in.weight:                                                 (hidden, channel, 3, 3)
每个 refine.net.res*.block.conv1.conv.weight:                                 (hidden, hidden, 3, 3)
每个 refine.net.res*.block.conv1.bn.weight/bias/running_mean/running_var:                  (hidden,)
每个 refine.net.res*.block.conv2.conv.weight:                                 (hidden, hidden, 3, 3)
每个 refine.net.res*.block.conv2.bn.weight/bias/running_mean/running_var:                  (hidden,)
refine.net.conv_out.weight:                                                (channel, hidden, 3, 3)
refine.net.conv_out.bias:                                                               (channel,)
residual_scale:                                                                               (1,)
'''


import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from .cnn_residual import CNNRefinementHead


class HybridDecoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None, hidden=16, num_blocks=2):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.feature_shape = (input_dim // d_model, d_model)
        code_dim = input_dim // reduction

        # ---- submodules ----
        self.semantic_projector = nn.Sequential(OrderedDict([
            ("norm", nn.LayerNorm(code_dim)),
            ("linear", nn.Linear(code_dim, code_dim)),
        ]))
        self.token_projection = nn.Linear(code_dim, input_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.token_mixer = TransformerEncoder(encoder_layer, num_layers=2,
                                              norm=nn.LayerNorm(d_model))
        self.refine = CNNRefinementHead(channel, hidden, num_blocks,
                                        zero_init_output=True)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def reset_refinement_output(self):
        self.refine.reset_output()

    def forward(self, code):
        batch_size = code.size(0)
        x = self.semantic_projector(code)
        tokens = self.token_projection(x)
        tokens = tokens.view(batch_size, self.feature_shape[0],
                             self.feature_shape[1])
        tokens = self.token_mixer(tokens)
        coarse = tokens.view(batch_size, self.channel, self.nt, self.nc)
        return coarse + self.residual_scale * self.refine(coarse)
