'''
TransNet Transformer 编码器：把 CSI 展成 token 序列，经自注意力后压缩为 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
seq_len = input_dim // d_model

模型架构：
.
├── View(N, seq_len, d_model)
├── TransformerEncoder
│   ├── TransformerEncoderLayer(d_model, nhead=2, dim_feedforward, batch_first=True)
│   │   ├── MultiheadAttention(d_model, 2, batch_first=True)
│   │   ├── Dropout(0.0)
│   │   ├── Add()
│   │   ├── LayerNorm(d_model)
│   │   ├── Linear(d_model, dim_feedforward)
│   │   ├── ReLU()
│   │   ├── Dropout(0.0)
│   │   ├── Linear(dim_feedforward, d_model)
│   │   ├── Dropout(0.0)
│   │   ├── Add()
│   │   └── LayerNorm(d_model)
│   └── TransformerEncoderLayer(d_model, nhead=2, dim_feedforward, batch_first=True)
├── Flatten()
└── Linear(input_dim, code_dim)

使用技术：
多头自注意力在 token 序列内建模全局 CSI 关系；前馈网络执行逐 token 非线性变换；
batch_first=True 明确使用 (N, seq_len, d_model) 维度约定。

保存模型权重时的参数维度：
每个 encoder.layers.*.self_attn.in_proj_weight:      (3 * d_model, d_model)
每个 encoder.layers.*.self_attn.in_proj_bias:                (3 * d_model,)
每个 encoder.layers.*.self_attn.out_proj.weight:         (d_model, d_model)
每个 encoder.layers.*.self_attn.out_proj.bias:                   (d_model,)
每个 encoder.layers.*.linear1.weight:            (dim_feedforward, d_model)
每个 encoder.layers.*.linear1.bias:                      (dim_feedforward,)
每个 encoder.layers.*.linear2.weight:            (d_model, dim_feedforward)
每个 encoder.layers.*.linear2.bias:                              (d_model,)
每个 encoder.layers.*.norm1.weight/bias:                         (d_model,)
每个 encoder.layers.*.norm2.weight/bias:                         (d_model,)
fc.weight:                                            (code_dim, input_dim)
fc.bias:                                                        (code_dim,)
'''


import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransNetEncoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        code_dim = input_dim // reduction
        self.feature_shape = (input_dim // d_model, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, code_dim)

    def forward(self, x):
        batch_size = x.size(0)
        memory = self.encoder(x.view(batch_size, self.feature_shape[0],
                                     self.feature_shape[1]))
        return self.fc(memory.flatten(1))
