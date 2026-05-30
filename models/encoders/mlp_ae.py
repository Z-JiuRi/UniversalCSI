'''
MLP 自编码器编码端：把 CSI 展平后用全连接网络压缩为 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
hidden = hidden or min(4096, input_dim)

模型架构：
.
├── Flatten()
├── Linear(input_dim, hidden)
├── LayerNorm(hidden)
├── GELU()
└── Linear(hidden, code_dim)

使用技术：
不引入卷积先验，直接把角延迟域矩阵当作全局向量建模；LayerNorm 稳定隐藏层；
GELU 提供平滑非线性变换。

保存模型权重时的参数维度：
net.fc1.weight:  (hidden, input_dim)
net.fc1.bias:              (hidden,)
net.norm.weight:           (hidden,)
net.norm.bias:             (hidden,)
net.fc2.weight:   (code_dim, hidden)
net.fc2.bias:            (code_dim,)
'''


import torch.nn as nn
from collections import OrderedDict


class MLPAEEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, hidden=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        hidden = hidden or min(4096, input_dim)
        self.net = nn.Sequential(OrderedDict([
            ("flatten", nn.Flatten()),
            ("fc1", nn.Linear(input_dim, hidden)),
            ("norm", nn.LayerNorm(hidden)),
            ("gelu", nn.GELU()),
            ("fc2", nn.Linear(hidden, input_dim // reduction)),
        ]))

    def forward(self, x):
        return self.net(x)
