'''
CsiNet 编码器：用于 CSI 压缩反馈，把 (N, channel, nt, nc) 压缩为 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction

模型架构：
.
├── Conv2d(channel, channel, 3, padding=1, bias=False)
├── BatchNorm2d(channel)
├── LeakyReLU(0.3)
├── Flatten()
└── Linear(input_dim, code_dim)

使用技术：
3x3 局部卷积提取角延迟域邻域特征；BatchNorm 稳定浅层特征分布；
Linear 瓶颈直接生成压缩反馈码字。

保存模型权重时的参数维度：
features.conv.weight:     (channel, channel, 3, 3)
features.bn.weight:                     (channel,)
features.bn.bias:                       (channel,)
features.bn.running_mean:               (channel,)
features.bn.running_var:                (channel,)
fc.weight:                   (code_dim, input_dim)
fc.bias:                               (code_dim,)
'''


import torch.nn as nn
from collections import OrderedDict


class CsiNetEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32):
        super().__init__()
        input_dim = channel * nt * nc
        code_dim = input_dim // reduction
        self.features = nn.Sequential(OrderedDict([
            ("conv3x3", nn.Conv2d(channel, channel, 3, padding=1, bias=False)),
            ("bn", nn.BatchNorm2d(channel)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.fc = nn.Linear(input_dim, code_dim)

    def forward(self, x):
        out = self.features(x)
        return self.fc(out.flatten(1))
