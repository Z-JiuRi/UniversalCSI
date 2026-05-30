'''
深度可分离 CNN 编码器：用 Depthwise + Pointwise 卷积压缩 CSI 到 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
feature_dim = 4 * width * (nt // 4) * (nc // 4)

模型架构：
.
├── stem
│   ├── Conv2d(channel, width, 3, padding=1, bias=False)
│   ├── BatchNorm2d(width)
│   └── LeakyReLU(0.3)
├── DepthwiseSeparableBlock(width, width, stride=1)
│   ├── Conv2d(width, width, 3, stride=1, padding=1, groups=width, bias=False)
│   ├── BatchNorm2d(width)
│   ├── LeakyReLU(0.3)
│   ├── Conv2d(width, width, 1, padding=0, bias=False)
│   ├── BatchNorm2d(width)
│   └── LeakyReLU(0.3)
├── DepthwiseSeparableBlock(width, 2 * width, stride=2)
├── DepthwiseSeparableBlock(2 * width, 4 * width, stride=2)
├── DepthwiseSeparableBlock(4 * width, 4 * width, stride=1)
├── Flatten()
└── Linear(feature_dim, code_dim)

使用技术：
深度卷积负责逐通道空间滤波；1x1 卷积负责通道混合；相比标准卷积减少参数量和计算量。

保存模型权重时的参数维度：
features.stem.conv.weight:                                                     (width, channel, 3, 3)
features.stem.bn.weight/bias/running_mean/running_var:                                       (width,)
每个 DepthwiseSeparableBlock.depthwise.conv.weight:                                    (in_ch, 1, 3, 3)
每个 DepthwiseSeparableBlock.depthwise.bn.weight/bias/running_mean/running_var:                (in_ch,)
每个 DepthwiseSeparableBlock.pointwise.conv.weight:                               (out_ch, in_ch, 1, 1)
每个 DepthwiseSeparableBlock.pointwise.bn.weight/bias/running_mean/running_var:               (out_ch,)
fc.weight:                                                                    (code_dim, feature_dim)
fc.bias:                                                                                  (code_dim,)
'''


import torch.nn as nn
from collections import OrderedDict


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
        ]))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ("depthwise", ConvBN(in_channels, in_channels, 3, stride=stride,
                                 groups=in_channels)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("pointwise", ConvBN(in_channels, out_channels, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableCsiEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, width=16):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % 4 == 0 and nc % 4 == 0
        self.features = nn.Sequential(OrderedDict([
            ("stem", ConvBN(channel, width, 3)),
            ("stem_relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("block1", DepthwiseSeparableBlock(width, width)),
            ("block2", DepthwiseSeparableBlock(width, 2 * width, stride=2)),
            ("block3", DepthwiseSeparableBlock(2 * width, 4 * width,
                                               stride=2)),
            ("block4", DepthwiseSeparableBlock(4 * width, 4 * width)),
        ]))
        feature_dim = 4 * width * (nt // 4) * (nc // 4)
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
