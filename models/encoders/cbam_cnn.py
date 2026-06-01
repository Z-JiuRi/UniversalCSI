'''
CBAM-CNN 编码器：在纯 CNN 下采样阶段加入 CBAM 注意力，输出 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
feature_dim = 4 * width * (nt // 4) * (nc // 4)

模型架构：
.
├── Conv2d(channel, width, 3, padding=1, bias=False)
├── BatchNorm2d(width)
├── LeakyReLU(0.3)
├── CBAMBlock(width)
│   ├── ChannelAttention(width)
│   │   ├── AdaptiveAvgPool2d(1)
│   │   ├── AdaptiveMaxPool2d(1)
│   │   ├── Conv2d(width, max(width // 16, 1), 1, bias=False)
│   │   ├── ReLU(inplace=True)
│   │   ├── Conv2d(max(width // 16, 1), width, 1, bias=False)
│   │   ├── Add()
│   │   ├── Sigmoid()
│   │   └── Mul()
│   └── SpatialAttention()
│       ├── Max(dim=1, keepdim=True)
│       ├── Mean(dim=1, keepdim=True)
│       ├── Concat(dim=1)
│       ├── Conv2d(2, 1, 7, padding=3, bias=False)
│       ├── Sigmoid()
│       └── Mul()
├── Conv2d(width, 2 * width, 3, stride=2, padding=1, bias=False)
├── BatchNorm2d(2 * width)
├── LeakyReLU(0.3)
├── CBAMBlock(2 * width)
├── Conv2d(2 * width, 4 * width, 3, stride=2, padding=1, bias=False)
├── BatchNorm2d(4 * width)
├── LeakyReLU(0.3)
├── CBAMBlock(4 * width)
├── Conv2d(4 * width, 4 * width, 3, padding=1, bias=False)
├── BatchNorm2d(4 * width)
├── LeakyReLU(0.3)
├── Flatten()
└── Linear(feature_dim, code_dim)

使用技术：
CBAM 先做通道注意力再做空间注意力；CNN 下采样保留局部角延迟结构，
注意力模块用于重标定重要通道和空间位置。

保存模型权重时的参数维度：
features.conv1.conv.weight:                                                   (width, channel, 3, 3)
features.conv1.bn.weight/bias/running_mean/running_var:                                     (width,)
features.cbam1.channel.mlp.0.weight:                              (max(width // 16, 1), width, 1, 1)
features.cbam1.channel.mlp.2.weight:                              (width, max(width // 16, 1), 1, 1)
features.cbam1.spatial.conv.weight:                                                     (1, 2, 7, 7)
features.conv2.conv.weight:                                                 (2 * width, width, 3, 3)
features.conv2.bn.weight/bias/running_mean/running_var:                                 (2 * width,)
features.cbam2.channel.mlp.0.weight:                    (max((2 * width) // 16, 1), 2 * width, 1, 1)
features.cbam2.channel.mlp.2.weight:                    (2 * width, max((2 * width) // 16, 1), 1, 1)
features.cbam2.spatial.conv.weight:                                                     (1, 2, 7, 7)
features.conv3.conv.weight:                                             (4 * width, 2 * width, 3, 3)
features.conv3.bn.weight/bias/running_mean/running_var:                                 (4 * width,)
features.cbam3.channel.mlp.0.weight:                    (max((4 * width) // 16, 1), 4 * width, 1, 1)
features.cbam3.channel.mlp.2.weight:                    (4 * width, max((4 * width) // 16, 1), 1, 1)
features.cbam3.spatial.conv.weight:                                                     (1, 2, 7, 7)
features.conv4.conv.weight:                                             (4 * width, 4 * width, 3, 3)
features.conv4.bn.weight/bias/running_mean/running_var:                                 (4 * width,)
fc.weight:                                                                   (code_dim, feature_dim)
fc.bias:                                                                                 (code_dim,)
'''


import torch
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


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(channels, hidden, 1, bias=False)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(hidden, channels, 1, bias=False)),
        ]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(scale)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_map = torch.max(x, dim=1, keepdim=True)[0]
        mean_map = torch.mean(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat((max_map, mean_map), dim=1)))
        return x * scale


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))


class CBAMCNNEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, width=16):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % 4 == 0 and nc % 4 == 0
        self.features = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(channel, width, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("cbam1", CBAMBlock(width)),
            ("conv2", ConvBN(width, 2 * width, 3, stride=2)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("cbam2", CBAMBlock(2 * width)),
            ("conv3", ConvBN(2 * width, 4 * width, 3, stride=2)),
            ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("cbam3", CBAMBlock(4 * width)),
            ("conv4", ConvBN(4 * width, 4 * width, 3)),
            ("relu4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        feature_dim = 4 * width * (nt // 4) * (nc // 4)
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
