'''
注意力 CNN 编码器：卷积下采样结合 SE/空间注意力，输出 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
feature_dim = 4 * width * (nt // 4) * (nc // 4)

模型架构：
.
├── Conv2d(channel, width, 3, padding=1, bias=False)
├── BatchNorm2d(width)
├── LeakyReLU(0.3)
├── SELayer(width)
│   ├── AdaptiveAvgPool2d(1)
│   ├── Flatten()
│   ├── Linear(width, max(width // 16, 1), bias=False)
│   ├── ReLU(inplace=True)
│   ├── Linear(max(width // 16, 1), width, bias=False)
│   ├── Sigmoid()
│   ├── View(N, width, 1, 1)
│   └── Mul()
├── Conv2d(width, 2 * width, 3, stride=2, padding=1, bias=False)
├── BatchNorm2d(2 * width)
├── LeakyReLU(0.3)
├── SpatialGate
│   ├── Max(dim=1, keepdim=True)
│   ├── Mean(dim=1, keepdim=True)
│   ├── Concat(dim=1)
│   ├── Conv2d(2, 1, 3, padding=1, bias=False)
│   ├── BatchNorm2d(1)
│   ├── Sigmoid()
│   └── Mul()
├── Conv2d(2 * width, 4 * width, 3, stride=2, padding=1, bias=False)
├── BatchNorm2d(4 * width)
├── LeakyReLU(0.3)
├── SELayer(4 * width)
├── Flatten()
└── Linear(feature_dim, code_dim)

使用技术：
SE 通道注意力突出重要通道；SpatialGate 根据最大池化/平均池化生成空间权重；
两次 stride=2 卷积降低空间分辨率后生成反馈码字。

保存模型权重时的参数维度：
features.conv1.conv.weight:                                                   (width, channel, 3, 3)
features.conv1.bn.weight/bias/running_mean/running_var:                                     (width,)
features.se1.fc.fc1.weight:                                             (max(width // 16, 1), width)
features.se1.fc.fc2.weight:                                             (width, max(width // 16, 1))
features.conv2.conv.weight:                                                 (2 * width, width, 3, 3)
features.conv2.bn.weight/bias/running_mean/running_var:                                 (2 * width,)
features.sa2.spatial.conv.weight:                                                       (1, 2, 3, 3)
features.sa2.spatial.bn.weight/bias/running_mean/running_var:                                   (1,)
features.conv3.conv.weight:                                             (4 * width, 2 * width, 3, 3)
features.conv3.bn.weight/bias/running_mean/running_var:                                 (4 * width,)
features.se3.fc.fc1.weight:                                   (max((4 * width) // 16, 1), 4 * width)
features.se3.fc.fc2.weight:                                   (4 * width, max((4 * width) // 16, 1))
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


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, 3, padding=1, relu=False)

    def forward(self, x):
        scale = torch.sigmoid(self.spatial(self.compress(x)))
        return x * scale


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(channel, hidden, bias=False)),
            ("relu", nn.ReLU(inplace=True)),
            ("fc2", nn.Linear(hidden, channel, bias=False)),
            ("sigmoid", nn.Sigmoid()),
        ]))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionCNNEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, width=16):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % 4 == 0 and nc % 4 == 0
        self.features = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(channel, width, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("se1", SELayer(width)),
            ("conv2", ConvBN(width, 2 * width, 3, stride=2)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("sa2", SpatialGate()),
            ("conv3", ConvBN(2 * width, 4 * width, 3, stride=2)),
            ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("se3", SELayer(4 * width)),
        ]))
        feature_dim = 4 * width * (nt // 4) * (nc // 4)
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
