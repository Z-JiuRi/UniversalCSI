'''
稀疏变换 ResNet 编码器：先做稀疏化变换，再用残差网络输出 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
feature_dim = 4 * width * (nt // 4) * (nc // 4)

模型架构：
.
├── transform
│   ├── Conv2d(channel, width, 1, padding=0, bias=False)
│   ├── BatchNorm2d(width)
│   ├── LeakyReLU(0.3)
│   ├── Conv2d(width, width, 3, padding=1, groups=width, bias=False)
│   ├── BatchNorm2d(width)
│   └── SoftThreshold(width)
│       ├── Parameter(1, width, 1, 1)
│       ├── Abs()
│       ├── Sub()
│       ├── ReLU()
│       ├── Sign()
│       └── Mul()
├── CsiResidualBlock(width)
│   ├── Conv2d(width, width, 3, padding=1, bias=False)
│   ├── BatchNorm2d(width)
│   ├── LeakyReLU(0.3)
│   ├── Conv2d(width, width, 3, padding=1, bias=False)
│   ├── BatchNorm2d(width)
│   ├── Add()
│   └── LeakyReLU(0.3)
├── stage2
│   ├── Conv2d(width, 2 * width, 3, stride=2, padding=1, bias=False)
│   ├── BatchNorm2d(2 * width)
│   ├── LeakyReLU(0.3)
│   └── CsiResidualBlock(2 * width)
├── stage3
│   ├── Conv2d(2 * width, 4 * width, 3, stride=2, padding=1, bias=False)
│   ├── BatchNorm2d(4 * width)
│   ├── LeakyReLU(0.3)
│   └── CsiResidualBlock(4 * width)
├── Flatten()
└── Linear(feature_dim, code_dim)

使用技术：
SoftThreshold 使用可学习阈值执行软收缩，突出 CSI 稀疏结构；深度卷积执行
逐通道空间滤波；残差块在稀疏变换后继续提取稳健特征。

保存模型权重时的参数维度：
transform.conv1x1.conv.weight:                                                 (width, channel, 1, 1)
transform.conv1x1.bn.weight/bias/running_mean/running_var:                                   (width,)
transform.depthwise.conv.weight:                                                     (width, 1, 3, 3)
transform.depthwise.bn.weight/bias/running_mean/running_var:                                 (width,)
transform.shrink.threshold:                                                          (1, width, 1, 1)
stage2.down.conv.weight:                                                     (2 * width, width, 3, 3)
stage2.down.bn.weight/bias/running_mean/running_var:                                     (2 * width,)
stage3.down.conv.weight:                                                 (4 * width, 2 * width, 3, 3)
stage3.down.bn.weight/bias/running_mean/running_var:                                     (4 * width,)
每个 CsiResidualBlock.block.conv1.conv.weight:                               (channels, channels, 3, 3)
每个 CsiResidualBlock.block.conv1.bn.weight/bias/running_mean/running_var:                  (channels,)
每个 CsiResidualBlock.block.conv2.conv.weight:                               (channels, channels, 3, 3)
每个 CsiResidualBlock.block.conv2.bn.weight/bias/running_mean/running_var:                  (channels,)
fc.weight:                                                                    (code_dim, feature_dim)
fc.bias:                                                                                  (code_dim,)
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


class CsiResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(channels, channels, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv2", ConvBN(channels, channels, 3)),
        ]))
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class SoftThreshold(nn.Module):
    def __init__(self, channels, init_threshold=0.01):
        super().__init__()
        self.threshold = nn.Parameter(
            torch.full((1, channels, 1, 1), init_threshold))

    def forward(self, x):
        threshold = torch.abs(self.threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class SparseTransformCsiEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, width=16):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % 4 == 0 and nc % 4 == 0
        self.transform = nn.Sequential(OrderedDict([
            ("conv1x1", ConvBN(channel, width, 1)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("depthwise", ConvBN(width, width, 3, groups=width)),
            ("shrink", SoftThreshold(width)),
        ]))
        self.stage1 = CsiResidualBlock(width)
        self.stage2 = nn.Sequential(OrderedDict([
            ("down", ConvBN(width, 2 * width, 3, stride=2)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("res", CsiResidualBlock(2 * width)),
        ]))
        self.stage3 = nn.Sequential(OrderedDict([
            ("down", ConvBN(2 * width, 4 * width, 3, stride=2)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("res", CsiResidualBlock(4 * width)),
        ]))
        feature_dim = 4 * width * (nt // 4) * (nc // 4)
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def forward(self, x):
        out = self.transform(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return self.fc(out.flatten(1))
