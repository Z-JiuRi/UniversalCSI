'''
CLNet 编码器：带空间注意力和通道注意力的 CSI 编码器，把输入压缩为 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction

模型架构：
.
├── encoder1
│   ├── Conv2d(channel, 2, 3, padding=1, bias=False)
│   ├── BatchNorm2d(2)
│   ├── LeakyReLU(0.3)
│   ├── Conv2d(2, 2, (1, 9), padding=(0, 4), bias=False)
│   ├── BatchNorm2d(2)
│   ├── LeakyReLU(0.3)
│   ├── Conv2d(2, 2, (9, 1), padding=(4, 0), bias=False)
│   └── BatchNorm2d(2)
├── SpatialGate
│   ├── Max(dim=1, keepdim=True)
│   ├── Mean(dim=1, keepdim=True)
│   ├── Concat(dim=1)
│   ├── Conv2d(2, 1, 3, padding=1, bias=False)
│   ├── BatchNorm2d(1)
│   ├── Sigmoid()
│   └── Mul()
├── encoder2
│   ├── Conv2d(channel, 32, 1, padding=0, bias=False)
│   └── BatchNorm2d(32)
├── SELayer(32)
│   ├── AdaptiveAvgPool2d(1)
│   ├── Flatten()
│   ├── Linear(32, max(32 // 16, 1), bias=False)
│   ├── ReLU(inplace=True)
│   ├── Linear(max(32 // 16, 1), 32, bias=False)
│   ├── Sigmoid()
│   ├── View(N, 32, 1, 1)
│   └── Mul()
├── Concat(dim=1)
├── encoder_conv
│   ├── LeakyReLU(0.3)
│   ├── Conv2d(34, 2, 1, padding=0, bias=False)
│   ├── BatchNorm2d(2)
│   └── LeakyReLU(0.3)
├── Flatten()
├── Unsqueeze(dim=2)
├── Conv1d(input_dim, code_dim, 1)
└── Squeeze(dim=2)

使用技术：
空间注意力使用通道最大池化和平均池化生成二维权重；SE 通道注意力重标定
32 通道分支；多尺度卷积和注意力融合后用 1D 卷积形成反馈码字。

保存模型权重时的参数维度：
encoder1.conv3x3_bn.conv.weight:                                       (2, channel, 3, 3)
encoder1.conv3x3_bn.bn.weight/bias/running_mean/running_var:                         (2,)
encoder1.conv1x9_bn.conv.weight:                                             (2, 2, 1, 9)
encoder1.conv1x9_bn.bn.weight/bias/running_mean/running_var:                         (2,)
encoder1.conv9x1_bn.conv.weight:                                             (2, 2, 9, 1)
encoder1.conv9x1_bn.bn.weight/bias/running_mean/running_var:                         (2,)
encoder2.conv.weight:                                                 (32, channel, 1, 1)
encoder2.bn.weight/bias/running_mean/running_var:                                   (32,)
sa.spatial.conv.weight:                                                      (1, 2, 3, 3)
sa.spatial.bn.weight/bias/running_mean/running_var:                                  (1,)
se.fc.fc1.weight:                                                  (max(32 // 16, 1), 32)
se.fc.fc2.weight:                                                  (32, max(32 // 16, 1))
encoder_conv.conv1x1_bn.conv.weight:                                        (2, 34, 1, 1)
encoder_conv.conv1x1_bn.bn.weight/bias/running_mean/running_var:                     (2,)
compress.weight:                                                 (code_dim, input_dim, 1)
compress.bias:                                                                (code_dim,)
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
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
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


class CLNetEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32):
        super().__init__()
        input_dim = channel * nt * nc
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(channel, 32, 1)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(34, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.sa = SpatialGate()
        self.se = SELayer(32)
        self.compress = nn.Conv1d(input_dim, input_dim // reduction, 1)

    def forward(self, x):
        out1 = self.sa(self.encoder1(x))
        out2 = self.se(self.encoder2(x))
        out = self.encoder_conv(torch.cat((out1, out2), dim=1))
        out = out.flatten(1).unsqueeze(2)
        return self.compress(out).squeeze(2)
