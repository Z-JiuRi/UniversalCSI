'''
CRNet 编码器：多分支卷积 CSI 编码器，把 (N, channel, nt, nc) 压缩为 (N, code_dim)。

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
├── encoder2
│   ├── Conv2d(channel, 2, 3, padding=1, bias=False)
│   └── BatchNorm2d(2)
├── Concat(dim=1)
├── encoder_conv
│   ├── LeakyReLU(0.3)
│   ├── Conv2d(4, 2, 1, padding=0, bias=False)
│   ├── BatchNorm2d(2)
│   └── LeakyReLU(0.3)
├── Flatten()
└── Linear(input_dim, code_dim)

使用技术：
多分支卷积扩大感受野；1x9 和 9x1 分解卷积建模角延迟域方向相关性；
1x1 卷积融合分支特征后用全连接层生成压缩码字。

保存模型权重时的参数维度：
encoder1.conv3x3_bn.conv.weight:                                    (2, channel, 3, 3)
encoder1.conv3x3_bn.bn.weight/bias/running_mean/running_var:                      (2,)
encoder1.conv1x9_bn.conv.weight:                                          (2, 2, 1, 9)
encoder1.conv1x9_bn.bn.weight/bias/running_mean/running_var:                      (2,)
encoder1.conv9x1_bn.conv.weight:                                          (2, 2, 9, 1)
encoder1.conv9x1_bn.bn.weight/bias/running_mean/running_var:                      (2,)
encoder2.conv.weight:                                               (2, channel, 3, 3)
encoder2.bn.weight/bias/running_mean/running_var:                                 (2,)
encoder_conv.conv1x1_bn.conv.weight:                                      (2, 4, 1, 1)
encoder_conv.conv1x1_bn.bn.weight/bias/running_mean/running_var:                  (2,)
fc.weight:                                                       (code_dim, input_dim)
fc.bias:                                                                   (code_dim,)
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


class CRNetEncoder(nn.Module):
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
        self.encoder2 = ConvBN(channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.fc = nn.Linear(input_dim, input_dim // reduction)

    def forward(self, x):
        out = torch.cat((self.encoder1(x), self.encoder2(x)), dim=1)
        out = self.encoder_conv(out)
        return self.fc(out.flatten(1))
