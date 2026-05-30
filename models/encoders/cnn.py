'''
纯 CNN 编码器：用多层卷积下采样 CSI，并输出 (N, code_dim) 压缩码字。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
feature_dim = 4 * width * (nt // 4) * (nc // 4)

模型架构：
.
├── Conv2d(channel, width, 3, padding=1, bias=False)
├── BatchNorm2d(width)
├── LeakyReLU(0.3)
├── Conv2d(width, 2 * width, 3, stride=2, padding=1, bias=False)
├── BatchNorm2d(2 * width)
├── LeakyReLU(0.3)
├── Conv2d(2 * width, 4 * width, 3, stride=2, padding=1, bias=False)
├── BatchNorm2d(4 * width)
├── LeakyReLU(0.3)
├── Conv2d(4 * width, 4 * width, 3, padding=1, bias=False)
├── BatchNorm2d(4 * width)
├── LeakyReLU(0.3)
├── Flatten()
└── Linear(feature_dim, code_dim)

使用技术：
stride=2 卷积完成两次可学习下采样；卷积块保留局部角延迟结构；
全连接瓶颈把紧凑特征映射为固定长度反馈码字。

保存模型权重时的参数维度：
features.conv1.conv.weight:                                   (width, channel, 3, 3)
features.conv1.bn.weight/bias/running_mean/running_var:                     (width,)
features.conv2.conv.weight:                                 (2 * width, width, 3, 3)
features.conv2.bn.weight/bias/running_mean/running_var:                 (2 * width,)
features.conv3.conv.weight:                             (4 * width, 2 * width, 3, 3)
features.conv3.bn.weight/bias/running_mean/running_var:                 (4 * width,)
features.conv4.conv.weight:                             (4 * width, 4 * width, 3, 3)
features.conv4.bn.weight/bias/running_mean/running_var:                 (4 * width,)
fc.weight:                                                   (code_dim, feature_dim)
fc.bias:                                                                 (code_dim,)
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


class CNNEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, width=16):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % 4 == 0 and nc % 4 == 0
        self.features = nn.Sequential(OrderedDict([
            ("conv1", ConvBN(channel, width, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv2", ConvBN(width, 2 * width, 3, stride=2)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv3", ConvBN(2 * width, 4 * width, 3, stride=2)),
            ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv4", ConvBN(4 * width, 4 * width, 3)),
            ("relu4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        feature_dim = 4 * width * (nt // 4) * (nc // 4)
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))
