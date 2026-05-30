'''
CNN 残差解码器：把 (N, code_dim) 码字重建为 (N, channel, nt, nc)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction

模型架构：
.
├── LayerNorm(code_dim)
├── Linear(code_dim, input_dim)
├── View(N, channel, nt, nc)
├── CNNRefinementHead
│   ├── Conv2d(channel, hidden, 3, padding=1, bias=False)
│   ├── LeakyReLU(0.3)
│   ├── ConvResidualBlock(hidden) * num_blocks
│   │   ├── Conv2d(hidden, hidden, 3, padding=1, bias=False)
│   │   ├── BatchNorm2d(hidden)
│   │   ├── LeakyReLU(0.3)
│   │   ├── Conv2d(hidden, hidden, 3, padding=1, bias=False)
│   │   ├── BatchNorm2d(hidden)
│   │   ├── Add()
│   │   └── LeakyReLU(0.3)
│   └── Conv2d(hidden, channel, 3, padding=1)
└── Add(coarse, refine)

使用技术：
LayerNorm 规范化码字；Linear 负责全局粗重建；CNN 残差头补偿局部细节。

保存模型权重时的参数维度：
code_norm.weight:                                                                   (code_dim,)
code_norm.bias:                                                                     (code_dim,)
fc_decoder.weight:                                                        (input_dim, code_dim)
fc_decoder.bias:                                                                   (input_dim,)
refine.net.conv_in.weight:                                              (hidden, channel, 3, 3)
每个 refine.net.res*.block.conv1.conv.weight:                              (hidden, hidden, 3, 3)
每个 refine.net.res*.block.conv1.bn.weight/bias/running_mean/running_var:               (hidden,)
每个 refine.net.res*.block.conv2.conv.weight:                              (hidden, hidden, 3, 3)
每个 refine.net.res*.block.conv2.bn.weight/bias/running_mean/running_var:               (hidden,)
refine.net.conv_out.weight:                                             (channel, hidden, 3, 3)
refine.net.conv_out.bias:                                                            (channel,)
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


class ConvResidualBlock(nn.Module):
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


class CNNRefinementHead(nn.Module):
    def __init__(self, channel=2, hidden=16, num_blocks=2,
                 zero_init_output=False):
        super().__init__()
        layers = [
            ("conv_in", nn.Conv2d(channel, hidden, 3, padding=1, bias=False)),
            ("relu_in", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]
        for idx in range(num_blocks):
            layers.append((f"res{idx + 1}", ConvResidualBlock(hidden)))
        layers.append(("conv_out", nn.Conv2d(hidden, channel, 3, padding=1)))
        self.net = nn.Sequential(OrderedDict(layers))
        if zero_init_output:
            self.reset_output()

    def reset_output(self):
        nn.init.constant_(self.net.conv_out.weight, 0)
        nn.init.constant_(self.net.conv_out.bias, 0)

    def forward(self, x):
        return self.net(x)


class CNNResidualDecoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None, hidden=16, num_blocks=2):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        self.channel = channel
        self.nt = nt
        self.nc = nc
        code_dim = input_dim // reduction
        self.code_norm = nn.LayerNorm(code_dim)
        self.fc_decoder = nn.Linear(code_dim, input_dim)
        self.refine = CNNRefinementHead(channel, hidden, num_blocks)

    def forward(self, code):
        batch_size = code.size(0)
        coarse = self.fc_decoder(self.code_norm(code))
        coarse = coarse.view(batch_size, self.channel, self.nt, self.nc)
        return coarse + self.refine(coarse)
