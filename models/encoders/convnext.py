'''
ConvNeXt CSI 编码器：ConvNeXt 风格卷积块把 CSI 压缩为 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
feature_dim = 4 * width * (nt // 4) * (nc // 4)

模型架构：
.
├── stem
│   └── Conv2d(channel, width, 3, padding=1)
├── ConvNeXtBlock(width) * depths[0]
│   ├── Conv2d(width, width, 7, padding=3, groups=width)
│   ├── GroupNorm(1, width)
│   ├── Conv2d(width, 4 * width, 1)
│   ├── GELU()
│   ├── Conv2d(4 * width, width, 1)
│   └── Add()
├── down1
│   └── Conv2d(width, 2 * width, 2, stride=2)
├── ConvNeXtBlock(2 * width) * depths[1]
├── down2
│   └── Conv2d(2 * width, 4 * width, 2, stride=2)
├── ConvNeXtBlock(4 * width) * depths[2]
├── Flatten()
├── LayerNorm(feature_dim)
└── Linear(feature_dim, code_dim)

使用技术：
大核深度卷积建模局部空间上下文；1x1 卷积实现通道扩展和压缩；
ConvNeXt 残差结构提升深层特征表达。

保存模型权重时的参数维度：
features.stem.weight:                         (width, channel, 3, 3)
features.stem.bias:                                         (width,)
features.down1.weight:                      (2 * width, width, 2, 2)
features.down1.bias:                                    (2 * width,)
features.down2.weight:                  (4 * width, 2 * width, 2, 2)
features.down2.bias:                                    (4 * width,)
每个 ConvNeXtBlock.dwconv.weight:                  (channels, 1, 7, 7)
每个 ConvNeXtBlock.dwconv.bias:                            (channels,)
每个 ConvNeXtBlock.norm.weight/bias:                       (channels,)
每个 ConvNeXtBlock.pwconv.conv1.weight: (4 * channels, channels, 1, 1)
每个 ConvNeXtBlock.pwconv.conv1.bias:                  (4 * channels,)
每个 ConvNeXtBlock.pwconv.conv2.weight: (channels, 4 * channels, 1, 1)
每个 ConvNeXtBlock.pwconv.conv2.bias:                      (channels,)
head_norm.weight:                                     (feature_dim,)
head_norm.bias:                                       (feature_dim,)
fc.weight:                                   (code_dim, feature_dim)
fc.bias:                                                 (code_dim,)
'''


import torch.nn as nn
from collections import OrderedDict


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, expansion=4, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dwconv = nn.Conv2d(channels, channels, kernel_size,
                                padding=padding, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pwconv = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(channels, expansion * channels, 1)),
            ("gelu", nn.GELU()),
            ("conv2", nn.Conv2d(expansion * channels, channels, 1)),
        ]))

    def forward(self, x):
        identity = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv(out)
        return identity + out


class ConvNeXtCsiEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, width=16,
                 depths=(1, 1, 1)):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % 4 == 0 and nc % 4 == 0
        layers = [
            ("stem", nn.Conv2d(channel, width, 3, padding=1)),
        ]
        for idx in range(depths[0]):
            layers.append((f"stage1_block{idx + 1}", ConvNeXtBlock(width)))
        layers.extend([
            ("down1", nn.Conv2d(width, 2 * width, 2, stride=2)),
        ])
        for idx in range(depths[1]):
            layers.append((f"stage2_block{idx + 1}",
                           ConvNeXtBlock(2 * width)))
        layers.extend([
            ("down2", nn.Conv2d(2 * width, 4 * width, 2, stride=2)),
        ])
        for idx in range(depths[2]):
            layers.append((f"stage3_block{idx + 1}",
                           ConvNeXtBlock(4 * width)))
        self.features = nn.Sequential(OrderedDict(layers))
        feature_dim = 4 * width * (nt // 4) * (nc // 4)
        self.head_norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def forward(self, x):
        out = self.features(x).flatten(1)
        return self.fc(self.head_norm(out))
