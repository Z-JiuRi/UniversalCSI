import torch
import torch.nn as nn
from collections import OrderedDict

from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)

__all__ = [
    "universal_csi",
    "UniversalCSIModel",
    "CsiNetEncoder",
    "CRNetEncoder",
    "CLNetEncoder",
    "TransNetEncoder",
    "TransNetDecoder",
    "CNNResidualDecoder",
    "HybridDecoder",
]


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


class CRBlock(nn.Module):
    def __init__(self, compact=False):
        super().__init__()
        k1 = [1, 3] if compact else [1, 9]
        k2 = [3, 1] if compact else [9, 1]
        self.path1 = nn.Sequential(OrderedDict([
            ("conv3x3", ConvBN(2, 7, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv_k1", ConvBN(7, 7, k1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv_k2", ConvBN(7, 7, k2)),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ("conv1x5", ConvBN(2, 7, [1, 5])),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv5x1", ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(14, 2, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = x
        out = torch.cat((self.path1(x), self.path2(x)), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)
        return self.relu(out + identity)


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


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

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


class TransNetEncoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        self.feature_shape = (input_dim // d_model, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, input_dim // reduction)

    def forward(self, x):
        batch_size = x.size(0)
        memory = self.encoder(x.view(batch_size, self.feature_shape[0],
                                     self.feature_shape[1]))
        return self.fc(memory.flatten(1))


class CodeAdapter(nn.Module):
    def __init__(self, code_dim, enabled=False):
        super().__init__()
        self.enabled = enabled
        self.net = nn.Sequential(
            nn.LayerNorm(code_dim),
            nn.Linear(code_dim, code_dim),
        ) if enabled else nn.Identity()

    def forward(self, code):
        return self.net(code)


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
    def __init__(self, channel=2, hidden=16, num_blocks=2):
        super().__init__()
        layers = [
            ("conv_in", nn.Conv2d(channel, hidden, 3, padding=1, bias=False)),
            ("relu_in", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]
        for idx in range(num_blocks):
            layers.append((f"res{idx + 1}", ConvResidualBlock(hidden)))
        layers.append(("conv_out", nn.Conv2d(hidden, channel, 3, padding=1)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)


class TransNetDecoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.feature_shape = (input_dim // d_model, d_model)
        self.fc_decoder = nn.Linear(input_dim // reduction, input_dim)
        decoder_layer = TransformerDecoderLayer(
            d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2,
                                          norm=nn.LayerNorm(d_model))

    def forward(self, code):
        batch_size = code.size(0)
        memory = self.fc_decoder(code).view(batch_size, self.feature_shape[0],
                                            self.feature_shape[1])
        out = self.decoder(memory, memory)
        out = out.view(batch_size, self.channel, self.nt, self.nc)
        return out


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


class HybridDecoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 dim_feedforward=None, hidden=16, num_blocks=2):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % d_model == 0
        assert input_dim % reduction == 0
        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.feature_shape = (input_dim // d_model, d_model)
        code_dim = input_dim // reduction
        self.code_norm = nn.LayerNorm(code_dim)
        self.fc_decoder = nn.Linear(code_dim, input_dim)
        decoder_layer = TransformerDecoderLayer(
            d_model, 2, dim_feedforward, dropout=0., batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2,
                                          norm=nn.LayerNorm(d_model))
        self.refine = CNNRefinementHead(channel, hidden, num_blocks)

    def forward(self, code):
        batch_size = code.size(0)
        memory = self.fc_decoder(self.code_norm(code))
        memory = memory.view(batch_size, self.feature_shape[0],
                             self.feature_shape[1])
        out = self.decoder(memory, memory)
        coarse = out.view(batch_size, self.channel, self.nt, self.nc)
        return coarse + self.refine(coarse)


class UniversalCSIModel(nn.Module):
    def __init__(self, encoder, decoder, code_adapter=None):
        super().__init__()
        self.encoder = encoder
        self.code_adapter = code_adapter if code_adapter is not None else nn.Identity()
        self.decoder = decoder
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        code = self.encode(x)
        return self.decoder(code)

    def encode(self, x):
        return self.code_adapter(self.encoder(x))


def build_encoder(name, reduction, d_model=64, channel=2, nt=32, nc=32,
                  dim_feedforward=None):
    name = name.lower()
    if name == "csinet":
        return CsiNetEncoder(reduction, channel, nt, nc)
    if name == "crnet":
        return CRNetEncoder(reduction, channel, nt, nc)
    if name == "clnet":
        return CLNetEncoder(reduction, channel, nt, nc)
    if name == "transnet":
        return TransNetEncoder(reduction, d_model, channel, nt, nc,
                               dim_feedforward)
    raise ValueError(f"Unknown encoder: {name}")


def build_decoder(name, reduction, d_model=64, channel=2, nt=32, nc=32,
                  dim_feedforward=None):
    name = name.lower()
    if name == "transnet":
        return TransNetDecoder(reduction, d_model, channel, nt, nc,
                               dim_feedforward)
    if name == "cnn_residual":
        return CNNResidualDecoder(reduction, d_model, channel, nt, nc,
                                  dim_feedforward)
    if name == "hybrid":
        return HybridDecoder(reduction, d_model, channel, nt, nc,
                             dim_feedforward)
    raise ValueError(f"Unknown decoder: {name}")


def universal_csi(encoder_name="transnet", reduction=4, d_model=64,
                  channel=2, nt=32, nc=32, dim_feedforward=None,
                  code_adapter=False, decoder_name="transnet"):
    input_dim = channel * nt * nc
    code_dim = input_dim // reduction
    encoder = build_encoder(encoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward)
    decoder = build_decoder(decoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward)
    adapter = CodeAdapter(code_dim, enabled=code_adapter)
    return UniversalCSIModel(encoder, decoder, adapter)
