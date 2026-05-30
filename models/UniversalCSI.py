'''
UniversalCSI 总模型工厂：组合任意编码器和解码器完成 CSI 压缩反馈自编码。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction

模型架构：
.
├── Encoder
│   ├── Input: (N, channel, nt, nc)
│   └── Output: (N, code_dim)
├── CodeAdapter(optional)
│   ├── LayerNorm(code_dim)
│   └── Linear(code_dim, code_dim)
└── Decoder
    ├── Input: (N, code_dim)
    └── Output: (N, channel, nt, nc)

使用技术：
编码器和解码器按架构分文件实现；encode(x) 固定返回完整压缩码字；
forward(x) 固定返回重建 CSI，用 MSE/NMSE 与原始 CSI 对齐评估。

保存模型权重时的参数维度：
encoder.*: 具体维度见 models/encoders/ 下对应架构文件顶部说明
code_adapter.net.0.weight:          (code_dim,) 仅 --code_adapter 启用时存在
code_adapter.net.0.bias:            (code_dim,) 仅 --code_adapter 启用时存在
code_adapter.net.1.weight: (code_dim, code_dim) 仅 --code_adapter 启用时存在
code_adapter.net.1.bias:            (code_dim,) 仅 --code_adapter 启用时存在
decoder.*: 具体维度见 models/decoders/ 下对应架构文件顶部说明
'''


import torch.nn as nn

from .decoders import CNNResidualDecoder, CNNRefinementHead, HybridDecoder
from .decoders import TransNetDecoder
from .encoders import AttentionCNNEncoder, CBAMCNNEncoder, CLNetEncoder
from .encoders import CNNEncoder
from .encoders import ConvNeXtCsiEncoder, CRNetEncoder, CsiNetEncoder
from .encoders import DepthwiseSeparableCsiEncoder, MLPAEEncoder
from .encoders import MLPMixerCsiEncoder, ResNetCsiEncoder
from .encoders import SparseTransformCsiEncoder, SwinCsiEncoder
from .encoders import TransNetEncoder

__all__ = [
    "universal_csi",
    "UniversalCSIModel",
    "CodeAdapter",
    "build_encoder",
    "build_decoder",
    "select_init_strategy",
    "CsiNetEncoder",
    "CNNEncoder",
    "CBAMCNNEncoder",
    "CRNetEncoder",
    "CLNetEncoder",
    "TransNetEncoder",
    "ResNetCsiEncoder",
    "DepthwiseSeparableCsiEncoder",
    "ConvNeXtCsiEncoder",
    "MLPMixerCsiEncoder",
    "AttentionCNNEncoder",
    "SwinCsiEncoder",
    "MLPAEEncoder",
    "SparseTransformCsiEncoder",
    "TransNetDecoder",
    "CNNResidualDecoder",
    "CNNRefinementHead",
    "HybridDecoder",
]


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


def select_init_strategy(encoder_name, decoder_name, code_adapter=False):
    encoder_name = encoder_name.lower()
    decoder_name = decoder_name.lower()
    if (encoder_name == "transnet" and decoder_name == "transnet" and not code_adapter):
        return "transnet"
    return "typed"


class UniversalCSIModel(nn.Module):
    def __init__(self, encoder, decoder, code_adapter=None, init_strategy="typed"):
        super().__init__()
        self.encoder = encoder
        self.code_adapter = code_adapter if code_adapter is not None else nn.Identity()
        self.decoder = decoder
        self.init_strategy = init_strategy
        self._reset_parameters(init_strategy)
        if hasattr(self.decoder, "reset_refinement_output"):
            self.decoder.reset_refinement_output()

    def _reset_parameters(self, strategy):
        if strategy == "transnet":
            self._reset_transnet_parameters()
        elif strategy == "typed":
            self._reset_typed_parameters()
        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    def _reset_transnet_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _reset_typed_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                continue

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                                    nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, a=0.3,
                                         nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                                    nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

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
    if name == "cnn":
        return CNNEncoder(reduction, channel, nt, nc)
    if name == "cbam_cnn":
        return CBAMCNNEncoder(reduction, channel, nt, nc)
    if name == "crnet":
        return CRNetEncoder(reduction, channel, nt, nc)
    if name == "clnet":
        return CLNetEncoder(reduction, channel, nt, nc)
    if name == "transnet":
        return TransNetEncoder(reduction, d_model, channel, nt, nc,
                               dim_feedforward)
    if name == "resnet":
        return ResNetCsiEncoder(reduction, channel, nt, nc)
    if name == "dscnn":
        return DepthwiseSeparableCsiEncoder(reduction, channel, nt, nc)
    if name == "convnext":
        return ConvNeXtCsiEncoder(reduction, channel, nt, nc)
    if name == "mlp_mixer":
        return MLPMixerCsiEncoder(reduction, d_model, channel, nt, nc)
    if name == "attention_cnn":
        return AttentionCNNEncoder(reduction, channel, nt, nc)
    if name == "swin":
        return SwinCsiEncoder(reduction, channel, nt, nc)
    if name == "mlp_ae":
        return MLPAEEncoder(reduction, channel, nt, nc)
    if name == "sparse_resnet":
        return SparseTransformCsiEncoder(reduction, channel, nt, nc)
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
    init_strategy = select_init_strategy(encoder_name, decoder_name,
                                         code_adapter)
    return UniversalCSIModel(encoder, decoder, adapter, init_strategy)
