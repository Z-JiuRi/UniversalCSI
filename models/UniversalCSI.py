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
└── Decoder
    ├── Input: (N, code_dim)
    └── Output: (N, channel, nt, nc)

使用技术：
编码器和解码器按架构分文件实现；encode(x) 固定返回完整压缩码字；
forward(x) 固定返回重建 CSI，用 MSE/NMSE 与原始 CSI 对齐评估。

保存模型权重时的参数维度：
encoder.*: 具体维度见 models/encoders/ 下对应架构文件顶部说明
decoder.*: 具体维度见 models/decoders/ 下对应架构文件顶部说明
'''


import torch
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


def select_init_strategy(encoder_name, decoder_name):
    encoder_name = encoder_name.lower()
    decoder_name = decoder_name.lower()
    if encoder_name == "transnet" and decoder_name == "transnet":
        return "transnet"
    return "typed"


class UniversalCSIModel(nn.Module):
    def __init__(self, encoder, decoder, init_strategy="typed",
                 code_adapter=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.code_adapter = code_adapter
        self.init_strategy = init_strategy
        self._reset_parameters(init_strategy)
        if hasattr(self.encoder, "reset_canonical_head_parameters"):
            self.encoder.reset_canonical_head_parameters()
        if self.code_adapter is not None:
            self.code_adapter.reset_parameters()
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
        code = self.encoder(x)
        if self.code_adapter is not None:
            code = self.code_adapter(code)
        return code

    @torch.no_grad()
    def adapter_metrics(self):
        if self.code_adapter is not None:
            return self.code_adapter.get_metrics()
        return {}



def build_encoder(name, reduction, d_model=64, channel=2, nt=32, nc=32,
                  dim_feedforward=None, canonical_head=None,
                  canonical_anchor_seed=0, canonical_lowrank_rank=0,
                  canonical_lowrank_scale=0.0,
                  canonical_codebook_size=1024,
                  canonical_codebook_temperature=1.0):
    name = name.lower()
    if canonical_head not in (None, "", "none") and name != "transnet":
        raise ValueError("canonical_head is currently supported only for "
                         "encoder=transnet")
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
                               dim_feedforward,
                               canonical_head=canonical_head,
                               canonical_anchor_seed=canonical_anchor_seed,
                               canonical_lowrank_rank=canonical_lowrank_rank,
                               canonical_lowrank_scale=canonical_lowrank_scale,
                               canonical_codebook_size=canonical_codebook_size,
                               canonical_codebook_temperature=canonical_codebook_temperature)
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
                  dim_feedforward=None, hidden=16, num_blocks=2):
    name = name.lower()
    if name == "transnet":
        return TransNetDecoder(reduction, d_model, channel, nt, nc,
                               dim_feedforward)
    if name == "cnn_residual":
        return CNNResidualDecoder(reduction, d_model, channel, nt, nc,
                                  dim_feedforward)
    if name == "hybrid":
        return HybridDecoder(reduction, d_model, channel, nt, nc,
                             dim_feedforward, hidden=hidden,
                             num_blocks=num_blocks)
    raise ValueError(f"Unknown decoder: {name}")


def universal_csi(encoder_name="transnet", reduction=4, d_model=64,
                  channel=2, nt=32, nc=32, dim_feedforward=None,
                  decoder_name="transnet", hidden=16, num_blocks=2,
                  adapter=None, adapter_hidden_dim=None,
                  canonical_head=None, canonical_anchor_seed=0,
                  canonical_lowrank_rank=0, canonical_lowrank_scale=0.0,
                  canonical_codebook_size=1024,
                  canonical_codebook_temperature=1.0):
    encoder = build_encoder(encoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward,
                            canonical_head=canonical_head,
                            canonical_anchor_seed=canonical_anchor_seed,
                            canonical_lowrank_rank=canonical_lowrank_rank,
                            canonical_lowrank_scale=canonical_lowrank_scale,
                            canonical_codebook_size=canonical_codebook_size,
                            canonical_codebook_temperature=canonical_codebook_temperature)
    decoder = build_decoder(decoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward, hidden=hidden,
                            num_blocks=num_blocks)
    init_strategy = select_init_strategy(encoder_name, decoder_name)
    code_adapter = None
    if adapter == "mlp":
        input_dim = channel * nt * nc
        code_dim = input_dim // reduction
        from .adapters import MLPAdapter
        code_adapter = MLPAdapter(code_dim, adapter_hidden_dim)
    elif adapter == "mlp_direct":
        input_dim = channel * nt * nc
        code_dim = input_dim // reduction
        from .adapters import MLPDirectAdapter
        code_adapter = MLPDirectAdapter(code_dim, adapter_hidden_dim)
    elif adapter == "transformer":
        input_dim = channel * nt * nc
        code_dim = input_dim // reduction
        from .adapters import TransformerAdapter
        code_adapter = TransformerAdapter(code_dim, d_model=d_model, dim_feedforward=adapter_hidden_dim)
    elif adapter is not None:
        raise ValueError(f"Unknown adapter: {adapter}")
    return UniversalCSIModel(encoder, decoder, init_strategy,
                             code_adapter=code_adapter)
