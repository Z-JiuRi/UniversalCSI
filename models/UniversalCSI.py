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

from .adapters import CodeAdapter
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
    "MultiSeedEncoderAdapterCSI",
    "build_encoder",
    "build_decoder",
    "multi_seed_adapter_csi",
    "select_init_strategy",
    "CodeAdapter",
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
    def __init__(self, encoder, decoder, init_strategy="typed"):
        super().__init__()
        self.encoder = encoder
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
        return self.encoder(x)


def _reset_module_parameters(module, strategy):
    if strategy == "transnet":
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return

    if strategy != "typed":
        raise ValueError(f"Unknown init strategy: {strategy}")

    for child in module.modules():
        if isinstance(child, nn.MultiheadAttention):
            if child.in_proj_weight is not None:
                nn.init.xavier_uniform_(child.in_proj_weight)
            if child.in_proj_bias is not None:
                nn.init.constant_(child.in_proj_bias, 0)
            continue

        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.constant_(child.bias, 0)
        elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                               nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(child.weight, a=0.3,
                                     nonlinearity="leaky_relu")
            if child.bias is not None:
                nn.init.constant_(child.bias, 0)
        elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.LayerNorm, nn.GroupNorm)):
            if child.weight is not None:
                nn.init.constant_(child.weight, 1)
            if child.bias is not None:
                nn.init.constant_(child.bias, 0)


def _seeded_reset(module, seed, strategy):
    devices = []
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        _reset_module_parameters(module, strategy)


def _seeded_adapter_reset(adapter, seed):
    devices = []
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        adapter.reset_parameters()


class MultiSeedEncoderAdapterCSI(nn.Module):
    def __init__(self, encoders, adapter, decoder):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.adapter = adapter
        self.decoder = decoder
        self.encoder_keys = list(encoders.keys())
        self.latest_adapter_stats = {}

    def forward(self, x):
        outputs = {}
        self.latest_adapter_stats = {}
        for key, encoder in self.encoders.items():
            code = encoder(x)
            adapted_code = self.adapter(code)
            with torch.no_grad():
                residual = adapted_code - code
                ratio = residual.norm() / code.norm().clamp_min(1e-12)
                self.latest_adapter_stats[key] = {
                    "residual_ratio": float(ratio.detach().cpu())
                }
            outputs[key] = self.decoder(adapted_code)
        return outputs

    def encode(self, x):
        return {key: encoder(x) for key, encoder in self.encoders.items()}

    def adapter_metrics(self):
        metrics = {
            "adapter/proj_weight_norm": float(
                self.adapter.proj.weight.detach().norm().cpu()),
            "adapter/proj_bias_norm": float(
                self.adapter.proj.bias.detach().norm().cpu()),
            "adapter/norm_weight_mean": float(
                self.adapter.norm.weight.detach().mean().cpu()),
            "adapter/norm_bias_norm": float(
                self.adapter.norm.bias.detach().norm().cpu()),
        }
        for key, stats in self.latest_adapter_stats.items():
            for name, value in stats.items():
                metrics[f"encoders/{key}/adapter_{name}"] = value
        return metrics


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
                  decoder_name="transnet", hidden=16, num_blocks=2):
    encoder = build_encoder(encoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward)
    decoder = build_decoder(decoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward, hidden=hidden,
                            num_blocks=num_blocks)
    init_strategy = select_init_strategy(encoder_name, decoder_name)
    return UniversalCSIModel(encoder, decoder, init_strategy)


def multi_seed_adapter_csi(encoder_name="transnet", reduction=4, d_model=64,
                           channel=2, nt=32, nc=32, dim_feedforward=None,
                           decoder_name="transnet", hidden=16, num_blocks=2,
                           encoder_seeds=None, decoder_seed=None):
    if not encoder_seeds:
        raise ValueError("encoder_seeds must contain at least one seed")

    init_strategy = select_init_strategy(encoder_name, decoder_name)
    encoders = {}
    for seed in encoder_seeds:
        key = f"seed{seed}"
        encoder = build_encoder(encoder_name, reduction, d_model, channel, nt,
                                nc, dim_feedforward)
        _seeded_reset(encoder, seed, init_strategy)
        encoders[key] = encoder

    decoder = build_decoder(decoder_name, reduction, d_model, channel, nt, nc,
                            dim_feedforward, hidden=hidden,
                            num_blocks=num_blocks)
    _reset_module_parameters(decoder, init_strategy)
    if hasattr(decoder, "reset_refinement_output"):
        decoder.reset_refinement_output()

    input_dim = channel * nt * nc
    if input_dim % reduction != 0:
        raise ValueError(
            f"input_dim={input_dim} must be divisible by reduction={reduction}")
    code_dim = input_dim // reduction
    adapter = CodeAdapter(code_dim)
    _seeded_adapter_reset(adapter, decoder_seed)
    return MultiSeedEncoderAdapterCSI(encoders, adapter, decoder)
