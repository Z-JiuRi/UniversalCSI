import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if rank <= 0:
            raise ValueError("LoRA rank should be positive")

        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base_linear.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        self.reset_lora_parameters()

        for param in self.base.parameters():
            param.requires_grad = False

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return base_out + self.scaling * lora_out

    def lora_weight(self):
        return self.scaling * (self.lora_b @ self.lora_a)

    def metrics(self):
        with torch.no_grad():
            delta = self.lora_weight()
            base_norm = self.base.weight.norm()
            delta_norm = delta.norm()
            return {
                "rank": self.rank,
                "alpha": self.alpha,
                "scaling": self.scaling,
                "lora_a_norm": self.lora_a.norm().item(),
                "lora_b_norm": self.lora_b.norm().item(),
                "delta_norm": delta_norm.item(),
                "base_weight_norm": base_norm.item(),
                "delta_base_ratio": (delta_norm / base_norm.clamp_min(1e-12)).item(),
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
            }


def mark_only_lora_trainable(model):
    for param in model.parameters():
        param.requires_grad = False

    trainable = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad = True
            module.lora_b.requires_grad = True
            trainable += module.lora_a.numel() + module.lora_b.numel()
    if trainable == 0:
        raise ValueError("No LoRALinear modules found")
    return trainable


def apply_decoder_lora(model, component, rank=8, alpha=16):
    if component != "token_projection":
        raise ValueError("Only --lora_component token_projection is supported")
    if not hasattr(model.decoder, "token_projection"):
        raise ValueError("decoder has no token_projection; use decoder=hybrid")
    model.decoder.token_projection = LoRALinear(
        model.decoder.token_projection, rank=rank, alpha=alpha)
    trainable = mark_only_lora_trainable(model)
    if hasattr(model, "freeze_encoder"):
        model.freeze_encoder()
    else:
        model.encoder.eval()
    if hasattr(model, "decoder_is_frozen"):
        model.decoder_is_frozen = True
    model.decoder.eval()
    return trainable


def collect_lora_metrics(model):
    metrics = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for key, value in module.metrics().items():
                metrics[f"{name}.{key}"] = value
    return metrics
