import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=None, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base module")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.reset_parameters()
        for param in self.base.parameters():
            param.requires_grad_(False)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return (
            self.base(x)
            + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
        )


def _set_submodule(root, path, module):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def inject_decoder_lora(decoder, target="fc_ffn", rank=8, alpha=None,
                        dropout=0.0):
    target = target.lower()
    if target not in ("fc", "ffn", "fc_ffn"):
        raise ValueError("target must be one of: fc, ffn, fc_ffn")

    injected = []
    if target in ("fc", "fc_ffn"):
        base = decoder.fc_decoder
        device = base.weight.device
        dtype = base.weight.dtype
        decoder.fc_decoder = LoRALinear(
            decoder.fc_decoder,
            rank=rank,
            alpha=alpha,
            dropout=dropout).to(device=device, dtype=dtype)
        injected.append("fc_decoder")

    if target in ("ffn", "fc_ffn"):
        for idx, layer in enumerate(decoder.decoder.layers):
            for name in ("linear1", "linear2"):
                path = f"decoder.layers.{idx}.{name}"
                base = getattr(layer, name)
                device = base.weight.device
                dtype = base.weight.dtype
                _set_submodule(
                    decoder,
                    path,
                    LoRALinear(
                        base,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout).to(device=device, dtype=dtype))
                injected.append(path)

    return injected


def mark_only_lora_trainable(model):
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters()
               if param.requires_grad)


def count_lora_parameters(model):
    return sum(param.numel() for name, param in model.named_parameters()
               if "lora_" in name)
