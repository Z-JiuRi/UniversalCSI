import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCodeResidualAdapter(nn.Module):
    def __init__(self, dim=512, lowrank_rank=0, mlp_hidden=0,
                 gate_lr_init=0.1, gate_mlp_init=0.1, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lowrank_rank = lowrank_rank
        self.mlp_hidden = mlp_hidden
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        if lowrank_rank > 0:
            self.lr_down = nn.Linear(dim, lowrank_rank, bias=False)
            self.lr_up = nn.Linear(lowrank_rank, dim, bias=False)
            self.gate_lr = nn.Parameter(torch.tensor(float(gate_lr_init)))
        else:
            self.lr_down = None
            self.lr_up = None
            self.register_buffer("gate_lr", torch.tensor(0.0))
        if mlp_hidden > 0:
            self.mlp_fc1 = nn.Linear(dim, mlp_hidden)
            self.mlp_fc2 = nn.Linear(mlp_hidden, dim)
            self.gate_mlp = nn.Parameter(torch.tensor(float(gate_mlp_init)))
        else:
            self.mlp_fc1 = None
            self.mlp_fc2 = None
            self.register_buffer("gate_mlp", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        if self.lr_down is not None:
            nn.init.kaiming_uniform_(self.lr_down.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lr_up.weight)
        if self.mlp_fc1 is not None:
            nn.init.xavier_uniform_(self.mlp_fc1.weight)
            nn.init.zeros_(self.mlp_fc1.bias)
            nn.init.zeros_(self.mlp_fc2.weight)
            nn.init.zeros_(self.mlp_fc2.bias)

    def forward(self, z0):
        u = self.dropout(self.norm(z0))
        delta = z0.new_zeros(z0.shape)
        if self.lr_down is not None:
            delta = delta + self.gate_lr * self.lr_up(self.lr_down(u))
        if self.mlp_fc1 is not None:
            delta = delta + self.gate_mlp * self.mlp_fc2(
                F.gelu(self.mlp_fc1(u)))
        return z0 + delta


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
                        dropout=0.0, fc_rank=None, ffn_rank=None,
                        fc_alpha=None, ffn_alpha=None):
    target = target.lower()
    if target not in ("fc", "ffn", "fc_ffn"):
        raise ValueError("target must be one of: fc, ffn, fc_ffn")

    injected = []
    if target in ("fc", "fc_ffn"):
        cur_rank = rank if fc_rank is None else fc_rank
        if cur_rank <= 0:
            pass
        else:
            cur_alpha = alpha if fc_alpha is None else fc_alpha
            if cur_alpha is None:
                cur_alpha = cur_rank
            base = decoder.fc_decoder
            device = base.weight.device
            dtype = base.weight.dtype
            decoder.fc_decoder = LoRALinear(
                decoder.fc_decoder,
                rank=cur_rank,
                alpha=cur_alpha,
                dropout=dropout).to(device=device, dtype=dtype)
            injected.append(f"fc_decoder:r{cur_rank}")

    if target in ("ffn", "fc_ffn"):
        cur_rank = rank if ffn_rank is None else ffn_rank
        if cur_rank <= 0:
            return injected
        cur_alpha = alpha if ffn_alpha is None else ffn_alpha
        if cur_alpha is None:
            cur_alpha = cur_rank
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
                        rank=cur_rank,
                        alpha=cur_alpha,
                        dropout=dropout).to(device=device, dtype=dtype))
                injected.append(f"{path}:r{cur_rank}")

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
