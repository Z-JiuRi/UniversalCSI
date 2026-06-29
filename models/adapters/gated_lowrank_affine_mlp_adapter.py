import torch
import torch.nn as nn


class GatedLowRankAffineMLPAdapter(nn.Module):
    """Gated low-rank affine calibration plus residual MLP refinement.

    Scheme D:
        z1 = z + U V LN(z) + b
        z_out = z1 + gate * MLP(LN(z1))

    gate is a learnable per-dimension vector.  It is initialized to a small
    value, while the MLP output projection is zero-initialized; therefore the
    adapter starts exactly as identity.
    """

    def __init__(self, adapter_dim, adapter_hidden_dim=None, rank=32,
                 gate_init=0.1):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if adapter_hidden_dim is None:
            adapter_hidden_dim = 4 * adapter_dim
        self.rank = rank
        self.gate_init = gate_init
        self.lowrank_norm = nn.LayerNorm(adapter_dim)
        self.down = nn.Linear(adapter_dim, rank, bias=False)
        self.up = nn.Linear(rank, adapter_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(adapter_dim))
        self.mlp_norm = nn.LayerNorm(adapter_dim)
        self.mlp = nn.Sequential(
            nn.Linear(adapter_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, adapter_dim),
        )
        self.gate = nn.Parameter(torch.empty(adapter_dim))
        self.register_buffer("_gate_mean", torch.tensor(0.0))
        self.register_buffer("_lowrank_ratio", torch.tensor(0.0))
        self.register_buffer("_mlp_ratio", torch.tensor(0.0))
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.lowrank_norm.weight)
        nn.init.zeros_(self.lowrank_norm.bias)
        nn.init.kaiming_uniform_(self.down.weight, a=0.3,
                                 nonlinearity="leaky_relu")
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.mlp_norm.weight)
        nn.init.zeros_(self.mlp_norm.bias)
        nn.init.kaiming_uniform_(self.mlp[0].weight, a=0.3,
                                 nonlinearity="leaky_relu")
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        nn.init.constant_(self.gate, self.gate_init)

    def forward(self, x):
        lowrank_delta = self.up(self.down(self.lowrank_norm(x))) + self.bias
        z1 = x + lowrank_delta
        raw_mlp_delta = self.mlp(self.mlp_norm(z1))
        mlp_delta = self.gate * raw_mlp_delta
        out = z1 + mlp_delta
        with torch.no_grad():
            denom = x.norm().clamp_min(1e-8)
            self._gate_mean.fill_(self.gate.mean())
            self._lowrank_ratio.fill_(lowrank_delta.norm() / denom)
            self._mlp_ratio.fill_(mlp_delta.norm() / denom)
            self._delta_ratio.fill_((out - x).norm() / denom)
        return out

    @torch.no_grad()
    def get_metrics(self):
        return {
            "adapter/rank": float(self.rank),
            "adapter/gate_mean": float(self._gate_mean.cpu()),
            "adapter/gate_std": float(self.gate.std().cpu()),
            "adapter/down_norm": float(self.down.weight.norm().cpu()),
            "adapter/up_norm": float(self.up.weight.norm().cpu()),
            "adapter/bias_norm": float(self.bias.norm().cpu()),
            "adapter/W1_norm": float(self.mlp[0].weight.norm().cpu()),
            "adapter/W2_norm": float(self.mlp[2].weight.norm().cpu()),
            "adapter/lowrank_ratio": float(self._lowrank_ratio.cpu()),
            "adapter/mlp_ratio": float(self._mlp_ratio.cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
