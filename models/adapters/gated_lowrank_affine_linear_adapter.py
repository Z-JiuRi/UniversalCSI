import torch
import torch.nn as nn


class GatedLowRankAffineLinearAdapter(nn.Module):
    """Gated low-rank affine calibration plus linear residual refinement.

    Lightweight D variant:
        z1 = z + U V LN(z) + b
        z_out = z1 + gate * Linear(LN(z1))

    The linear residual is zero-initialized, so the adapter starts exactly as
    identity while using far fewer parameters than the two-layer MLP variant.
    """

    def __init__(self, adapter_dim, rank=32, gate_init=0.1):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.gate_init = gate_init
        self.lowrank_norm = nn.LayerNorm(adapter_dim)
        self.down = nn.Linear(adapter_dim, rank, bias=False)
        self.up = nn.Linear(rank, adapter_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(adapter_dim))
        self.linear_norm = nn.LayerNorm(adapter_dim)
        self.linear = nn.Linear(adapter_dim, adapter_dim)
        self.gate = nn.Parameter(torch.empty(adapter_dim))
        self.register_buffer("_gate_mean", torch.tensor(0.0))
        self.register_buffer("_lowrank_ratio", torch.tensor(0.0))
        self.register_buffer("_linear_ratio", torch.tensor(0.0))
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.lowrank_norm.weight)
        nn.init.zeros_(self.lowrank_norm.bias)
        nn.init.kaiming_uniform_(self.down.weight, a=0.3,
                                 nonlinearity="leaky_relu")
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.linear_norm.weight)
        nn.init.zeros_(self.linear_norm.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.constant_(self.gate, self.gate_init)

    def forward(self, x):
        lowrank_delta = self.up(self.down(self.lowrank_norm(x))) + self.bias
        z1 = x + lowrank_delta
        linear_delta = self.gate * self.linear(self.linear_norm(z1))
        out = z1 + linear_delta
        with torch.no_grad():
            denom = x.norm().clamp_min(1e-8)
            self._gate_mean.fill_(self.gate.mean())
            self._lowrank_ratio.fill_(lowrank_delta.norm() / denom)
            self._linear_ratio.fill_(linear_delta.norm() / denom)
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
            "adapter/linear_norm": float(self.linear.weight.norm().cpu()),
            "adapter/lowrank_ratio": float(self._lowrank_ratio.cpu()),
            "adapter/linear_ratio": float(self._linear_ratio.cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
