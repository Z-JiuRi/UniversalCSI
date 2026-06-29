import torch
import torch.nn as nn


class LowRankAffineAdapter(nn.Module):
    """Low-rank residual affine code calibration.

    Scheme B:
        z_out = z + U V LN(z) + b

    U is zero-initialized, so the adapter starts as identity even though V
    is randomly initialized and ready to receive gradients.
    """

    def __init__(self, adapter_dim, rank=32):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.norm = nn.LayerNorm(adapter_dim)
        self.down = nn.Linear(adapter_dim, rank, bias=False)
        self.up = nn.Linear(rank, adapter_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(adapter_dim))
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_uniform_(self.down.weight, a=0.3,
                                 nonlinearity="leaky_relu")
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        delta = self.up(self.down(self.norm(x))) + self.bias
        with torch.no_grad():
            self._delta_ratio.fill_(
                delta.norm() / x.norm().clamp_min(1e-8))
        return x + delta

    @torch.no_grad()
    def get_metrics(self):
        return {
            "adapter/rank": float(self.rank),
            "adapter/down_norm": float(self.down.weight.norm().cpu()),
            "adapter/up_norm": float(self.up.weight.norm().cpu()),
            "adapter/bias_norm": float(self.bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
