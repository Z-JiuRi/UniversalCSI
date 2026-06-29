import torch
import torch.nn as nn


class DiagonalAffineAdapter(nn.Module):
    """Per-dimension affine code calibration.

    Scheme A:
        z_out = gamma * z + b

    The adapter starts as identity with gamma=1 and b=0.
    """

    def __init__(self, adapter_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(adapter_dim))
        self.bias = nn.Parameter(torch.zeros(adapter_dim))
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        out = x * self.gamma + self.bias
        with torch.no_grad():
            delta = out - x
            self._delta_ratio.fill_(
                delta.norm() / x.norm().clamp_min(1e-8))
        return out

    @torch.no_grad()
    def get_metrics(self):
        return {
            "adapter/gamma_mean": float(self.gamma.mean().cpu()),
            "adapter/gamma_std": float(self.gamma.std().cpu()),
            "adapter/bias_norm": float(self.bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
