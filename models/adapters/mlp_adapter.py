import torch
import torch.nn as nn


class MLPAdapter(nn.Module):
    """MLP adapter with residual connection and zero-initialized output projection.

    Applies LayerNorm -> Linear -> GELU -> Linear, then adds the result
    back to the input via a residual connection.

    The first Linear (W1) is Kaiming-initialized to produce non-zero
    activations so gradients can flow.  The second Linear (W2) is
    zero-initialized so the adapter starts as an identity function.
    """

    def __init__(self, adapter_dim, adapter_hidden_dim=None):
        super().__init__()
        if adapter_hidden_dim is None:
            adapter_hidden_dim = 4 * adapter_dim
        self.norm = nn.LayerNorm(adapter_dim)
        self.mlp = nn.Sequential(
            nn.Linear(adapter_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, adapter_dim),
        )
        self.register_buffer("_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_uniform_(self.mlp[0].weight, a=0.3,
                                 nonlinearity='leaky_relu')
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        delta = self.mlp(self.norm(x))
        with torch.no_grad():
            self._ratio.fill_(delta.norm() / x.norm().clamp_min(1e-8))
        return x + delta

    @torch.no_grad()
    def get_metrics(self):
        """Return lightweight metrics about the adapter state."""
        w1 = self.mlp[0].weight
        w2 = self.mlp[2].weight
        return {
            "adapter/W1_norm": float(w1.norm().cpu()),
            "adapter/W1_mean": float(w1.mean().cpu()),
            "adapter/W1_std": float(w1.std().cpu()),
            "adapter/W2_norm": float(w2.norm().cpu()),
            "adapter/W2_mean": float(w2.mean().cpu()),
            "adapter/W2_std": float(w2.std().cpu()),
            "adapter/delta_ratio": float(self._ratio.cpu()),
        }
