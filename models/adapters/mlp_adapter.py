import torch.nn as nn


class MLPAdapter(nn.Module):
    """MLP adapter with residual connection and zero-initialization.

    Applies LayerNorm -> Linear -> GELU -> Linear, then adds the result
    back to the input via a residual connection.  Zero-initializing both
    Linear layers means the adapter starts as an identity function, so
    training begins from the frozen model's exact behaviour.
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
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        return x + self.mlp(self.norm(x))
