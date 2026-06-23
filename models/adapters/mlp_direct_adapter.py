import torch.nn as nn


class MLPDirectAdapter(nn.Module):
    """MLP adapter WITHOUT residual connection.

    Applies LayerNorm -> Linear -> GELU -> Linear.  No residual means
    the adapter needs to learn the full target transformation from
    scratch rather than starting from identity.

    W1 is Kaiming-initialized for healthy activation variance.
    W2 is initialized with small random values (std=0.01) so the
    initial adapter output is a near-zero perturbation — the decoder
    sees something close to the original code at step 0 and the
    adapter gradually learns the full transformation.
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
        nn.init.kaiming_uniform_(self.mlp[0].weight, a=0.3,
                                 nonlinearity='leaky_relu')
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        return self.mlp(self.norm(x))
