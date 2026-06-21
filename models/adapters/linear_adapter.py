import torch.nn as nn


class LinearAdapter(nn.Module):
    def __init__(self, adapter_dim):
        super().__init__()
        self.norm = nn.LayerNorm(adapter_dim)
        self.proj = nn.Linear(adapter_dim, adapter_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, code):
        return code + self.proj(self.norm(code))
