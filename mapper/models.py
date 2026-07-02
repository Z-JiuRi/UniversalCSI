import math

import torch
import torch.nn as nn


class IdentityMapper(nn.Module):
    def forward(self, x):
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, residual_scale=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        return x + self.residual_scale * self.net(self.norm(x))


class DeepResidualMLPMapper(nn.Module):
    def __init__(self, dim, hidden_dim=2048, num_blocks=4, dropout=0.0,
                 residual_scale=1.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(dim, hidden_dim, dropout, residual_scale)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class AffineCouplingBlock(nn.Module):
    def __init__(self, dim, hidden_dim=1024, clamp=0.1, swap=False,
                 dropout=0.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("AffineCouplingBlock requires an even code dim")
        self.half = dim // 2
        self.clamp = clamp
        self.swap = swap
        self.norm = nn.LayerNorm(self.half)
        self.net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.half * 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        if self.swap:
            a, b = b, a
        st = self.net(self.norm(a))
        s, t = st.chunk(2, dim=-1)
        s = self.clamp * torch.tanh(s)
        b = b * torch.exp(s) + t
        if self.swap:
            a, b = b, a
        return torch.cat([a, b], dim=-1)


class FlowMapper(nn.Module):
    def __init__(self, dim, hidden_dim=1024, num_blocks=8, clamp=0.1,
                 dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            AffineCouplingBlock(
                dim,
                hidden_dim=hidden_dim,
                clamp=clamp,
                swap=bool(i % 2),
                dropout=dropout)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class HiddenResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        return x + self.net(self.norm(x))


class DeltaMLPMapper(nn.Module):
    def __init__(self, dim, input_dim=None, hidden_dim=2048, num_blocks=4,
                 dropout=0.0):
        super().__init__()
        input_dim = input_dim or dim
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            HiddenResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


class HybridFlowMLPMapper(nn.Module):
    def __init__(self, dim, hidden_dim=2048, flow_hidden_dim=1024,
                 num_flow_blocks=6, num_mlp_blocks=2, clamp=0.1,
                 dropout=0.0):
        super().__init__()
        self.flow = FlowMapper(
            dim,
            hidden_dim=flow_hidden_dim,
            num_blocks=num_flow_blocks,
            clamp=clamp,
            dropout=dropout)
        self.mlp = DeepResidualMLPMapper(
            dim,
            hidden_dim=hidden_dim,
            num_blocks=num_mlp_blocks,
            dropout=dropout)

    def forward(self, x):
        return self.mlp(self.flow(x))


def build_mapper(name, dim, hidden_dim=2048, num_blocks=4,
                 flow_hidden_dim=1024, flow_blocks=8, clamp=0.1,
                 dropout=0.0, input_dim=None):
    name = name.lower()
    if name == "identity":
        return IdentityMapper()
    if name in ("delta_mlp", "residual_delta_mlp", "affine_residual_mlp"):
        return DeltaMLPMapper(
            dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout)
    if name in ("mlp", "deep_mlp", "residual_mlp"):
        return DeepResidualMLPMapper(
            dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            dropout=dropout)
    if name in ("flow", "coupling_flow"):
        return FlowMapper(
            dim,
            hidden_dim=flow_hidden_dim,
            num_blocks=flow_blocks,
            clamp=clamp,
            dropout=dropout)
    if name in ("hybrid", "hybrid_flow_mlp"):
        return HybridFlowMLPMapper(
            dim,
            hidden_dim=hidden_dim,
            flow_hidden_dim=flow_hidden_dim,
            num_flow_blocks=flow_blocks,
            num_mlp_blocks=num_blocks,
            clamp=clamp,
            dropout=dropout)
    raise ValueError(f"Unknown mapper: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
