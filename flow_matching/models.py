import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    if t.ndim != 1:
        t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half, 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class AlignmentLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("weight", torch.eye(dim))
        self.register_buffer("bias", torch.zeros(dim))

    @torch.no_grad()
    def set_transform(self, weight, bias):
        self.weight.copy_(weight.float())
        self.bias.copy_(bias.float())

    def forward(self, source):
        return source.matmul(self.weight) + self.bias


class VelocityBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
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
        return x + self.net(self.norm(x))


class VelocityMLP(nn.Module):
    def __init__(self, dim, hidden_dim=2048, num_blocks=4, time_dim=128,
                 condition="source_start", dropout=0.0):
        super().__init__()
        if condition not in ("source", "start", "source_start", "none"):
            raise ValueError(f"Unknown condition: {condition}")
        self.condition = condition
        self.time_dim = time_dim
        if condition == "source_start":
            cond_dim = dim * 2
        elif condition in ("source", "start"):
            cond_dim = dim
        else:
            cond_dim = 0
        self.x_proj = nn.Linear(dim, hidden_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim else None
        self.blocks = nn.ModuleList([
            VelocityBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.zeros_(self.x_proj.bias)
        if self.cond_proj is not None:
            nn.init.xavier_uniform_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        for module in self.t_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def build_condition(self, source, start):
        if self.condition == "source_start":
            return torch.cat([source, start], dim=-1)
        if self.condition == "source":
            return source
        if self.condition == "start":
            return start
        return None

    def forward(self, x, t, source, start):
        h = self.x_proj(x)
        h = h + self.t_proj(timestep_embedding(t, self.time_dim))
        cond = self.build_condition(source, start)
        if cond is not None:
            h = h + self.cond_proj(cond)
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h))


class FlowMatchingTranslator(nn.Module):
    def __init__(self, dim, hidden_dim=2048, num_blocks=4, time_dim=128,
                 condition="source_start", dropout=0.0):
        super().__init__()
        self.alignment = AlignmentLayer(dim)
        self.velocity = VelocityMLP(
            dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            time_dim=time_dim,
            condition=condition,
            dropout=dropout)

    def start(self, source):
        return self.alignment(source)

    def velocity_at(self, x, t, source):
        start = self.start(source)
        return self.velocity(x, t, source, start)

    @torch.no_grad()
    def sample(self, source, steps=16, method="euler"):
        if steps < 1:
            raise ValueError("steps must be positive")
        method = method.lower()
        x = self.start(source)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((source.size(0),), i * dt, device=source.device)
            v = self.velocity_at(x, t, source)
            if method == "euler":
                x = x + dt * v
            elif method == "heun":
                x_pred = x + dt * v
                t_next = torch.full(
                    (source.size(0),),
                    min((i + 1) * dt, 1.0),
                    device=source.device)
                v_next = self.velocity_at(x_pred, t_next, source)
                x = x + 0.5 * dt * (v + v_next)
            else:
                raise ValueError(f"Unknown ODE method: {method}")
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

