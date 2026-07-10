import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, residual_scale=0.1,
                 use_norm=True, learnable_gate=False, gate_max=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = residual_scale
        self.learnable_gate = learnable_gate
        self.gate_max = gate_max
        if learnable_gate:
            if gate_max <= 0:
                raise ValueError("gate_max must be positive when learnable_gate=True")
            init = min(max(residual_scale / gate_max, 1e-6), 1.0 - 1e-6)
            raw = torch.logit(torch.full((dim,), init))
            self.raw_gate = nn.Parameter(raw)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def gate(self):
        if not self.learnable_gate:
            return self.residual_scale
        return self.gate_max * torch.sigmoid(self.raw_gate)

    def forward(self, x):
        return x + self.gate() * self.net(self.norm(x))

    @torch.no_grad()
    def gate_metrics(self, prefix):
        if not self.learnable_gate:
            return {}
        gate = self.gate().detach().float().cpu()
        return {
            f"{prefix}/gate_mean": float(gate.mean()),
            f"{prefix}/gate_std": float(gate.std()),
            f"{prefix}/gate_min": float(gate.min()),
            f"{prefix}/gate_max": float(gate.max()),
        }


class FinalGate(nn.Module):
    def __init__(self, dim, mode="none", gate_max=1.0, gate_init=1.0,
                 adaptive_hidden=128):
        super().__init__()
        self.mode = mode
        self.gate_max = gate_max
        self.gate_init = gate_init
        self._last_gate = None
        if mode == "none":
            return
        if mode == "final_unbounded":
            self.gate = nn.Parameter(torch.full((dim,), float(gate_init)))
            return
        if gate_max <= 0:
            raise ValueError("gate_max must be positive")
        init = min(max(gate_init / gate_max, 1e-6), 1.0 - 1e-6)
        raw_init = torch.logit(torch.full((dim,), init))
        if mode == "final_static":
            self.raw_gate = nn.Parameter(raw_init)
        elif mode == "final_adaptive":
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, adaptive_hidden),
                nn.GELU(),
                nn.Linear(adaptive_hidden, dim),
            )
            nn.init.xavier_uniform_(self.net[1].weight)
            nn.init.zeros_(self.net[1].bias)
            nn.init.zeros_(self.net[3].weight)
            with torch.no_grad():
                self.net[3].bias.copy_(raw_init)
        else:
            raise ValueError(f"Unknown final gate mode: {mode}")

    def forward(self, z0, delta):
        if self.mode == "none":
            return z0 + delta
        if self.mode == "final_unbounded":
            gate = self.gate.view(1, -1)
            self._last_gate = gate
            return z0 + gate * delta
        if self.mode == "final_static":
            gate = self.gate_max * torch.sigmoid(self.raw_gate).view(1, -1)
        else:
            gate = self.gate_max * torch.sigmoid(self.net(z0))
        self._last_gate = gate
        return z0 + gate * delta

    def regularization(self):
        if self.mode == "none":
            return None
        if self.mode == "final_unbounded":
            return self.gate.abs().mean()
        if self.mode == "final_static":
            return (self.gate_max * torch.sigmoid(self.raw_gate)).mean()
        if self._last_gate is None:
            return None
        return self._last_gate.mean()

    @torch.no_grad()
    def metrics(self, z0=None):
        if self.mode == "none":
            return {}
        if self.mode == "final_unbounded":
            gate = self.gate.detach()
        if self.mode == "final_static":
            gate = self.gate_max * torch.sigmoid(self.raw_gate.detach())
        elif self.mode == "final_adaptive":
            if z0 is None:
                return {}
            gate = self.gate_max * torch.sigmoid(self.net(z0).detach())
        gate = gate.float().cpu()
        return {
            "adapter/final_gate_mean": float(gate.mean()),
            "adapter/final_gate_std": float(gate.std()),
            "adapter/final_gate_min": float(gate.min()),
            "adapter/final_gate_max": float(gate.max()),
        }


class LowRankResidualBlock(nn.Module):
    def __init__(self, dim, rank=64, dropout=0.0, residual_scale=0.1,
                 use_norm=True, learnable_gate=False, gate_max=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.down = nn.Linear(dim, rank)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(rank, dim)
        self.residual_scale = residual_scale
        self.learnable_gate = learnable_gate
        self.gate_max = gate_max
        if learnable_gate:
            if gate_max <= 0:
                raise ValueError("gate_max must be positive when learnable_gate=True")
            init = min(max(residual_scale / gate_max, 1e-6), 1.0 - 1e-6)
            self.raw_gate = nn.Parameter(torch.logit(torch.full((dim,), init)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def gate(self):
        if not self.learnable_gate:
            return self.residual_scale
        return self.gate_max * torch.sigmoid(self.raw_gate)

    def forward(self, x):
        residual = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + self.gate() * residual

    @torch.no_grad()
    def gate_metrics(self, prefix):
        if not self.learnable_gate:
            return {}
        gate = self.gate().detach().float().cpu()
        return {
            f"{prefix}/gate_mean": float(gate.mean()),
            f"{prefix}/gate_std": float(gate.std()),
            f"{prefix}/gate_min": float(gate.min()),
            f"{prefix}/gate_max": float(gate.max()),
        }


class AffineResidualMLPMapper(nn.Module):
    """Closed-form affine alignment followed by identity-initialized residual MLP.

    Data flow:
        source_code -> source @ W + b -> z0
        z0 -> residual blocks -> mapped_code

    W and b are registered as buffers by default, so training only updates the
    residual MLP. This keeps the model initialized exactly at the offline
    affine solution.
    """

    def __init__(self, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 learnable_residual_gate=False, gate_max=0.5,
                 gate_mode="block", final_gate_max=1.0,
                 final_gate_init=1.0, adaptive_gate_hidden=128):
        super().__init__()
        if weight.ndim != 2 or weight.size(0) != weight.size(1):
            raise ValueError(f"weight must be square 2D, got {tuple(weight.shape)}")
        if bias.ndim != 1 or bias.size(0) != weight.size(1):
            raise ValueError(
                f"bias must be ({weight.size(1)},), got {tuple(bias.shape)}")
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, hidden_dim, dropout, residual_scale,
                          use_norm=use_block_norm,
                          learnable_gate=learnable_residual_gate and gate_mode == "block",
                          gate_max=gate_max)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        if gate_mode == "block":
            final_mode = "none"
        elif gate_mode in ("none", "final_static", "final_adaptive",
                           "final_unbounded"):
            final_mode = gate_mode
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        self.final_gate = FinalGate(
            dim, mode=final_mode, gate_max=final_gate_max,
            gate_init=final_gate_init, adaptive_hidden=adaptive_gate_hidden)
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self._last_z0 = None

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = self.final_gate(z0, x - z0)
        with torch.no_grad():
            self._delta_ratio.fill_((x - z0).norm() / z0.norm().clamp_min(1e-8))
            self._last_z0 = z0.detach()
        return x

    def gate_regularization(self):
        return self.final_gate.regularization()

    @torch.no_grad()
    def get_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        return {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        } | self._gate_metrics() | self.final_gate.metrics(self._last_z0)

    @torch.no_grad()
    def _gate_metrics(self):
        values = {}
        means = []
        maxes = []
        for idx, block in enumerate(self.blocks):
            block_metrics = block.gate_metrics(f"adapter/block{idx}")
            values.update(block_metrics)
            if block_metrics:
                means.append(block_metrics[f"adapter/block{idx}/gate_mean"])
                maxes.append(block_metrics[f"adapter/block{idx}/gate_max"])
        if means:
            values["adapter/gate_mean_avg"] = float(sum(means) / len(means))
            values["adapter/gate_max_max"] = float(max(maxes))
        return values


class AffineLowRankResidualMapper(nn.Module):
    def __init__(self, weight, bias, rank=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, use_block_norm=True, use_final_norm=False,
                 train_affine=False, learnable_residual_gate=False,
                 gate_max=0.5):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            LowRankResidualBlock(
                dim, rank=rank, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm,
                learnable_gate=learnable_residual_gate,
                gate_max=gate_max)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.register_buffer("_delta_ratio", torch.tensor(0.0))

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        with torch.no_grad():
            self._delta_ratio.fill_((x - z0).norm() / z0.norm().clamp_min(1e-8))
        return x

    @torch.no_grad()
    def get_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        values = {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
        means = []
        maxes = []
        for idx, block in enumerate(self.blocks):
            metrics = block.gate_metrics(f"adapter/block{idx}")
            values.update(metrics)
            if metrics:
                means.append(metrics[f"adapter/block{idx}/gate_mean"])
                maxes.append(metrics[f"adapter/block{idx}/gate_max"])
        if means:
            values["adapter/gate_mean_avg"] = float(sum(means) / len(means))
            values["adapter/gate_max_max"] = float(max(maxes))
        return values


class AffineLinearStackMapper(nn.Module):
    """Affine alignment followed by identity-initialized linear stack.

    This is mainly a baseline. Without nonlinear activations, the whole mapper
    is still affine, so it should not improve over the closed-form affine fit.
    """

    def __init__(self, weight, bias, num_layers=1, train_affine=False):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, source):
        x = source.matmul(self.alignment_weight) + self.alignment_bias
        for layer in self.layers:
            x = layer(x)
        return x


class DirectMLPMapper(nn.Module):
    """Non-residual source-to-target MLP baseline."""

    def __init__(self, dim, hidden_dim=1024, num_layers=4, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, source):
        return self.net(source)


def build_mapper(mapper_type, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 learnable_residual_gate=False, gate_max=0.5,
                 lowrank_rank=64, gate_mode="block",
                 final_gate_max=1.0, final_gate_init=1.0,
                 adaptive_gate_hidden=128):
    mapper_type = mapper_type.lower()
    if mapper_type == "affine_residual_mlp":
        return AffineResidualMLPMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max, gate_mode=gate_mode,
            final_gate_max=final_gate_max,
            final_gate_init=final_gate_init,
            adaptive_gate_hidden=adaptive_gate_hidden)
    if mapper_type == "affine_lowrank_residual":
        return AffineLowRankResidualMapper(
            weight, bias, rank=lowrank_rank, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max)
    if mapper_type == "affine_linear":
        return AffineLinearStackMapper(
            weight, bias, num_layers=num_blocks, train_affine=train_affine)
    if mapper_type == "direct_mlp":
        return DirectMLPMapper(
            weight.size(0), hidden_dim=hidden_dim, num_layers=num_blocks,
            dropout=dropout)
    raise ValueError(f"Unknown mapper_type: {mapper_type}")
