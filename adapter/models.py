import torch
import torch.nn as nn
import torch.nn.functional as F


def init_zero_linear(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_identity_projection(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    with torch.no_grad():
        diag = min(linear.weight.size(0), linear.weight.size(1))
        linear.weight[:diag, :diag].copy_(torch.eye(
            diag, dtype=linear.weight.dtype, device=linear.weight.device))


def split_groups(dim, num_groups):
    if dim % num_groups != 0:
        raise ValueError(f"dim={dim} must be divisible by num_groups={num_groups}")
    return num_groups, dim // num_groups


def build_activation(name):
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in ("identity", "none", "linear"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class AffineStartMixin:
    def _init_affine(self, weight, bias, train_affine):
        if weight.ndim != 2 or weight.size(0) != weight.size(1):
            raise ValueError(f"weight must be square 2D, got {tuple(weight.shape)}")
        if bias.ndim != 1 or bias.size(0) != weight.size(1):
            raise ValueError(
                f"bias must be ({weight.size(1)},), got {tuple(bias.shape)}")
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.register_buffer("_delta_ratio", torch.tensor(0.0))

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def _update_delta_ratio(self, z0, out):
        with torch.no_grad():
            self._delta_ratio.fill_(
                (out - z0).norm() / z0.norm().clamp_min(1e-8))

    @torch.no_grad()
    def _base_metrics(self):
        trainable_affine = isinstance(self.alignment_weight, nn.Parameter)
        return {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }


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


class FiLMResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, residual_scale=0.1,
                 use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        residual = self.net(self.norm(x))
        residual = residual * self.gamma.view(1, -1) + self.beta.view(1, -1)
        return x + self.residual_scale * residual

    @torch.no_grad()
    def film_metrics(self, prefix):
        gamma = self.gamma.detach().float().cpu()
        beta = self.beta.detach().float().cpu()
        return {
            f"{prefix}/gamma_mean": float(gamma.mean()),
            f"{prefix}/gamma_std": float(gamma.std(unbiased=False)),
            f"{prefix}/gamma_min": float(gamma.min()),
            f"{prefix}/gamma_max": float(gamma.max()),
            f"{prefix}/beta_mean": float(beta.mean()),
            f"{prefix}/beta_std": float(beta.std(unbiased=False)),
            f"{prefix}/beta_norm": float(beta.norm()),
        }


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, bottleneck_dim=128, dropout=0.0,
                 residual_scale=0.1, use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.full = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.lowrank = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.full[0].weight)
        nn.init.zeros_(self.full[0].bias)
        nn.init.zeros_(self.full[3].weight)
        nn.init.zeros_(self.full[3].bias)
        nn.init.xavier_uniform_(self.lowrank[0].weight)
        nn.init.zeros_(self.lowrank[0].bias)
        nn.init.zeros_(self.lowrank[3].weight)
        nn.init.zeros_(self.lowrank[3].bias)

    def forward(self, x):
        h = self.norm(x)
        return x + self.residual_scale * (self.full(h) + self.lowrank(h))


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


class CodewordDeltaSelfAttention(nn.Module):
    """Self-attention over scalar codeword positions.

    Each position remains one token. The token feature is built from the
    affine-aligned value, the residual MLP proposed change, or both. This
    models sample-dependent relations between code dimensions without a
    Transformer feed-forward block.
    """

    def __init__(self, code_dim, attention_dim=32, num_heads=4,
                 dropout=0.0, input_mode="value_delta",
                 use_position=True):
        super().__init__()
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if num_heads <= 0 or attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim={attention_dim} must be divisible by "
                f"num_heads={num_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("attention dropout must be in [0, 1)")
        if input_mode not in ("value", "delta", "value_delta"):
            raise ValueError(f"Unknown attention input mode: {input_mode}")

        input_dim = 2 if input_mode == "value_delta" else 1
        self.code_dim = code_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.dropout = dropout
        self.input_mode = input_mode
        self.use_position = use_position

        self.value_norm = nn.LayerNorm(code_dim)
        self.delta_norm = nn.LayerNorm(code_dim)
        self.input_proj = nn.Linear(input_dim, attention_dim)
        if use_position:
            self.position_embedding = nn.Parameter(
                torch.empty(1, code_dim, attention_dim))
        else:
            self.register_buffer(
                "position_embedding",
                torch.zeros(1, code_dim, attention_dim),
                persistent=False)
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.qkv = nn.Linear(attention_dim, 3 * attention_dim)
        self.context_proj = nn.Linear(attention_dim, attention_dim)
        self.output_norm = nn.LayerNorm(attention_dim)
        self.output_proj = nn.Linear(attention_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        if self.use_position:
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        init_zero_linear(self.output_proj)

    def _build_tokens(self, value, delta):
        value = self.value_norm(value)
        delta = self.delta_norm(delta)
        if self.input_mode == "value":
            token_input = value.unsqueeze(-1)
        elif self.input_mode == "delta":
            token_input = delta.unsqueeze(-1)
        else:
            token_input = torch.stack((value, delta), dim=-1)
        return self.input_proj(token_input) + self.position_embedding

    def forward(self, value, delta):
        tokens = self._build_tokens(value, delta)
        batch_size, num_tokens, _ = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value_features = qkv.unbind(0)
        context = F.scaled_dot_product_attention(
            query,
            key,
            value_features,
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(
            batch_size, num_tokens, self.attention_dim)
        context = self.context_proj(context)
        return self.output_proj(self.output_norm(context)).squeeze(-1)


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


class AffineResidualMLPAttentionMapper(AffineResidualMLPMapper):
    """Residual MLP followed by a delta-aware codeword attention correction."""

    def __init__(self, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 learnable_residual_gate=False, gate_max=0.5,
                 gate_mode="block", final_gate_max=1.0,
                 final_gate_init=1.0, adaptive_gate_hidden=128,
                 attention_dim=32, attention_heads=4,
                 attention_dropout=0.0, attention_scale=0.1,
                 attention_input="value_delta",
                 attention_use_position=True):
        super().__init__(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max, gate_mode=gate_mode,
            final_gate_max=final_gate_max,
            final_gate_init=final_gate_init,
            adaptive_gate_hidden=adaptive_gate_hidden)
        if attention_scale < 0:
            raise ValueError("attention_scale must be non-negative")
        self.code_attention = CodewordDeltaSelfAttention(
            weight.size(0), attention_dim=attention_dim,
            num_heads=attention_heads, dropout=attention_dropout,
            input_mode=attention_input,
            use_position=attention_use_position)
        self.attention_scale = attention_scale
        self.attention_input = attention_input
        self.register_buffer("_mlp_delta_ratio", torch.tensor(0.0))
        self.register_buffer("_attention_delta_ratio", torch.tensor(0.0))

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)

        mlp_delta = x - z0
        attention_delta = self.attention_scale * self.code_attention(
            z0, mlp_delta)
        x = x + attention_delta
        x = self.final_norm(x)
        x = self.final_gate(z0, x - z0)
        with torch.no_grad():
            z0_norm = z0.norm().clamp_min(1e-8)
            self._mlp_delta_ratio.fill_(mlp_delta.norm() / z0_norm)
            self._attention_delta_ratio.fill_(
                attention_delta.norm() / z0_norm)
            self._delta_ratio.fill_((x - z0).norm() / z0_norm)
            self._last_z0 = z0.detach()
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = super().get_metrics()
        values.update({
            "adapter/mlp_delta_ratio": float(self._mlp_delta_ratio.cpu()),
            "adapter/attention_delta_ratio": float(
                self._attention_delta_ratio.cpu()),
            "adapter/attention_dim": float(
                self.code_attention.attention_dim),
            "adapter/attention_heads": float(
                self.code_attention.num_heads),
            "adapter/attention_scale": float(self.attention_scale),
            "adapter/attention_use_position": float(
                self.code_attention.use_position),
        })
        return values


class AffineFiLMResidualMLPMapper(nn.Module):
    """Affine alignment plus residual MLP blocks with static FiLM modulation."""

    def __init__(self, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(
                dim, hidden_dim, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm)
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
        for idx, block in enumerate(self.blocks):
            values.update(block.film_metrics(f"adapter/block{idx}"))
        return values


class AffineMultiScaleResidualMLPMapper(nn.Module):
    """Affine alignment plus full+bottleneck residual blocks."""

    def __init__(self, weight, bias, hidden_dim=1024, bottleneck_dim=128,
                 num_blocks=4, dropout=0.0, residual_scale=0.1,
                 use_block_norm=True, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        dim = weight.size(0)
        if train_affine:
            self.alignment_weight = nn.Parameter(weight.clone())
            self.alignment_bias = nn.Parameter(bias.clone())
        else:
            self.register_buffer("alignment_weight", weight.clone())
            self.register_buffer("alignment_bias", bias.clone())
        self.blocks = nn.ModuleList([
            MultiScaleResidualBlock(
                dim, hidden_dim, bottleneck_dim=bottleneck_dim,
                dropout=dropout, residual_scale=residual_scale,
                use_norm=use_block_norm)
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
        return {
            "adapter/train_affine": float(trainable_affine),
            "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
            "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }


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


class BottleneckResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.0,
                 residual_scale=0.1, use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        init_zero_linear(self.up)

    def forward(self, x):
        delta = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + self.residual_scale * delta


class AffineBottleneckResidualMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, bottleneck_dim=128, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.blocks = nn.ModuleList([
            BottleneckResidualBlock(
                dim, bottleneck_dim=bottleneck_dim, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.bottleneck_dim = bottleneck_dim

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/bottleneck_dim"] = float(self.bottleneck_dim)
        return values


class GroupGatedResidualBlock(nn.Module):
    def __init__(self, dim, num_groups=16, group_hidden=64, dropout=0.0,
                 residual_scale=0.1, gate_init=0.5, use_norm=True,
                 gate_hidden=64):
        super().__init__()
        self.num_groups, self.group_dim = split_groups(dim, num_groups)
        self.residual_scale = residual_scale
        self.gate_init = gate_init
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.value = nn.Sequential(
            nn.Linear(self.group_dim, group_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(group_hidden, self.group_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_groups),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.value[0].weight)
        nn.init.zeros_(self.value[0].bias)
        init_zero_linear(self.value[3])
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        init_zero_linear(self.gate[2])
        init = min(max(self.gate_init, 1e-6), 1.0 - 1e-6)
        with torch.no_grad():
            self.gate[2].bias.fill_(float(torch.logit(torch.tensor(init))))

    def forward(self, x):
        n = x.size(0)
        x_norm = self.norm(x)
        grouped = x_norm.view(n, self.num_groups, self.group_dim)
        delta = self.value(grouped).view(n, -1)
        gate = torch.sigmoid(self.gate(x_norm)).view(n, self.num_groups, 1)
        delta = delta.view(n, self.num_groups, self.group_dim) * gate
        self._last_gate = gate.detach()
        return x + self.residual_scale * delta.view(n, -1)

    @torch.no_grad()
    def gate_metrics(self, prefix):
        if not hasattr(self, "_last_gate"):
            return {}
        gate = self._last_gate.float().cpu()
        return {
            f"{prefix}/gate_mean": float(gate.mean()),
            f"{prefix}/gate_std": float(gate.std()),
            f"{prefix}/gate_min": float(gate.min()),
            f"{prefix}/gate_max": float(gate.max()),
        }


class AffineGroupGatedMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, num_groups=16, group_hidden=64,
                 gate_hidden=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, gate_init=0.5, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_groups = num_groups
        self.blocks = nn.ModuleList([
            GroupGatedResidualBlock(
                dim, num_groups=num_groups, group_hidden=group_hidden,
                gate_hidden=gate_hidden, dropout=dropout,
                residual_scale=residual_scale, gate_init=gate_init,
                use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_groups"] = float(self.num_groups)
        means = []
        for idx, block in enumerate(self.blocks):
            metrics = block.gate_metrics(f"adapter/block{idx}")
            values.update(metrics)
            if metrics:
                means.append(metrics[f"adapter/block{idx}/gate_mean"])
        if means:
            values["adapter/group_gate_mean_avg"] = float(sum(means) / len(means))
        return values


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, token_dim, token_hidden=64,
                 channel_hidden=64, dropout=0.0, residual_scale=0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, num_tokens),
        )
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, token_dim),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.token_mlp[0].weight)
        nn.init.zeros_(self.token_mlp[0].bias)
        init_zero_linear(self.token_mlp[3])
        nn.init.xavier_uniform_(self.channel_mlp[0].weight)
        nn.init.zeros_(self.channel_mlp[0].bias)
        init_zero_linear(self.channel_mlp[3])

    def forward(self, x):
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + self.residual_scale * y
        y = self.channel_mlp(self.channel_norm(x))
        return x + self.residual_scale * y


class AffineTokenMixerMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, num_tokens=16, token_hidden=64,
                 channel_hidden=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_tokens, self.token_dim = split_groups(dim, num_tokens)
        self.blocks = nn.ModuleList([
            MixerBlock(
                self.num_tokens, self.token_dim, token_hidden=token_hidden,
                channel_hidden=channel_hidden, dropout=dropout,
                residual_scale=residual_scale)
            for _ in range(num_blocks)
        ])
        self.final_norm = (
            nn.LayerNorm(self.token_dim) if use_final_norm else nn.Identity())

    def forward(self, source):
        z0 = self.start(source)
        x = z0.view(z0.size(0), self.num_tokens, self.token_dim)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x).reshape(z0.size(0), -1)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_tokens"] = float(self.num_tokens)
        values["adapter/token_dim"] = float(self.token_dim)
        return values


class AffineTinyTransformerMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, num_tokens=16, num_heads=2,
                 transformer_ffn_dim=128, num_blocks=2, dropout=0.0,
                 residual_scale=0.1, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_tokens, self.token_dim = split_groups(dim, num_tokens)
        if self.token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim={self.token_dim} must be divisible by heads={num_heads}")
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True)
        self.norm = nn.LayerNorm(self.token_dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.out = nn.Linear(dim, dim)
        self.residual_scale = residual_scale
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        init_zero_linear(self.out)

    def forward(self, source):
        z0 = self.start(source)
        tokens = z0.view(z0.size(0), self.num_tokens, self.token_dim)
        features = self.encoder(self.norm(tokens)).reshape(z0.size(0), -1)
        x = z0 + self.residual_scale * self.out(features)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_tokens"] = float(self.num_tokens)
        values["adapter/token_dim"] = float(self.token_dim)
        return values


class ExpertBottleneck(nn.Module):
    def __init__(self, dim, bottleneck_dim=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        init_zero_linear(self.net[3])

    def forward(self, x):
        return self.net(x)


class MoEBottleneckBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim=64, num_experts=4,
                 gate_hidden=64, dropout=0.0, residual_scale=0.1,
                 use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.experts = nn.ModuleList([
            ExpertBottleneck(dim, bottleneck_dim=bottleneck_dim, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.zeros_(self.router[0].bias)
        nn.init.zeros_(self.router[2].weight)
        nn.init.zeros_(self.router[2].bias)

    def forward(self, x):
        x_norm = self.norm(x)
        weights = torch.softmax(self.router(x_norm), dim=-1)
        expert_outputs = torch.stack(
            [expert(x_norm) for expert in self.experts], dim=1)
        delta = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
        self._last_router = weights.detach()
        return x + self.residual_scale * delta

    @torch.no_grad()
    def router_metrics(self, prefix):
        if not hasattr(self, "_last_router"):
            return {}
        router = self._last_router.float().cpu()
        entropy = -(router * router.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return {
            f"{prefix}/router_entropy": float(entropy),
            f"{prefix}/router_max_prob": float(router.max(dim=-1).values.mean()),
        }


class AffineMoEBottleneckMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, bottleneck_dim=64, num_experts=4,
                 gate_hidden=64, num_blocks=4, dropout=0.0,
                 residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_experts = num_experts
        self.blocks = nn.ModuleList([
            MoEBottleneckBlock(
                dim, bottleneck_dim=bottleneck_dim, num_experts=num_experts,
                gate_hidden=gate_hidden, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_experts"] = float(self.num_experts)
        entropies = []
        for idx, block in enumerate(self.blocks):
            metrics = block.router_metrics(f"adapter/block{idx}")
            values.update(metrics)
            if metrics:
                entropies.append(metrics[f"adapter/block{idx}/router_entropy"])
        if entropies:
            values["adapter/router_entropy_avg"] = float(
                sum(entropies) / len(entropies))
        return values


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=128, mask_even=True, scale=0.1):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("AffineCouplingLayer requires an even dim")
        self.dim = dim
        self.half = dim // 2
        self.mask_even = mask_even
        self.scale = scale
        self.net = nn.Sequential(
            nn.LayerNorm(self.half),
            nn.Linear(self.half, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.half * 2),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)
        init_zero_linear(self.net[3])

    def forward(self, x):
        a, b = x[:, :self.half], x[:, self.half:]
        if self.mask_even:
            cond, target = a, b
        else:
            cond, target = b, a
        log_s, t = self.net(cond).chunk(2, dim=-1)
        log_s = self.scale * torch.tanh(log_s)
        target = target * torch.exp(log_s) + t
        if self.mask_even:
            return torch.cat([cond, target], dim=-1)
        return torch.cat([target, cond], dim=-1)


class AffineCouplingFlowMapper(nn.Module, AffineStartMixin):
    def __init__(self, weight, bias, flow_hidden_dim=128, num_blocks=4,
                 residual_scale=0.1, use_final_norm=False,
                 train_affine=False):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.layers = nn.ModuleList([
            AffineCouplingLayer(
                dim, hidden_dim=flow_hidden_dim, mask_even=(idx % 2 == 0),
                scale=residual_scale)
            for idx in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        return self._base_metrics()


class AffineIterativeResidualMapper(nn.Module, AffineStartMixin):
    """Affine start + multi-step residual refine (shared or unshared).

    Motivated by per-sample latent refinement through a frozen decoder: a
    small residual network applied repeatedly can approximate learned gradient
    steps in code space while keeping parameter count well below the full AE.
    """

    def __init__(self, weight, bias, hidden_dim=512, num_iters=8,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 share_weights=True, use_step_embed=True):
        super().__init__()
        if num_iters < 1:
            raise ValueError(f"num_iters must be >= 1, got {num_iters}")
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.num_iters = int(num_iters)
        self.share_weights = bool(share_weights)
        self.use_step_embed = bool(use_step_embed)
        self.residual_scale = residual_scale
        self.hidden_dim = hidden_dim
        if self.use_step_embed:
            self.step_embed = nn.Embedding(self.num_iters, dim)
            nn.init.zeros_(self.step_embed.weight)
        else:
            self.step_embed = None
        if self.share_weights:
            self.block = ResidualBlock(
                dim, hidden_dim, dropout=dropout,
                residual_scale=residual_scale, use_norm=use_block_norm,
                learnable_gate=False)
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList([
                ResidualBlock(
                    dim, hidden_dim, dropout=dropout,
                    residual_scale=residual_scale, use_norm=use_block_norm,
                    learnable_gate=False)
                for _ in range(self.num_iters)
            ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for step in range(self.num_iters):
            if self.step_embed is not None:
                x = x + self.step_embed.weight[step].view(1, -1)
            if self.share_weights:
                x = self.block(x)
            else:
                x = self.blocks[step](x)
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/num_iters"] = float(self.num_iters)
        values["adapter/share_weights"] = float(self.share_weights)
        values["adapter/use_step_embed"] = float(self.use_step_embed)
        values["adapter/hidden_dim"] = float(self.hidden_dim)
        return values


class AffineSensWeightedResidualMapper(nn.Module, AffineStartMixin):
    """Affine + residual MLP with a learnable diagonal sensitivity gate on delta.

    Projects residual updates through a soft non-negative diagonal so the model
    can emphasize decoder-sensitive code dimensions without unfreezing decoder.
    """

    def __init__(self, weight, bias, hidden_dim=512, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 sens_init=1.0):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                dim, hidden_dim, dropout=dropout,
                residual_scale=1.0, use_norm=use_block_norm,
                learnable_gate=False)
            for _ in range(num_blocks)
        ])
        # softplus(raw) keeps gate > 0; init near sens_init
        init = max(float(sens_init), 1e-4)
        raw = torch.log(torch.expm1(torch.tensor(init)))
        self.raw_sens = nn.Parameter(torch.full((dim,), float(raw)))
        self.residual_scale = residual_scale
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def sensitivity(self):
        return F.softplus(self.raw_sens) + 1e-4

    def forward(self, source):
        z0 = self.start(source)
        x = z0
        for block in self.blocks:
            x = block(x)
        delta = x - z0
        sens = self.sensitivity().view(1, -1)
        # re-center sensitivity to mean 1 so scale is controlled by residual_scale
        sens = sens / sens.mean().clamp_min(1e-6)
        out = z0 + self.residual_scale * sens * delta
        out = self.final_norm(out)
        self._update_delta_ratio(z0, out)
        return out

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        sens = self.sensitivity().detach().float().cpu()
        values["adapter/sens_mean"] = float(sens.mean())
        values["adapter/sens_std"] = float(sens.std(unbiased=False))
        values["adapter/sens_min"] = float(sens.min())
        values["adapter/sens_max"] = float(sens.max())
        return values


class AffineWholeResidualMLPMapper(nn.Module, AffineStartMixin):
    """Affine alignment plus one identity-initialized whole residual MLP."""

    def __init__(self, weight, bias, hidden_dims=None, dropout=0.0,
                 residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 activation="gelu"):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        hidden_dims = list(hidden_dims or [512, 512])
        layers = []
        in_dim = dim
        self.norm = nn.LayerNorm(dim) if use_block_norm else nn.Identity()
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, int(hidden_dim)),
                build_activation(activation),
                nn.Dropout(dropout),
            ])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.residual_scale = residual_scale
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.hidden_dims = hidden_dims
        self.reset_parameters()

    def reset_parameters(self):
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        init_zero_linear(linear_layers[-1])

    def forward(self, source):
        z0 = self.start(source)
        delta = self.net(self.norm(z0))
        x = z0 + self.residual_scale * delta
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/whole_mlp_depth"] = float(len(self.hidden_dims) + 1)
        return values


class LegacyMLPAdapterMapper(nn.Module):
    """Exact single-wide residual MLP used by the historical CodeAdapter.

    The residual branch is deliberately kept identical to
    ``models.adapters.MLPAdapter``:

        LayerNorm -> Linear(dim, hidden) -> GELU -> Linear(hidden, dim)
        output = input + residual_scale * branch(input)

    The second linear layer is zero-initialized, so this starts as identity.
    ``use_affine_alignment`` optionally prepends the offline least-squares
    alignment.  This lets one experiment isolate no alignment, frozen affine
    buffer, and trainable affine without changing the MLP itself.
    """

    def __init__(self, weight, bias, hidden_dim=2048, dropout=0.0,
                 residual_scale=1.0, use_affine_alignment=False,
                 train_affine=False):
        super().__init__()
        if weight.ndim != 2 or weight.size(0) != weight.size(1):
            raise ValueError(f"weight must be square 2D, got {tuple(weight.shape)}")
        if bias.ndim != 1 or bias.size(0) != weight.size(1):
            raise ValueError(
                f"bias must be ({weight.size(1)},), got {tuple(bias.shape)}")
        if train_affine and not use_affine_alignment:
            raise ValueError("train_affine requires use_affine_alignment=True")

        dim = weight.size(0)
        self.use_affine_alignment = bool(use_affine_alignment)
        self.residual_scale = float(residual_scale)
        if self.use_affine_alignment:
            if train_affine:
                self.alignment_weight = nn.Parameter(weight.clone())
                self.alignment_bias = nn.Parameter(bias.clone())
            else:
                self.register_buffer("alignment_weight", weight.clone())
                self.register_buffer("alignment_bias", bias.clone())

        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer("_delta_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        # Match models/adapters/mlp_adapter.py exactly.
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_uniform_(self.net[0].weight, a=0.3,
                                 nonlinearity="leaky_relu")
        nn.init.zeros_(self.net[0].bias)
        init_zero_linear(self.net[3])

    def start(self, source):
        if not self.use_affine_alignment:
            return source
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        z0 = self.start(source)
        out = z0 + self.residual_scale * self.net(self.norm(z0))
        with torch.no_grad():
            self._delta_ratio.fill_(
                (out - z0).norm() / z0.norm().clamp_min(1e-8))
        return out

    @torch.no_grad()
    def get_metrics(self):
        values = {
            "adapter/use_affine_alignment": float(self.use_affine_alignment),
            "adapter/train_affine": float(
                self.use_affine_alignment
                and isinstance(self.alignment_weight, nn.Parameter)),
            "adapter/delta_ratio": float(self._delta_ratio.cpu()),
        }
        if self.use_affine_alignment:
            values.update({
                "adapter/affine_weight_norm": float(self.alignment_weight.norm().cpu()),
                "adapter/affine_bias_norm": float(self.alignment_bias.norm().cpu()),
            })
        return values


class AffineWholeDirectMLPMapper(nn.Module, AffineStartMixin):
    """Affine alignment followed by one identity-initialized whole MLP."""

    def __init__(self, weight, bias, hidden_dims=None, dropout=0.0,
                 use_block_norm=False, use_final_norm=False,
                 train_affine=False, activation="identity"):
        super().__init__()
        self._init_affine(weight, bias, train_affine)
        dim = weight.size(0)
        hidden_dims = list(hidden_dims or [512, 512])
        layers = []
        in_dim = dim
        self.norm = nn.LayerNorm(dim) if use_block_norm else nn.Identity()
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, int(hidden_dim)),
                build_activation(activation),
                nn.Dropout(dropout),
            ])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()
        self.hidden_dims = hidden_dims
        self.reset_parameters()

    def reset_parameters(self):
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            init_identity_projection(layer)
        init_identity_projection(linear_layers[-1])

    def forward(self, source):
        z0 = self.start(source)
        x = self.net(self.norm(z0))
        x = self.final_norm(x)
        self._update_delta_ratio(z0, x)
        return x

    @torch.no_grad()
    def get_metrics(self):
        values = self._base_metrics()
        values["adapter/whole_mlp_depth"] = float(len(self.hidden_dims) + 1)
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
                 adaptive_gate_hidden=128, bottleneck_dim=128,
                 num_groups=16, group_hidden=64, gate_hidden=64,
                 gate_init=0.5, num_tokens=16, token_hidden=64,
                 channel_hidden=64, num_heads=2,
                 transformer_ffn_dim=128, num_experts=4,
                 attention_dim=32, attention_heads=4,
                 attention_dropout=0.0, attention_scale=0.1,
                 attention_input="value_delta",
                 attention_use_position=True,
                 flow_hidden_dim=128, whole_mlp_dims=None,
                 whole_mlp_activation="gelu", use_affine_alignment=True):
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
    if mapper_type == "affine_residual_mlp_attention":
        return AffineResidualMLPAttentionMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max, gate_mode=gate_mode,
            final_gate_max=final_gate_max,
            final_gate_init=final_gate_init,
            adaptive_gate_hidden=adaptive_gate_hidden,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            attention_scale=attention_scale,
            attention_input=attention_input,
            attention_use_position=attention_use_position)
    if mapper_type == "affine_iterative_residual":
        # num_blocks is reused as the number of refine iterations.
        return AffineIterativeResidualMapper(
            weight, bias, hidden_dim=hidden_dim, num_iters=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine, share_weights=True,
            use_step_embed=True)
    if mapper_type == "affine_iterative_residual_unshared":
        return AffineIterativeResidualMapper(
            weight, bias, hidden_dim=hidden_dim, num_iters=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine, share_weights=False,
            use_step_embed=True)
    if mapper_type == "affine_sens_weighted_residual":
        return AffineSensWeightedResidualMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_film_residual_mlp":
        return AffineFiLMResidualMLPMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_multiscale_residual_mlp":
        return AffineMultiScaleResidualMLPMapper(
            weight, bias, hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_lowrank_residual":
        return AffineLowRankResidualMapper(
            weight, bias, rank=lowrank_rank, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max)
    if mapper_type == "affine_bottleneck_residual":
        return AffineBottleneckResidualMapper(
            weight, bias, bottleneck_dim=bottleneck_dim,
            num_blocks=num_blocks, dropout=dropout,
            residual_scale=residual_scale, use_block_norm=use_block_norm,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_group_gated":
        return AffineGroupGatedMapper(
            weight, bias, num_groups=num_groups, group_hidden=group_hidden,
            gate_hidden=gate_hidden, num_blocks=num_blocks, dropout=dropout,
            residual_scale=residual_scale, gate_init=gate_init,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_token_mixer":
        return AffineTokenMixerMapper(
            weight, bias, num_tokens=num_tokens, token_hidden=token_hidden,
            channel_hidden=channel_hidden, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_tiny_transformer":
        return AffineTinyTransformerMapper(
            weight, bias, num_tokens=num_tokens, num_heads=num_heads,
            transformer_ffn_dim=transformer_ffn_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_moe_bottleneck":
        return AffineMoEBottleneckMapper(
            weight, bias, bottleneck_dim=bottleneck_dim,
            num_experts=num_experts, gate_hidden=gate_hidden,
            num_blocks=num_blocks, dropout=dropout,
            residual_scale=residual_scale, use_block_norm=use_block_norm,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_coupling_flow":
        return AffineCouplingFlowMapper(
            weight, bias, flow_hidden_dim=flow_hidden_dim,
            num_blocks=num_blocks, residual_scale=residual_scale,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_whole_residual_mlp":
        return AffineWholeResidualMLPMapper(
            weight, bias, hidden_dims=whole_mlp_dims, dropout=dropout,
            residual_scale=residual_scale, use_block_norm=use_block_norm,
            use_final_norm=use_final_norm, train_affine=train_affine,
            activation=whole_mlp_activation)
    if mapper_type == "affine_whole_direct_mlp":
        return AffineWholeDirectMLPMapper(
            weight, bias, hidden_dims=whole_mlp_dims, dropout=dropout,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine, activation=whole_mlp_activation)
    if mapper_type == "legacy_mlp_adapter":
        return LegacyMLPAdapterMapper(
            weight, bias, hidden_dim=hidden_dim, dropout=dropout,
            residual_scale=residual_scale,
            use_affine_alignment=use_affine_alignment,
            train_affine=train_affine)
    if mapper_type == "affine_linear":
        return AffineLinearStackMapper(
            weight, bias, num_layers=num_blocks, train_affine=train_affine)
    if mapper_type == "direct_mlp":
        return DirectMLPMapper(
            weight.size(0), hidden_dim=hidden_dim, num_layers=num_blocks,
            dropout=dropout)
    raise ValueError(f"Unknown mapper_type: {mapper_type}")
