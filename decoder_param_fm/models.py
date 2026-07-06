import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    if t.ndim == 0:
        t = t[None]
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


class ConditionExtractor(nn.Module):
    def __init__(self, method="random", code_dim=512, num_tokens=512,
                 d_model=512, num_heads=8, num_layers=2):
        super().__init__()
        self.method = method
        self.code_dim = code_dim
        self.num_tokens = num_tokens
        if method == "set_transformer":
            self.code_proj = nn.Sequential(
                nn.LayerNorm(code_dim),
                nn.Linear(code_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.queries = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
            self.layers = nn.ModuleList([
                nn.MultiheadAttention(
                    d_model, num_heads, batch_first=True, dropout=0.0)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, code_dim)

    @torch.no_grad()
    def _random(self, codes):
        n = codes.size(0)
        if n >= self.num_tokens:
            idx = torch.randperm(n, device=codes.device)[:self.num_tokens]
            out = codes[idx]
            mask = torch.ones(self.num_tokens, dtype=torch.bool, device=codes.device)
            return out, mask
        out = codes.new_zeros(self.num_tokens, codes.size(1))
        out[:n] = codes
        mask = torch.zeros(self.num_tokens, dtype=torch.bool, device=codes.device)
        mask[:n] = True
        return out, mask

    @torch.no_grad()
    def _svd(self, codes):
        z = codes.float()
        z = z - z.mean(dim=0, keepdim=True)
        # full_matrices=False keeps Vh at (min(N,D), D); pad if needed.
        _u, s, vh = torch.linalg.svd(z, full_matrices=False)
        k = min(self.num_tokens, vh.size(0))
        out = codes.new_zeros(self.num_tokens, codes.size(1))
        out[:k] = s[:k, None].to(out.dtype) * vh[:k].to(out.dtype)
        mask = torch.zeros(self.num_tokens, dtype=torch.bool, device=codes.device)
        mask[:k] = True
        return out, mask

    def forward(self, codes):
        if self.method == "random":
            return self._random(codes)
        if self.method == "svd":
            return self._svd(codes)
        if self.method != "set_transformer":
            raise ValueError(f"Unknown condition extractor: {self.method}")
        kv = self.code_proj(codes).unsqueeze(0)
        q = self.queries.unsqueeze(0)
        for attn in self.layers:
            delta, _ = attn(q, kv, kv, need_weights=False)
            q = self.norm(q + delta)
        out = self.out(q.squeeze(0))
        mask = torch.ones(self.num_tokens, dtype=torch.bool, device=codes.device)
        return out, mask


class ConditionEncoder(nn.Module):
    def __init__(self, code_dim=512, hidden_dim=512, cond_dim=512):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(code_dim),
            nn.Linear(code_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(cond_dim * 3),
            nn.Linear(cond_dim * 3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, tokens, mask):
        h = self.token_mlp(tokens)
        mask_f = mask.to(h.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=0).clamp_min(1.0)
        mean = (h * mask_f).sum(dim=0) / denom
        var = ((h - mean).pow(2) * mask_f).sum(dim=0) / denom
        std = var.clamp_min(1e-12).sqrt()
        masked = h.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = masked.max(dim=0).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool,
                               torch.zeros_like(max_pool))
        global_cond = self.pool(torch.cat([mean, std, max_pool], dim=-1))
        return h, global_cond


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(self.norm(x))


class ParamFM(nn.Module):
    def __init__(self, num_tensors, max_layer_id, max_token_offset,
                 token_size=512, hidden_dim=512, num_blocks=4, time_dim=128,
                 cond_dim=512, condition_inject="film", num_heads=8,
                 hyper_lora_rank=16, dropout=0.0):
        super().__init__()
        self.token_size = token_size
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.condition_inject = condition_inject
        self.param_proj = nn.Linear(token_size, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tensor_embed = nn.Embedding(num_tensors, hidden_dim)
        self.layer_embed = nn.Embedding(max_layer_id + 1, hidden_dim)
        self.offset_embed = nn.Embedding(max_token_offset + 1, hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        if condition_inject == "film":
            self.film = nn.Linear(cond_dim, hidden_dim * 2)
        elif condition_inject == "cross_attention":
            self.cond_token_proj = nn.Linear(cond_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True, dropout=0.0)
            self.cross_norm = nn.LayerNorm(hidden_dim)
        elif condition_inject == "hyper_lora":
            self.hyper_a = nn.Linear(cond_dim, hidden_dim * hyper_lora_rank)
            self.hyper_b = nn.Linear(cond_dim, hyper_lora_rank * token_size)
            self.hyper_alpha = nn.Linear(cond_dim, 1)
            self.hyper_lora_rank = hyper_lora_rank
        else:
            raise ValueError(f"Unknown condition injection: {condition_inject}")
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, token_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def meta_embedding(self, meta):
        return (
            self.tensor_embed(meta["tensor_ids"])
            + self.layer_embed(meta["layer_ids"])
            + self.offset_embed(meta["token_offsets"])
        )

    def forward(self, theta_tokens, t, meta, cond_tokens, cond_mask,
                global_cond):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=theta_tokens.dtype,
                             device=theta_tokens.device)
        if t.ndim == 0:
            t = t[None]
        t_emb = self.time_proj(timestep_embedding(t, self.time_dim)).squeeze(0)
        h = self.param_proj(theta_tokens)
        h = h + self.meta_embedding(meta)
        h = h + t_emb.unsqueeze(0)
        if self.condition_inject == "film":
            gamma, beta = self.film(global_cond).chunk(2, dim=-1)
            h = h * (1.0 + gamma.unsqueeze(0)) + beta.unsqueeze(0)
        elif self.condition_inject == "cross_attention":
            cond_h = self.cond_token_proj(cond_tokens).unsqueeze(0)
            key_padding_mask = ~cond_mask.unsqueeze(0)
            attn_out, _ = self.cross_attn(
                h.unsqueeze(0), cond_h, cond_h,
                key_padding_mask=key_padding_mask,
                need_weights=False)
            h = self.cross_norm(h + attn_out.squeeze(0))
        for block in self.blocks:
            h = block(h)
        base = self.out(self.norm(h))
        if self.condition_inject != "hyper_lora":
            return base
        a = self.hyper_a(global_cond).view(self.hidden_dim, self.hyper_lora_rank)
        b = self.hyper_b(global_cond).view(self.hyper_lora_rank, self.token_size)
        alpha = torch.tanh(self.hyper_alpha(global_cond)).view(1)
        lowrank = self.norm(h).matmul(a).matmul(b)
        return base + alpha * lowrank


class DecoderParamFMSystem(nn.Module):
    def __init__(self, condition_extractor, condition_encoder, param_fm):
        super().__init__()
        self.condition_extractor = condition_extractor
        self.condition_encoder = condition_encoder
        self.param_fm = param_fm

    def encode_condition(self, guide_codes):
        cond_tokens, cond_mask = self.condition_extractor(guide_codes)
        enc_tokens, global_cond = self.condition_encoder(cond_tokens, cond_mask)
        return enc_tokens, cond_mask, global_cond

    def forward(self, theta_tokens, t, meta, guide_codes):
        cond_tokens, cond_mask, global_cond = self.encode_condition(guide_codes)
        return self.param_fm(
            theta_tokens, t, meta, cond_tokens, cond_mask, global_cond)
