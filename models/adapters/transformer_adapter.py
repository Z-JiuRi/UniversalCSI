import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransformerAdapter(nn.Module):
    """Transformer-based adapter with residual connection.

    Splits the code vector into tokens, processes them through a
    lightweight Transformer encoder, then projects back.  The output
    projection is zero-initialized so the adapter starts as identity.

    Architecture:
      x → LN → tokenize → reshape(B, N, d) → TransformerEncoder
              → reshape(B, -) → out_proj(zero-init) → +x
    """

    def __init__(self, adapter_dim, d_model=64, nhead=2,
                 dim_feedforward=256, num_layers=1, dropout=0.0):
        super().__init__()
        assert adapter_dim % d_model == 0, \
            f"adapter_dim ({adapter_dim}) must be divisible by d_model ({d_model})"
        self.num_tokens = adapter_dim // d_model
        self.d_model = d_model

        self.norm = nn.LayerNorm(adapter_dim)
        self.tokenize = nn.Linear(adapter_dim, adapter_dim, bias=False)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout,
            batch_first=True)
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model))

        self.out_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)

        self.register_buffer("_ratio", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_uniform_(self.tokenize.weight, a=0.3,
                                 nonlinearity='leaky_relu')
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        B = x.size(0)
        t = self.tokenize(self.norm(x))
        t = t.view(B, self.num_tokens, self.d_model)
        t = self.transformer(t)
        t = t.reshape(B, -1)
        delta = self.out_proj(t)
        with torch.no_grad():
            self._ratio.fill_(delta.norm() / x.norm().clamp_min(1e-8))
        return x + delta

    @torch.no_grad()
    def get_metrics(self):
        """Return lightweight metrics about the adapter state."""
        metrics = {
            "adapter/out_proj_norm": float(self.out_proj.weight.norm().cpu()),
            "adapter/delta_ratio": float(self._ratio.cpu()),
        }
        # attention projection weight norms (first layer)
        attn = self.transformer.layers[0].self_attn
        metrics["adapter/attn_Q_norm"] = float(attn.in_proj_weight[:self.d_model].norm().cpu())
        metrics["adapter/attn_K_norm"] = float(attn.in_proj_weight[self.d_model:2*self.d_model].norm().cpu())
        metrics["adapter/attn_V_norm"] = float(attn.in_proj_weight[2*self.d_model:].norm().cpu())
        metrics["adapter/attn_O_norm"] = float(attn.out_proj.weight.norm().cpu())
        # FFN weight norms (first layer)
        ffn = self.transformer.layers[0]
        metrics["adapter/ffn_W1_norm"] = float(ffn.linear1.weight.norm().cpu())
        metrics["adapter/ffn_W2_norm"] = float(ffn.linear2.weight.norm().cpu())
        return metrics
