'''
MLP-Mixer CSI 编码器：把 CSI 切成 patch token 后用 MLP 混合，输出 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
num_tokens = (nt // patch) * (nc // patch)

模型架构：
.
├── Conv2d(channel, d_model, patch, stride=patch)
├── Flatten(start_dim=2)
├── Transpose(1, 2)
├── MixerBlock(num_tokens, d_model) * depth
│   ├── LayerNorm(d_model)
│   ├── Transpose(1, 2)
│   ├── Linear(num_tokens, 2 * num_tokens)
│   ├── GELU()
│   ├── Linear(2 * num_tokens, num_tokens)
│   ├── Transpose(1, 2)
│   ├── Add()
│   ├── LayerNorm(d_model)
│   ├── Linear(d_model, 4 * d_model)
│   ├── GELU()
│   ├── Linear(4 * d_model, d_model)
│   └── Add()
├── LayerNorm(d_model)
├── Flatten()
└── Linear(num_tokens * d_model, code_dim)

使用技术：
patch embedding 将二维 CSI 网格转成 token；token MLP 混合空间位置；
channel MLP 混合每个 token 的特征通道，不使用自注意力。

保存模型权重时的参数维度：
patch_embed.weight:                     (d_model, channel, patch, patch)
patch_embed.bias:                                             (d_model,)
每个 MixerBlock.token_norm.weight/bias:                         (d_model,)
每个 MixerBlock.token_mlp.0.weight:           (2 * num_tokens, num_tokens)
每个 MixerBlock.token_mlp.0.bias:                        (2 * num_tokens,)
每个 MixerBlock.token_mlp.2.weight:           (num_tokens, 2 * num_tokens)
每个 MixerBlock.token_mlp.2.bias:                            (num_tokens,)
每个 MixerBlock.channel_norm.weight/bias:                       (d_model,)
每个 MixerBlock.channel_mlp.0.weight:               (4 * d_model, d_model)
每个 MixerBlock.channel_mlp.0.bias:                         (4 * d_model,)
每个 MixerBlock.channel_mlp.2.weight:               (d_model, 4 * d_model)
每个 MixerBlock.channel_mlp.2.bias:                             (d_model,)
norm.weight:                                                  (d_model,)
norm.bias:                                                    (d_model,)
fc.weight:                              (code_dim, num_tokens * d_model)
fc.bias:                                                     (code_dim,)
'''


import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, d_model, hidden_tokens=None,
                 hidden_channels=None):
        super().__init__()
        hidden_tokens = hidden_tokens or 2 * num_tokens
        hidden_channels = hidden_channels or 4 * d_model
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, hidden_tokens),
            nn.GELU(),
            nn.Linear(hidden_tokens, num_tokens),
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, d_model),
        )

    def forward(self, x):
        out = self.token_norm(x).transpose(1, 2)
        out = self.token_mlp(out).transpose(1, 2)
        x = x + out
        return x + self.channel_mlp(self.channel_norm(x))


class MLPMixerCsiEncoder(nn.Module):
    def __init__(self, reduction=4, d_model=64, channel=2, nt=32, nc=32,
                 patch=4, depth=2):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % patch == 0 and nc % patch == 0
        num_tokens = (nt // patch) * (nc // patch)
        self.patch_embed = nn.Conv2d(channel, d_model, patch, stride=patch)
        self.blocks = nn.Sequential(*[
            MixerBlock(num_tokens, d_model) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(num_tokens * d_model, input_dim // reduction)

    def forward(self, x):
        out = self.patch_embed(x).flatten(2).transpose(1, 2)
        out = self.blocks(out)
        out = self.norm(out)
        return self.fc(out.flatten(1))
