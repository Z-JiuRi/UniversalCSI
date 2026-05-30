'''
Swin 风格 CSI 编码器：窗口自注意力编码 CSI，输出 (N, code_dim)。

记号：
input_dim = channel * nt * nc
code_dim = input_dim // reduction
feature_dim = 2 * embed_dim * (nt // (patch * 2)) * (nc // (patch * 2))

模型架构：
.
├── patch_embed
│   ├── Conv2d(channel, embed_dim, patch, stride=patch)
│   ├── BatchNorm2d(embed_dim)
│   └── LeakyReLU(0.3)
├── WindowAttentionBlock(embed_dim, window_size, num_heads=4)
│   ├── Permute(N, C, H, W -> N, H, W, C)
│   ├── View(N, H // window_size, window_size, W // window_size, window_size, C)
│   ├── Permute(N, H // window_size, W // window_size, window_size, window_size, C)
│   ├── Contiguous()
│   ├── View(num_windows * N, window_size * window_size, C)
│   ├── LayerNorm(embed_dim)
│   ├── MultiheadAttention(embed_dim, num_heads, batch_first=True)
│   ├── Add()
│   ├── LayerNorm(embed_dim)
│   ├── Linear(embed_dim, 4 * embed_dim)
│   ├── GELU()
│   ├── Linear(4 * embed_dim, embed_dim)
│   ├── Add()
│   ├── View(N, H // window_size, W // window_size, window_size, window_size, C)
│   ├── Permute(N, H // window_size, window_size, W // window_size, window_size, C)
│   ├── Contiguous()
│   ├── View(N, H, W, C)
│   ├── Permute(N, C, H, W)
│   └── Contiguous()
├── patch_merge
│   ├── Conv2d(embed_dim, 2 * embed_dim, 2, stride=2)
│   ├── BatchNorm2d(2 * embed_dim)
│   └── LeakyReLU(0.3)
├── WindowAttentionBlock(2 * embed_dim, window_size, num_heads=4)
├── Flatten()
└── Linear(feature_dim, code_dim)

使用技术：
局部窗口多头自注意力建模角延迟域局部 token 关系；patch embedding 和 patch
merging 构建分层分辨率；Transformer 残差 MLP 提升全局可表达性。

保存模型权重时的参数维度：
patch_embed.conv.weight:                             (embed_dim, channel, patch, patch)
patch_embed.conv.bias:                                                     (embed_dim,)
patch_embed.bn.weight/bias/running_mean/running_var:                       (embed_dim,)
patch_merge.conv.weight:                               (2 * embed_dim, embed_dim, 2, 2)
patch_merge.conv.bias:                                                 (2 * embed_dim,)
patch_merge.bn.weight/bias/running_mean/running_var:                   (2 * embed_dim,)
每个 WindowAttentionBlock.norm1.weight/bias:                                       (dim,)
每个 WindowAttentionBlock.attn.in_proj_weight:                             (3 * dim, dim)
每个 WindowAttentionBlock.attn.in_proj_bias:                                   (3 * dim,)
每个 WindowAttentionBlock.attn.out_proj.weight:                                (dim, dim)
每个 WindowAttentionBlock.attn.out_proj.bias:                                      (dim,)
每个 WindowAttentionBlock.norm2.weight/bias:                                       (dim,)
每个 WindowAttentionBlock.mlp.fc1.weight:                                  (4 * dim, dim)
每个 WindowAttentionBlock.mlp.fc1.bias:                                        (4 * dim,)
每个 WindowAttentionBlock.mlp.fc2.weight:                                  (dim, 4 * dim)
每个 WindowAttentionBlock.mlp.fc2.bias:                                            (dim,)
fc.weight:                                                      (code_dim, feature_dim)
fc.bias:                                                                    (code_dim,)
'''


import torch.nn as nn
from collections import OrderedDict


class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(dim, mlp_ratio * dim)),
            ("gelu", nn.GELU()),
            ("fc2", nn.Linear(mlp_ratio * dim, dim)),
        ]))

    def _window_partition(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        assert h % ws == 0 and w % ws == 0
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(b, h // ws, ws, w // ws, ws, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, ws * ws, c), h, w

    def _window_reverse(self, windows, b, h, w):
        ws = self.window_size
        c = windows.size(-1)
        x = windows.view(b, h // ws, w // ws, ws, ws, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(b, h, w, c)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        b = x.size(0)
        windows, h, w = self._window_partition(x)
        attn_input = self.norm1(windows)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input,
                                need_weights=False)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))
        return self._window_reverse(windows, b, h, w)


class SwinCsiEncoder(nn.Module):
    def __init__(self, reduction=4, channel=2, nt=32, nc=32, embed_dim=32,
                 patch=2, window_size=4):
        super().__init__()
        input_dim = channel * nt * nc
        assert input_dim % reduction == 0
        assert nt % (patch * 2) == 0 and nc % (patch * 2) == 0
        assert (nt // patch) % window_size == 0
        assert (nc // patch) % window_size == 0
        assert (nt // (patch * 2)) % window_size == 0
        assert (nc // (patch * 2)) % window_size == 0
        self.patch_embed = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(channel, embed_dim, patch, stride=patch)),
            ("bn", nn.BatchNorm2d(embed_dim)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.stage1 = WindowAttentionBlock(embed_dim, window_size, num_heads=4)
        self.patch_merge = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(embed_dim, 2 * embed_dim, 2, stride=2)),
            ("bn", nn.BatchNorm2d(2 * embed_dim)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.stage2 = WindowAttentionBlock(2 * embed_dim, window_size,
                                           num_heads=4)
        feature_dim = 2 * embed_dim * (nt // (patch * 2)) * (nc // (patch * 2))
        self.fc = nn.Linear(feature_dim, input_dim // reduction)

    def forward(self, x):
        out = self.patch_embed(x)
        out = self.stage1(out)
        out = self.patch_merge(out)
        out = self.stage2(out)
        return self.fc(out.flatten(1))
