# Decoder 选型与 HybridDecoder 推荐设计

## 问题定义

在普通 CSI autoencoder 中，decoder 只需要和同源 encoder 配合：

```text
encoder_A + decoder_A
```

但 UniversalCSI 的研究目标更接近：

```text
encoder_A / encoder_B / encoder_C / unknown encoder
  -> shared BS decoder
  -> reconstruction
```

因此 decoder 选型问题应定义为：

```text
哪个 fixed base decoder 最适合作为多 encoder / 多厂家 LoRA 适配的底座？
```

而不是简单地问：

```text
哪个 decoder 端到端训练 NMSE 最低？
```

## Gemini 和 Codex 的观点合并

Gemini 早期建议：

```text
必须使用 Transformer Decoder。
```

理由是：

- CNN decoder 有局部归纳偏置。
- 不同 UE encoder 的 code 流形差异很大。
- Transformer 表达能力更自由，更适合作为 foundation decoder。

Codex 的保留意见：

```text
CNN decoder 的偏置作用在输出 CSI map 的空间结构上，
不一定等于假设输入 code 来自 CNN encoder。
```

CSI 输出始终是：

```text
(B, 2, 32, 32)
```

它具有共享的角延迟域局部稀疏性和结构连续性。利用这一输出空间先验可能提升泛化稳定性。

Gemini 后续文档也收敛到 Hybrid 路线：

```text
Transformer 负责全局语义对齐；
CNN residual blocks 负责物理结构细化。
```

因此合并后的主推荐是：

```text
HybridDecoder = Transformer global alignment + CNN residual refinement
```

## 三类 decoder 候选

### A. Pure TransNetDecoder

结构：

```text
code
  -> fc_decoder
  -> reshape to tokens
  -> TransformerDecoder / token mixer
  -> reshape to CSI map
  -> return out
```

优点：

- 当前已有。
- 表达能力强。
- LoRA 插入点规则。
- 是最重要 baseline。

风险：

- 缺少空间 refinement。
- 对不同 encoder code 分布可能较敏感。
- 可能需要更大模型容量去重新学习 CSI 局部先验。

### B. Pure CNNResidualDecoder

结构：

```text
code
  -> LayerNorm
  -> Linear expand
  -> reshape to CSI map
  -> CNN residual blocks
  -> return out
```

优点：

- 简单稳定。
- 利用输出 CSI map 的局部结构。
- 训练成本较低。

风险：

- 全局 code 语义对齐能力弱于 Transformer。
- LoRA 插入点相对少。
- 不一定适合作为最终 foundation decoder。

定位：

```text
必要 baseline，但不是主推荐最终方案。
```

### C. HybridDecoder

结构：

```text
code
  -> CodeAdapter
  -> Transformer token projection / token mixer
  -> reshape to CSI map
  -> CNN residual refinement
  -> return coarse + residual
```

优点：

- Transformer 负责对齐异构 encoder code 流形。
- CNN 负责利用 CSI 输出空间物理先验。
- 适合固定 base decoder + LoRA adapter 研究。
- 可通过消融关闭 CNN head，验证其真实收益。

定位：

```text
合并建议中的主推荐 decoder。
```

## 推荐 HybridDecoder 结构

默认 COST2100 参数：

```text
channel = 2
nt = 32
nc = 32
input_dim = 2048
cr = 4
code_dim = 512
d_model = 64
seq_len = 32
```

### 总体 pipeline

```text
输入: code (B, input_dim / cr)

Stage 1: CodeAdapter / Universal Semantic Projector
  LayerNorm(code_dim)
  Linear(code_dim, code_dim)

Stage 2: Token Projection
  LoRALinear(code_dim, input_dim)
  reshape -> (B, seq_len, d_model)

Stage 3: Transformer Global Alignment
  TransformerEncoderLayer 或 TransformerDecoderLayer * N
  batch_first=True
  no causal mask

Stage 4: CSI Map Reconstruction
  reshape -> (B, channel, nt, nc)

Stage 5: CNN Residual Refinement
  residual = CNNRefiner(coarse)
  output = coarse + residual

输出: (B, channel, nt, nc)
```

注意：

```text
最终输出为线性值，匹配 [-0.5, 0.5] 数据分布。
```

### 为什么 TransformerEncoderLayer 也可以

虽然命名上叫 decoder，但这里不是自回归生成。code 已经是完整压缩表示，不存在未来 token 泄露问题。

因此 Stage 3 可以使用：

```text
TransformerEncoderLayer
```

作为 token mixer。

也可以使用：

```text
TransformerDecoderLayer(memory, memory)
```

与当前 TransNetDecoder 更一致。

建议初期为了与当前 TransNetDecoder 对齐，保留 decoder-style 实现：

```python
tokens = self.transformer(tokens, tokens)
```

后续可消融：

```text
encoder-style token mixer vs decoder-style token mixer
```

## CNN residual head 设计

### 初版轻量结构

推荐：

```text
Conv2d(channel, hidden, 3, padding=1)
LeakyReLU(0.3)
ResidualBlock(hidden)
ResidualBlock(hidden)
Conv2d(hidden, channel, 3, padding=1)
```

输出：

```text
residual: (B, channel, nt, nc)
out = coarse + residual
```

### ResidualBlock

简单版本：

```text
x
  -> Conv2d(hidden, hidden, 3, padding=1)
  -> LeakyReLU(0.3)
  -> Conv2d(hidden, hidden, 3, padding=1)
  -> add x
  -> LeakyReLU(0.3)
```

可选加入 normalization：

```text
BatchNorm2d
GroupNorm
no norm
```

如果 batch size 小，优先测试：

```text
GroupNorm 或 no norm
```

### 超参建议

```text
refine_hidden = 16 or 32
refine_blocks = 2
kernel_size = 3
activation = LeakyReLU(0.3)
```

不要一开始做太重的 CNN head。CNN head 应该是输出 refinement，而不是替代 Transformer 主干。

## LoRA 友好的 HybridDecoder

HybridDecoder 中最关键的 LoRA 注入点：

```text
1. initial projection / fc_decoder / LoRALinear_1
2. Transformer FFN linear1 / linear2
3. attention q/k/v/out projections
4. optional CNN head last Conv2d
```

优先级：

```text
fc_decoder > FFN > attention > CNN head
```

初期建议只给：

```text
fc_decoder
```

加 LoRA，验证最低复杂度适配能力。

## 推荐伪代码

```python
class HybridDecoder(nn.Module):
    def __init__(self, reduction, d_model, channel, nt, nc,
                 dim_feedforward, refine_hidden=32, refine_blocks=2):
        super().__init__()
        self.input_dim = channel * nt * nc
        self.code_dim = self.input_dim // reduction
        self.seq_len = self.input_dim // d_model

        self.code_norm = nn.LayerNorm(self.code_dim)
        self.fc_decoder = nn.Linear(self.code_dim, self.input_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=dim_feedforward,
            dropout=0.,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(d_model),
        )

        self.refine = CNNResidualHead(
            channel=channel,
            hidden=refine_hidden,
            blocks=refine_blocks,
        )

    def forward(self, code):
        b = code.size(0)
        code = self.code_norm(code)
        tokens = self.fc_decoder(code).view(b, self.seq_len, self.d_model)
        tokens = self.transformer(tokens, tokens)
        coarse = tokens.view(b, self.channel, self.nt, self.nc)
        residual = self.refine(coarse)
        return coarse + residual
```

实际实现中需要保存：

```python
self.d_model
self.channel
self.nt
self.nc
```

## 推荐结论

合并结论不是：

```text
纯 Transformer 一定最好
```

也不是：

```text
CNN decoder 一定更适合 CSI
```

而是：

```text
以 Transformer 作为 foundation 主干，
以轻量 CNN residual head 注入输出空间物理先验，
最终通过冻结 base decoder 后的 LoRA 适配能力决定是否保留 CNN head。
```

