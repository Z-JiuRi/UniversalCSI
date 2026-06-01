# TransNet 重构与模型维度契约

## 背景

TransNet 是 UniversalCSI 当前共享 decoder 的来源，也是多厂家泛化研究中最重要的 base decoder baseline。

两边材料都强调了一个共同点：

```text
TransNet 需要使用现代 PyTorch 官方 Transformer API，并显式使用 batch_first=True。
```

这既减少自定义 attention 代码的风险，也更适合后续对 `nn.Linear` 层注入 LoRA。

## 原始 TransNet 实现中的问题

历史 TransNet 实现或类似自定义 Transformer 实现中容易出现两个问题。

### batch_first 维度错位

PyTorch 旧版 `TransformerEncoderLayer` 默认输入格式是：

```text
(seq_len, batch, embed_dim)
```

但 CSI 代码中常见的输入整理方式是：

```text
(batch, seq_len, embed_dim)
```

如果没有显式设置：

```python
batch_first=True
```

则 attention 可能错误地把 batch 维当成 sequence 维，导致不同样本之间发生注意力交互。

正确约定：

```text
Transformer input: (B, seq_len, d_model)
```

### 自定义 attention 参数顺序风险

Gemini 材料提到，原始自定义 `scale_dot_attention` 曾存在参数顺序问题：

```text
调用位置参数顺序与函数签名不一致
```

这类问题在自定义 attention 中很难通过 shape 错误暴露，可能变成 silent bug。

因此建议使用 PyTorch 官方实现：

```python
nn.TransformerEncoderLayer
nn.TransformerDecoderLayer
nn.TransformerEncoder
nn.TransformerDecoder
```

## 当前推荐实现原则

### 使用官方 PyTorch Transformer

推荐：

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=2,
    dim_feedforward=dim_feedforward,
    dropout=0.,
    batch_first=True,
)

decoder_layer = nn.TransformerDecoderLayer(
    d_model=d_model,
    nhead=2,
    dim_feedforward=dim_feedforward,
    dropout=0.,
    batch_first=True,
)
```

优点：

- 语义清晰。
- 减少自定义 attention bug。
- 可能使用 PyTorch 后端优化。
- 更容易替换内部 Linear 为 LoRA 版本。

### 不使用自回归 mask

CSI 重建不是语言模型式自回归生成任务。

输入 code 是整体压缩表示，输出是完整 CSI map：

```text
code -> full reconstruction
```

因此不需要 causal mask。

Transformer 在这里更接近：

```text
token mixer / global context mixer
```

而不是自回归 decoder。

## TransNet 维度契约

默认 COST2100 sparse CSI：

```text
input shape = (B, 2, 32, 32)
input_dim = 2 * 32 * 32 = 2048
d_model = 64
seq_len = input_dim / d_model = 32
```

压缩率参数：

```text
cr = reduction denominator
code_dim = input_dim / cr
```

例如：

```text
cr = 4
code_dim = 2048 / 4 = 512
```

### Encoder 侧

TransNet encoder：

```text
input:  (B, 2, 32, 32)
view:   (B, 32, 64)
encoder output: (B, 32, 64)
flatten: (B, 2048)
fc_encoder: (B, 2048) -> (B, 512)
```

### Decoder 侧

TransNet decoder：

```text
code: (B, 512)
fc_decoder: (B, 512) -> (B, 2048)
view: (B, 32, 64)
TransformerDecoder / token mixer: (B, 32, 64)
reshape: (B, 2, 32, 32)
return out
```

## UniversalCSI 的统一接口

所有 encoder 应输出：

```text
code: (B, code_dim)
```

所有 decoder 应接收：

```text
code: (B, code_dim)
```

并输出：

```text
reconstruction: (B, channel, nt, nc)
```

在当前设定下：

```text
channel = 2
nt = 32
nc = 32
```

但实现应尽量支持：

```text
channel * nt * nc
```

作为通用 `input_dim`。

必须满足：

```text
input_dim % d_model == 0
input_dim % cr == 0
```

## TransNetDecoder 作为 baseline 的定位

TransNetDecoder 是后续所有 decoder 方案的强 baseline：

```text
code
  -> fc_decoder
  -> Transformer token mixer
  -> reshape
  -> linear output
```

优点：

- 表达能力强。
- LoRA 插入点清晰。
- 与 Gemini 的 foundation decoder 建议一致。
- 与当前 UniversalCSI 主线兼容。

局限：

- 缺少输出空间的局部 refinement 先验。
- 对不同 encoder code 分布可能敏感。
- 未必是最终多厂家 LoRA 适配能力最强的 base decoder。

因此合并建议是：

```text
保留 TransNetDecoder 作为 baseline；
在此基础上实现 HybridDecoder 做主推荐候选。
```

## LoRA 友好性

TransNet / HybridDecoder 中最适合 LoRA 的位置：

```text
fc_decoder / initial projection
Transformer FFN linear1 / linear2
attention projection layers
```

这些都是标准线性层或可被替换为 LoRALinear 的模块。

这也是为什么 Transformer 主干适合作为 base decoder：

```text
结构规则
参数矩阵明确
LoRA 注入自然
后续生成 LoRA 权重更容易定义
```

## 当前模型输出约定

无论是 TransNetDecoder 还是 HybridDecoder，最终输出都应保持：

```python
return out
```

不加：

```text
sigmoid
hsigmoid
tanh
Identity
```

因为当前数据范围已经是：

```text
[-0.5, 0.5]
```

