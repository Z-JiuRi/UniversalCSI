# 更通用 Decoder 的设计建议

## 目标

UniversalCSI 当前目标是支持不同 encoder：

```text
csinet
crnet
clnet
transnet
```

但 BS 端 decoder 希望尽量共享。用户进一步提出：未来不同 encoder 可以代表不同厂家 UE 设备，而 BS 端 decoder 固定。

因此 decoder 的设计目标不是只在某一个 encoder 上达到最优，而是：

```text
能够稳定接收不同 encoder 产生的 code 分布，并重建同一个中心化 CSI 空间。
```

## 原始 decoder 类型分析

### CsiNet / CRNet / CLNet decoder

这些模型的 decoder 本质类似：

```text
code
  -> Linear 或 ConvTranspose1d 展开
  -> reshape to (B, 2, 32, 32)
  -> CNN residual/refinement blocks
  -> sigmoid/hsigmoid
```

优点：

- CNN 对局部结构有强归纳偏置。
- 对 CSI sparse angle-delay 域中的局部稀疏结构较友好。
- 训练相对稳定。

缺点：

- 原版最终输出激活绑定 `[0,1]` 数据设定。
- 如果直接复用，需要删除最终 `sigmoid/hsigmoid`。

### TransNet decoder

TransNet decoder 本质是：

```text
code
  -> fc_decoder
  -> reshape to sequence
  -> TransformerDecoder
  -> reshape to (B, 2, nt, nc)
```

优点：

- 更偏全局建模。
- 能对长程 token 关系建模。
- 和 TransNet encoder 同源时结构自然。

缺点：

- 对 code 分布可能更敏感。
- 如果 encoder 不是 TransNet encoder，code 的语义和 token 化方式未必匹配。
- 缺少 CNN refinement 的局部重建先验。

## 通用 decoder 的核心问题

不同 encoder 即使输出维度相同：

```text
code_dim = channel * nt * nc / cr
```

其 code 分布仍可能差异很大：

- CsiNet encoder：浅层卷积 + FC，全局压缩。
- CRNet encoder：多尺度卷积分支。
- CLNet encoder：注意力门控 + 轻量卷积。
- TransNet encoder：Transformer token 混合。

因此 decoder 不应过度假设 code 来自某一种 encoder。

更通用的 decoder 应该先处理 code 分布差异，再进入重建模块。

## 推荐基础结构

建议采用三段式 decoder：

```text
code: (B, code_dim)
  -> code normalization / projection
  -> coarse reconstruction
  -> spatial refinement
  -> output: (B, channel, nt, nc)
```

一个实用版本：

```text
LayerNorm(code_dim)
  -> Linear(code_dim, input_dim)
  -> reshape to (B, channel, nt, nc)
  -> CNN residual blocks
  -> final Conv2d
  -> out = coarse + residual
```

其中：

```text
input_dim = channel * nt * nc
```

该设计有两个关键点：

1. `LayerNorm` 缓和不同 encoder code 的尺度差异。
2. CNN residual refinement 捕获共享的 CSI 空间结构。

## 为什么建议 residual refinement

直接让 CNN 输出完整重建：

```text
out = cnn(coarse)
```

会让 CNN 同时承担全局恢复和局部修正。

更建议：

```text
out = coarse + refinement(coarse)
```

这样：

- `Linear` 展开负责粗重建。
- CNN 只学习局部残差。
- 对不同 encoder code 分布更稳。
- 后续对 decoder 做 LoRA 时，也更容易定位适配位置。

## 候选 decoder 方案

### 方案 A：纯 TransNet decoder

```text
code
  -> Linear
  -> TransformerDecoder
  -> out
```

优点：

- 当前实现已有。
- 与 TransNet baseline 对齐。

缺点：

- 对非 TransNet encoder 未必最通用。
- 缺少局部 refinement。

适合作为 baseline。

### 方案 B：CNN residual decoder

```text
code
  -> LayerNorm
  -> Linear expand
  -> reshape
  -> CRBlock/RefineNet blocks
  -> out = coarse + residual
```

优点：

- 简单稳定。
- 适合多 encoder。
- 适合作为通用 decoder 的强 baseline。

缺点：

- 全局建模弱于 Transformer。

建议优先实现。

### 方案 C：Hybrid decoder

```text
code
  -> LayerNorm
  -> Linear to tokens
  -> Transformer blocks
  -> reshape
  -> CNN residual refinement
  -> out
```

优点：

- 兼具全局建模和局部重建先验。
- 更适合作为最终 base decoder 候选。

缺点：

- 结构更复杂。
- 需要更多实验判断是否值得。

建议在 CNN residual baseline 后再尝试。

## 不建议的方向

### 不建议保留最终 sigmoid/hsigmoid

当前数据为 `[-0.5,0.5]`，最终输出应为线性实值。

### 不建议一开始做过深 U-Net

输入只有 `(2,32,32)`，且 code 已经高度压缩。过深 decoder 可能：

- 参数量过大。
- 训练不稳定。
- 掩盖 encoder 差异。
- 不利于后续 LoRA 分析。

### 不建议一开始对 decoder 加过多可配置分支

通用 decoder 初期应保持结构可控。否则很难判断收益来自：

- code normalization
- CNN refinement
- Transformer mixing
- 参数量增加
- 训练随机性

## 建议实验矩阵

先增加 decoder 参数：

```bash
--decoder transnet
--decoder cnn
--decoder hybrid
```

然后对每个 encoder 测试：

```bash
--encoder csinet
--encoder crnet
--encoder clnet
--encoder transnet
```

形成矩阵：

```text
encoder x decoder
```

关注两类指标：

1. 端到端训练 NMSE。
2. 固定 decoder 后，仅训练 encoder 或 adapter 的 NMSE。

第二类指标更接近多厂家泛化目标。

## 当前建议优先级

短期建议：

```text
先实现 CNN residual decoder baseline
```

中期建议：

```text
实现 hybrid decoder:
LayerNorm(code) -> Linear -> Transformer -> CNN residual refinement
```

长期建议：

```text
选择最容易被 LoRA 低秩适配的 decoder，而不是只选择端到端 NMSE 最低的 decoder。
```

