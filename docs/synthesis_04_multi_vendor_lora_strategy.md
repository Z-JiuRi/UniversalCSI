# 多厂家 UE 泛化与 LoRA 生成策略

## 研究目标

目标是在 FDD Massive MIMO CSI feedback 中提升跨多个 UE 厂家的泛化能力。

抽象设定：

```text
UE vendor A -> encoder_A
UE vendor B -> encoder_B
UE vendor C -> encoder_C
unknown UE vendor -> encoder_unknown

BS side -> fixed base decoder
adaptation -> LoRA / adapter injected into decoder
```

最终希望：

```text
BS 不需要为每个厂家维护完整 decoder；
只需固定一个 base decoder，并为不同厂家生成少量 LoRA 参数。
```

## 固定 BS base decoder 是否合理

合理。

现实通信部署中：

- BS 硬件固定且算力较强。
- UE 设备多样，厂家可能有私有压缩算法。
- BS 维护多套完整 decoder 的部署和认证成本高。

更现实的形态是：

```text
fixed base decoder W0
small vendor-specific / generated adapter ΔW
```

也就是：

```text
decoder weights = W0 + ΔW_lora
```

## 不建议完全无适配

不同 UE encoder 输出的 code 分布可能非常不同：

```text
CsiNet encoder -> shallow CNN + FC code
CRNet encoder -> multi-resolution CNN code
CLNet encoder -> attention-gated CNN code
TransNet encoder -> Transformer mixed code
unknown vendor -> private code manifold
```

如果 BS decoder 完全固定且没有任何 adapter，泛化可能过硬。

因此推荐：

```text
固定 base decoder 主干；
允许少量 LoRA / adapter 根据厂家或 domain 变化。
```

## code_adapter 的角色升级

原始 `code_adapter`：

```text
LayerNorm(code_dim) + Linear(code_dim, code_dim)
```

在普通训练中它只是 optional trick。

在多厂家 LoRA 生成任务中，它应升级为：

```text
Universal Semantic Projector
```

作用：

```text
raw private code
  -> normalized / projected latent
  -> decoder input
  -> LoRA generator condition
```

原因：

- 不同厂家 encoder 的 code 流形不同。
- 直接用 raw code 作为 diffusion/flow-matching 条件可能泛化差。
- 需要先投影到统一语义空间。

## Decoder 侧 LoRA 插入点

### 1. Initial projection / fc_decoder

位置：

```text
code -> fc_decoder -> input_dim / token sequence
```

这是最重要位置。

原因：

- 它是 code 进入 BS decoder 的第一层。
- 不同厂家 code 分布差异首先需要在这里对齐。
- 参数结构简单，适合 LoRA 和 LoRA 生成。

推荐优先测试：

```text
LoRA rank = 4, 8, 16
LoRA alpha = 2 * rank or 4 * rank
```

### 2. Transformer FFN

位置：

```text
Transformer layer linear1
Transformer layer linear2
```

作用：

- 调整 token feature 非线性变换。
- 训练通常比 attention LoRA 稳定。

### 3. Attention projections

位置：

```text
q projection
k projection
v projection
out projection
```

作用：

- 改变 token 间交互方式。
- 适合更复杂的 domain shift。

风险：

- 生成参数更多。
- 训练更难。

### 4. CNN residual head

初期不建议动态化 CNN head。

原因：

- CNN head 应作为共享物理 refinement 先验。
- 如果它也被 LoRA 大幅改变，难以分析收益来源。
- LoRA generator 输出维度会增加。

只有当实验显示不同 vendor 的残差图样式差异很大时，再考虑对 CNN head 的最后一层 `Conv2d` 做轻量 LoRA。

## LoRA 生成：样本级 vs 厂家级

### 不推荐初期使用单样本 code 生成 LoRA

形式：

```text
single code -> generator -> LoRA weights
decoder(code; LoRA(code)) -> reconstruction
```

问题：

- code 同时包含厂家压缩特征和瞬态信道信息。
- 生成器可能把样本噪声当成厂家特征。
- 每个样本都生成 LoRA，推理开销高。
- 泛化解释困难。

### 推荐厂家级 / domain-level LoRA

推荐形式：

```text
K calibration codes
  -> domain embedding
  -> generator
  -> LoRA weights
```

部署流程：

```text
1. 新 UE 接入 BS。
2. BS 请求 K 个 calibration CSI。
3. UE encoder 发送 {code_1, ..., code_K}。
4. BS 聚合得到 domain embedding。
5. LoRA generator 根据 domain embedding 生成 LoRA。
6. 该 LoRA 在整个 UE session 中保持固定。
```

这样生成的是厂家级或 encoder 级 adapter，而不是每个样本的动态参数。

## Domain embedding 提取

最简单 baseline：

```python
z_k = MLP(code_k)
domain_embedding = mean(z_k, dim=K)
```

要求：

```text
permutation-invariant
stable across samples
captures vendor / encoder manifold
```

可选升级：

```text
attention pooling
DeepSets
Set Transformer
vendor id embedding
code statistics: mean / variance / covariance summary
```

建议初期：

```text
MeanPooling(MLP(code_1 ... code_K))
```

因为它足够简单，便于判断 LoRA 生成路线是否有效。

## LoRA generator

### Phase 1 generator baseline: MLP

先使用：

```text
domain_embedding -> MLP -> LoRA A/B matrices
```

不要直接上 diffusion 或 flow-matching。

原因：

- LoRA 参数是高维结构化输出。
- 复杂生成模型训练成本高。
- 如果 MLP generator 都无收益，复杂生成模型大概率也不会稳定。

### Diffusion / Flow-Matching 何时引入

只有当以下条件满足时再考虑：

```text
static per-encoder LoRA 有明显收益；
MLP generator 能接近 static LoRA 上限；
但在 unseen vendor 或多模态 adapter 上仍有瓶颈。
```

适用场景：

- 同一 domain 下存在多个有效 LoRA 解。
- 需要建模 LoRA 参数分布，而不是单点回归。
- 需要在未知 encoder/vendor 间做连续插值。

## Static LoRA 的重要性

在训练生成器之前，必须先做：

```text
每个 encoder/vendor 一套 static LoRA
```

目的：

1. 验证 LoRA 插入位置是否有效。
2. 验证 base decoder 是否适合低秩适配。
3. 给 LoRA generator 提供上限参考。
4. 可作为 ground-truth LoRA dataset。

如果 static LoRA 无法恢复 NMSE 损失，则不应进入 LoRA generator 阶段。此时需要重新考虑：

```text
base decoder 结构
LoRA 插入位置
LoRA rank
code_adapter
训练方式
```

## 推荐最终系统形态

```text
Calibration stage:
  UE sends K compressed codes
  BS builds domain embedding
  LoRA generator outputs LoRA weights

Inference stage:
  UE sends real-time code
  BS decodes with fixed base decoder + generated LoRA
```

数学形式：

```text
z_domain = Pool({Adapter(code_k)}_{k=1}^K)
ΔW_lora = Generator(z_domain)
H_hat = Decoder_W0+ΔW(code)
```

## 关键评价指标

除 NMSE 外，必须记录：

```text
LoRA rank
LoRA parameter count
LoRA gain over no-LoRA
static LoRA upper bound
generated LoRA gap to static LoRA
calibration sample count K
unknown vendor zero-shot / few-shot performance
inference overhead
```

## 合并建议

主路线：

```text
HybridDecoder base W0
  + code_adapter / domain projector
  + fc_decoder LoRA first
  + per-vendor static LoRA
  + MLP generator
  + diffusion / flow-matching only if needed
```

不要跳过 static LoRA。

不要一开始使用单样本 code 生成 LoRA。

不要把 CNN head 过早 LoRA 化。

