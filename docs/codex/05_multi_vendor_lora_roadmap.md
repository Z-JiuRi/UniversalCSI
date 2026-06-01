# 多厂家 UE 泛化与 LoRA 生成路线

## 用户目标

用户的长期研究目标是：

```text
多厂家泛化提升
```

设定中：

- 不同 encoder 对应不同厂家 UE 设备。
- BS 端 decoder 希望固定。
- 后续希望对 decoder 做 LoRA。
- 进一步希望用 encoder 输出的压缩数据指导 LoRA 参数生成。
- LoRA 参数生成方法可能使用 diffusion 或 flow-matching。

抽象流程：

```text
UE_i encoder_i(x)
  -> compressed code_i
  -> BS shared decoder with generated LoRA
  -> reconstructed CSI
```

## 固定 decoder 是否合理

固定 decoder 是合理的，而且符合通信系统部署逻辑。

在实际系统中，BS 端维护一个统一 decoder 比为每个厂家维护一套完整 decoder 更现实。否则会带来：

- 部署复杂度高。
- 厂家认证困难。
- 版本维护成本大。
- 新 UE 接入时需要更新 BS 主模型。

但需要区分两种固定：

```text
完全固定 decoder，没有任何适配
```

和：

```text
固定 base decoder 主干，只允许小规模 LoRA/adapter 适配
```

后者更合理。

建议最终目标设定为：

```text
base decoder architecture + base weights 固定
generated LoRA / adapter 可变
encoder 厂家侧可变
```

## 为什么前期仍应探索 decoder

虽然最终希望固定 decoder，但前期探索 decoder 很有必要。

原因是 LoRA 的效果强依赖 base decoder 的结构。如果 base decoder 对不同 encoder code 分布非常脆弱，那么后续 LoRA 生成器需要学习很复杂的补丁，训练难度会明显增加。

前期 decoder 选择不应只看端到端 NMSE，还应看：

```text
冻结 base decoder 后，少量 LoRA 参数能否有效适配不同 encoder。
```

这更接近最终任务。

## 推荐 base decoder 方向

建议优先考虑 hybrid 或 CNN residual decoder，而不是只依赖纯 TransNet decoder。

### 纯 TransNet decoder

```text
code -> Linear -> TransformerDecoder -> out
```

适合作为 baseline。

风险：

- 对 code token 分布较敏感。
- 不同厂家 encoder 产生的 code 分布可能不稳定。

### CNN residual decoder

```text
code
  -> LayerNorm
  -> Linear expand
  -> reshape
  -> CNN residual refinement
  -> out
```

优点：

- 对不同 encoder 更稳。
- 空间结构先验强。
- 后续 LoRA 插入位置清晰。

### Hybrid decoder

```text
code
  -> LayerNorm
  -> Linear to token sequence
  -> Transformer / token mixer
  -> reshape
  -> CNN residual refinement
  -> out
```

优点：

- Transformer 负责全局关系。
- CNN 负责局部 sparse CSI 结构。
- 更适合作为最终 base decoder 候选。

建议路线：

```text
先做 CNN residual baseline，再做 hybrid。
```

## LoRA 应该插在哪里

不建议一开始对 decoder 所有层都加 LoRA。

优先级如下。

### 1. code expand / fc_decoder

位置：

```text
code -> fc_decoder / Linear expand
```

这是最优先的 LoRA 位置。

原因：

- 不同厂家 encoder 的差异首先体现在 code 分布。
- expand 层直接决定 code 如何映射回 latent feature。
- 参数位置少，便于分析。

### 2. Transformer FFN

如果 decoder 使用 Transformer，可对：

```text
linear1
linear2
```

加 LoRA。

优点：

- 参数效率较高。
- 能调整特征变换方式。
- 比 attention projection 更容易训练。

### 3. Attention projection

可考虑对：

```text
q_proj
k_proj
v_proj
out_proj
```

加 LoRA。

适用于 hybrid 或 TransNet decoder。

风险：

- 生成 LoRA 参数更复杂。
- 适配空间更大，训练不稳定性更高。

### 4. CNN refinement 后几层

可对 CNN residual refinement 的：

```text
1x1 conv
最后几层 conv
```

加 LoRA 或轻量 adapter。

适合修正重建细节，但不适合承担主要 code 分布对齐任务。

## LoRA 生成条件：样本级还是厂家级

用户提出希望用 encoder 输出的压缩数据指导 LoRA 参数生成。

这可以做，但要区分：

```text
sample-conditioned LoRA
```

和：

```text
domain/vendor-conditioned LoRA
```

### 样本级 LoRA

形式：

```text
code -> generator -> LoRA weights
decoder(code; LoRA(code)) -> reconstruction
```

优点：

- 表达能力强。
- 每个样本都有动态适配。

风险：

- 时延和计算成本高。
- 容易把样本信息和厂家域信息混在一起。
- 泛化解释困难。
- 生成器可能过拟合训练样本。

### 厂家级 / encoder 级 LoRA

形式：

```text
UE id / encoder id / calibration codes -> domain embedding -> LoRA weights
```

优点：

- 更符合“多厂家泛化”问题定义。
- 适配的是厂家域差异，而不是每个样本的偶然差异。
- 更容易部署。
- 更容易做 unseen vendor 评估。

建议优先做厂家级或 encoder 级 LoRA。

## 如果没有厂家 ID

如果没有显式 UE/vendor ID，可以用少量 calibration code 估计 domain embedding：

```text
{code_1, code_2, ..., code_K}
  -> set encoder / pooling
  -> domain embedding
  -> LoRA generator
```

这比单个 sample code 更稳定。

可选聚合方式：

```text
mean pooling
attention pooling
Set Transformer
DeepSets
```

最简单 baseline：

```text
domain_embedding = mean(MLP(code_k))
```

## Diffusion / Flow-Matching 的位置

Diffusion 或 flow-matching 可以用于生成 LoRA 参数，但不建议一开始就上。

原因：

- LoRA 参数是高维结构化输出。
- 生成模型训练复杂。
- 如果简单 hypernetwork 无收益，复杂生成模型通常也难以带来可靠收益。

建议实验顺序：

```text
1. 固定 base decoder，测试不同 encoder 的 zero-shot 表现。
2. 每个 encoder 训练一个独立 LoRA，验证 LoRA 空间是否足够。
3. 用 MLP hypernetwork 从 domain embedding 预测 LoRA。
4. 如果 MLP hypernetwork 有收益，再考虑 diffusion / flow-matching。
```

这样可以逐步回答：

```text
LoRA 空间是否足够？
厂家差异是否能由 LoRA 表示？
LoRA 是否能由 code/domain embedding 预测？
复杂生成模型是否必要？
```

## 推荐实验阶段

### Phase 0：建立强 base decoder

训练多个 decoder 候选：

```text
TransNet decoder
CNN residual decoder
Hybrid decoder
```

比较：

```text
端到端 NMSE
跨 encoder 泛化 NMSE
冻结 decoder 后适配能力
```

### Phase 1：固定 base decoder，训练 per-encoder LoRA

设定：

```text
base decoder frozen
encoder_i fixed or trainable
LoRA_i trainable
```

目标：

```text
证明少量 LoRA 参数确实能适配不同 encoder/vendor。
```

如果 per-encoder LoRA 没有明显收益，说明：

- LoRA 插入位置不合适，或
- base decoder 不适合低秩适配，或
- encoder code 差异过大。

此时不应直接进入 diffusion/flow-matching。

### Phase 2：LoRA generator baseline

先用简单 MLP：

```text
domain embedding -> MLP -> LoRA A/B matrices
```

或：

```text
code statistics -> MLP -> LoRA
```

目标：

```text
证明 LoRA 参数可以从 encoder/code 信息中预测。
```

### Phase 3：Diffusion / Flow-Matching

在 Phase 2 有正结果后，再考虑：

```text
domain embedding -> diffusion/flow matching -> LoRA distribution
```

适用场景：

- LoRA 参数存在多模态。
- 一个 vendor/domain 下有多种可行 adapter。
- 希望生成具有不确定性的 adapter。

## 评价指标建议

除 NMSE 外，建议增加以下评估：

### Cross-vendor zero-shot

```text
train vendors: A, B, C
test vendor: D
decoder frozen
no LoRA or generated LoRA
```

### Few-shot calibration

```text
给未知 vendor K 个 calibration samples
生成 domain embedding
预测 LoRA
测试剩余 samples
```

### LoRA 参数效率

记录：

```text
LoRA rank
可训练参数量
NMSE gain
推理开销
```

### 稳定性

比较多个随机种子：

```text
seed 42
seed 2026
seed 3407
```

避免把单次训练波动误认为泛化提升。

## 最终建议

最终固定 decoder 是合理的，但建议固定的是：

```text
base decoder architecture + base decoder weights
```

同时允许：

```text
small generated LoRA / adapter
```

优先路线：

```text
Hybrid base decoder:
LayerNorm(code)
  -> Linear expand
  -> lightweight Transformer / token mixer
  -> reshape
  -> CRNet/CLNet-style CNN residual refinement
  -> linear output

LoRA first:
fc_expand + Transformer FFN

Generator first:
domain embedding -> MLP -> LoRA

Generator later:
domain embedding -> diffusion / flow-matching -> LoRA
```

核心判断标准：

```text
不是哪个 decoder 端到端最强，
而是哪个 fixed base decoder 最容易被低秩、可生成的 adapter 适配到新厂家。
```

