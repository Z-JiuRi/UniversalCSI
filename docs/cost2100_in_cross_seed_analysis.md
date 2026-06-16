# COST2100 In 场景跨 Seed Encoder/Decoder 适配实验分析

本文总结 `exps/COST2100/in` 下已有实验结果，重点分析：

- 不同 seed 独立训练出的 encoder 能否直接接入另一个 seed 的 decoder。
- adapter、LoRA、只训练 `fc_decoder` 等后验补丁为什么效果有限。
- 哪条路线从实验上已经证明有效。

结论先行：

> 当前结果表明，独立 seed 训练出的 autoencoder latent/code space 不可直接互换。
> 后验加小模块可以从完全崩溃恢复一部分，但明显卡在 `-20 ~ -21 dB` 附近；
> 真正稳定有效的是在训练阶段就固定目标 decoder，让 encoder 从一开始学习目标
> decoder 的码字坐标系。

## 实验目录概览

相关实验主要分布在：

```text
exps/COST2100/in/
├── seed42/                 独立训练 baseline
├── seed2026/               独立训练 baseline
├── seed3407/               独立训练 baseline
├── frozen_decoder/         冻结目标 decoder 后重新训练 encoder
├── adapter/                冻结 encoder/decoder 后训练中间 adapter
├── lora/                   对 hybrid decoder 的 token_projection 做 LoRA
└── fc_decoder/             只训练 TransNetDecoder.fc_decoder
```

默认数据路径为：

```text
/storage/hujiacong/zxd/datasets/cost2100/in_train.pt
/storage/hujiacong/zxd/datasets/cost2100/in_val.pt
/storage/hujiacong/zxd/datasets/cost2100/in_test.pt
```

默认模型维度：

```text
channel = 2
nt = 32
nc = 32
cr = 4
input_dim = 2048
code_dim = 512
```

以下 NMSE 均来自对应 `run.log`，单位为 dB，数值越低越好。

## 1. 独立训练 Baseline 是正常的

首先看每个 seed 独立训练得到的完整 autoencoder。它们本身并没有明显训练失败。

### TransNet Encoder + Hybrid Decoder

| 实验 | Seed | Best NMSE | Final/Test NMSE | 说明 |
|---|---:|---:|---:|---|
| `seed42/transnet_hybrid` | 42 | `-28.407` | `-28.407` | 独立训练 |
| `seed2026/transnet_hybrid` | 2026 | `-28.207` | `-25.870` | best 正常，final 有回退 |
| `seed3407/transnet_hybrid` | 3407 | `-27.562` | `-27.562` | 独立训练 |

### TransNet Encoder + TransNet Decoder

| 实验 | Seed | Best NMSE | Final/Test NMSE | 说明 |
|---|---:|---:|---:|---|
| `seed42/transnet_transnet` | 42 | `-28.126` | `-28.126` | 独立训练 |
| `seed2026/transnet_transnet` | 2026 | `-28.180` | `-28.180` | 独立训练 |
| `seed3407/transnet_transnet` | 3407 | `-28.520` | `-28.520` | 独立训练 |

这说明问题不是某个 seed 模型自身学不好，而是不同 seed 的 encoder/decoder 组合后
码字空间不兼容。

## 2. 独立 Seed Encoder 直接接独立 Seed Decoder 会严重崩溃

跨 seed 错配的初始结果非常差：

| 场景 | 组合 | 微调前 NMSE |
|---|---|---:|
| adapter/hybrid | seed3407 encoder + seed42 hybrid decoder | `+23.854` |
| fc_decoder/transnet | seed3407 encoder + seed42 transnet decoder | `+15.461` |
| LoRA/hybrid | seed42 encoder + seed2026 hybrid decoder | `+28.634` |

这些正数 NMSE 说明错配不是小幅分布偏移，而是 decoder 基本读不懂另一个 seed 的
码字表示。

直观理解：

```text
独立训练 autoencoder 时：

encoder_A(x) -> code_A -> decoder_A(code_A) -> x
encoder_B(x) -> code_B -> decoder_B(code_B) -> x

重建损失只要求 decoder_A 能读 code_A、decoder_B 能读 code_B，
并不要求 code_A 和 code_B 使用同一个坐标系。
```

因此两个 seed 可能学到等价但互不兼容的 latent space，例如旋转、缩放、置换、
非线性扭曲，甚至不同的语义分配。

## 3. 后验 Adapter 能救一部分，但天花板明显不足

典型独立 seed 错配实验：

```text
pretrained_encoder = exps/COST2100/in/seed3407/transnet_hybrid/checkpoints/best_nmse.pth
pretrained_decoder = exps/COST2100/in/seed42/transnet_hybrid/checkpoints/best_nmse.pth
code_adapter = true
```

结果如下：

| 实验 | Teacher Code | Code Loss | 微调前 NMSE | Best NMSE | Final/Test NMSE |
|---|---|---|---:|---:|---:|
| `adapter/seed3407/transnet3407_hybrid42` | 否 | 无 | `+23.854` | `-20.646` | `-20.645` |
| `adapter/seed3407/transnet3407_hybrid42_learnable_lambda` | 是 | 可学习 lambda | `+23.854` | `-20.393` | `-20.393` |
| `adapter/seed3407/transnet3407_hybrid42_mlp_code_only` | 是 | code-only, `lambda=0.1` | `+23.854` | `-21.059` | `-21.059` |
| `adapter/seed3407/transnet3407_hybrid42_0.1` | 是 | 固定 `lambda=0.1` | `+23.854` | `-0.038` | `-0.027` |
| `adapter/seed3407/transnet3407_hybrid42_new_0.1` | 是 | 固定 `lambda=0.1` | `+23.854` | `-17.501` | `-17.500` |
| `adapter/seed3407/transnet3407_hybrid42_new_0.5` | 是 | 固定 `lambda=0.5` | `+23.854` | `-17.476` | `-17.475` |
| `adapter/seed3407/transnet3407_hybrid42_new_true` | 是 | code-only | `+23.854` | `-17.470` | `-17.469` |

观察：

- Adapter 确实能把模型从 `+23.854 dB` 拉回负数 NMSE。
- 但最好也只有约 `-21 dB`，离原始 baseline 的 `-28 dB` 差距很大。
- 固定 teacher code loss 容易伤害重建目标，`lambda=0.1` 的一组甚至只到
  `-0.027 dB`。
- code-only 能对齐一部分码字，但无法保证 decoder 重建达到原始模型水平。

这说明后验 adapter 更像是在做翻译补丁，而不是从根本上让 encoder 学会目标 decoder
需要的表示。

## 4. LoRA 只改 token_projection 明显不够

LoRA 实验配置：

```text
encoder = transnet
decoder = hybrid
lora_component = token_projection
pretrained_encoder = seed42/transnet_hybrid/base/checkpoints/best_nmse.pth
pretrained_decoder = seed2026/transnet_hybrid/checkpoints/best_nmse.pth
```

结果：

| 实验 | Rank | Alpha | 微调前 NMSE | Best NMSE | Final/Test NMSE |
|---|---:|---:|---:|---:|---:|
| `lora/seed2026/transnet_hybrid_token_projection_rank8_alpha16` | 8 | 16 | `+28.634` | `-1.733` | `-1.732` |
| `lora/seed2026/transnet_hybrid_token_projection_rank16_alpha32` | 16 | 32 | `+28.634` | `-2.720` | `-2.720` |
| `lora/seed2026/transnet_hybrid_token_projection_rank32_alpha64` | 32 | 64 | `+28.634` | `-4.883` | `-4.879` |
| `lora/seed2026/transnet_hybrid_token_projection_rank64_alpha128` | 64 | 128 | `+28.634` | `-8.043` | `-8.042` |

观察：

- rank 增大时性能单调改善，但 rank64 仍只有 `-8.043 dB`。
- 只改 `HybridDecoder.token_projection` 的容量或位置不足以解决跨 seed latent
  mismatch。
- 继续单纯堆 LoRA rank，性价比很低。

结论：LoRA 这条线可以证明“projection 层参与了错配”，但它不是解决方案。

## 5. 只训练 TransNetDecoder.fc_decoder 也有天花板

实验配置：

```text
encoder = transnet
decoder = transnet
pretrained_encoder = seed3407/transnet_transnet/checkpoints/best_nmse.pth
pretrained_decoder = seed42/transnet_transnet/checkpoints/best_nmse.pth
train_fc_decoder = true
```

结果：

| 实验 | Teacher Code | 训练目标 | 微调前 NMSE | Best NMSE | Final/Test NMSE |
|---|---|---|---:|---:|---:|
| `fc_decoder/seed3407/transnet3407_transnet42_recon_only` | 否 | 重建 MSE | `+15.461` | `-21.627` | `-21.627` |
| `fc_decoder/seed3407/transnet3407_transnet42_lambda` | 是 | 重建 + 可学习 code loss | `+15.461` | `-21.229` | `-21.229` |
| `fc_decoder/seed3407/transnet3407_transnet42_code_only` | 是 | code-only | `+15.461` | `-14.991` | `-14.984` |

观察：

- 只训 `fc_decoder` 比 LoRA 强，能到 `-21.6 dB`。
- 但依然明显低于完整 baseline 的 `-28 dB`。
- code-only 对最终重建不够友好。

这说明 TransNet decoder 的第一层线性扩展确实能适配一部分 code space，但后续
Transformer decoder 层仍然依赖原 seed 的 token 分布。

## 6. 真正有效：冻结目标 Decoder，从头训练 Encoder

`frozen_decoder` 实验是最关键的证据。它固定目标 decoder，只训练新 encoder。

### 固定 Seed42 Hybrid Decoder，训练 TransNet Encoder

| 实验 | Encoder Seed | Best NMSE | Final/Test NMSE |
|---|---:|---:|---:|
| `frozen_decoder/seed0/transnet_hybrid` | 0 | `-28.620` | `-28.620` |
| `frozen_decoder/seed1/transnet_hybrid` | 1 | `-28.620` | `-28.620` |
| `frozen_decoder/seed2026/transnet_hybrid` | 2026 | `-28.616` | `-28.615` |
| `frozen_decoder/seed3407/transnet_hybrid` | 3407 | `-28.613` | `-28.613` |
| `frozen_decoder/seed42/transnet_hybrid` | 42 | `-28.656` | `-28.656` |
| `frozen_decoder/seed666/transnet_hybrid` | 666 | `-28.628` | `-28.628` |
| `frozen_decoder/seed999/transnet_hybrid` | 999 | `-28.619` | `-28.618` |

### 固定 Seed42 TransNet Decoder，训练 TransNet Encoder

| 实验 | Encoder Seed | Best NMSE | Final/Test NMSE |
|---|---:|---:|---:|
| `frozen_decoder/seed0/transnet_transnet` | 0 | `-27.907` | `-27.907` |
| `frozen_decoder/seed1/transnet_transnet` | 1 | `-27.988` | `-27.988` |
| `frozen_decoder/seed2026/transnet_transnet` | 2026 | `-27.985` | `-27.985` |
| `frozen_decoder/seed3407/transnet_transnet` | 3407 | `-27.972` | `-27.972` |
| `frozen_decoder/seed42/transnet_transnet` | 42 | `-28.007` | `-28.007` |
| `frozen_decoder/seed666/transnet_transnet` | 666 | `-27.976` | `-27.976` |
| `frozen_decoder/seed999/transnet_transnet` | 999 | `-28.025` | `-28.025` |

观察：

- 换多个 encoder seed 后，结果都非常稳定。
- Hybrid decoder 固定时，TransNet encoder 基本都到 `-28.6 dB`。
- TransNet decoder 固定时，也稳定在 `-28 dB` 左右。

这组实验说明：

```text
decoder 本身可以跨 seed 使用；
关键是 encoder 必须在训练过程中被目标 decoder 约束。
```

换句话说，失败的不是“decoder 不能泛化”，而是“独立训练完的 encoder code space
已经不是目标 decoder 的输入语言”。

## 7. 注意两个容易误读的 Adapter Control

有两组 adapter 结果看起来非常好：

| 实验 | 微调前 NMSE | Best NMSE | Final/Test NMSE |
|---|---:|---:|---:|
| `adapter/seed3407/transnet3407_hybrid42_no_teacher_code` | `-28.174` | `-28.207` | `-28.207` |
| `adapter/seed3407/transnet3407_hybrid42_learnable_lambda_joint` | `-28.174` | `-28.209` | `-28.209` |

但这两组使用的是：

```text
pretrained_encoder = frozen_decoder/seed3407/transnet_hybrid/checkpoints/best_nmse.pth
pretrained_decoder = frozen_decoder/seed42/transnet_hybrid/checkpoints/best_nmse.pth
```

这些 encoder 本来就是在同一个目标 decoder 约束下训练出来的，因此初始 NMSE 已经是
`-28.174 dB`。这不是“独立 seed encoder 通过 adapter 成功对齐独立 seed decoder”，
而是“已经对齐过的 encoder/decoder 再加 adapter 基本保持性能”。

这个 control 的意义是：

- 如果 encoder 训练时已经进了目标 decoder 的坐标系，后续 adapter 不会破坏太多。
- 它不能证明后验 adapter 能解决独立 seed 错配。

## 8. 跨 Encoder 架构的 Frozen Decoder 结果

固定 seed42 hybrid decoder 后，不同 encoder 架构重新训练的表现也比较稳定：

| Encoder | Seed42 独立训练 Hybrid | Frozen Decoder Seed3407 | Frozen Decoder Seed2026 |
|---|---:|---:|---:|
| `csinet` | `-22.447` | `-20.897` | `-22.574` |
| `cnn` | `-25.616` | `-25.726` | `-25.270` |
| `cbam_cnn` | `-24.685` | `-25.312` | `-24.568` |
| `crnet` | `-24.822` | `-20.498` | `-21.158` |
| `clnet` | `-28.412` | `-28.960` | `-28.785` |
| `resnet` | `-24.880` | `-25.272` | `-24.592` |
| `dscnn` | `-22.593` | `-22.808` | `-21.258` |
| `convnext` | `-29.050` | `-29.179` | `-29.037` |
| `mlp_mixer` | `-28.853` | `-29.020` | `-28.884` |
| `attention_cnn` | `-26.675` | `-26.857` | `-26.385` |
| `swin` | `-28.688` | `-29.291` | `-28.946` |
| `mlp_ae` | `-25.833` | `-26.312` | `-26.192` |
| `sparse_resnet` | `-25.659` | `-26.019` | `-25.338` |
| `transnet` | `-28.407` | `-28.613` | `-28.616` |

观察：

- `convnext`、`mlp_mixer`、`swin`、`clnet`、`transnet` 与固定 decoder 配合很好。
- `crnet` 在 frozen decoder 下明显退化，说明部分 encoder 架构可能不适合该 decoder
  的输入约束或当前训练超参。
- 这进一步支持“训练时对齐 decoder”这条路线，而不是后验拼接。

## 9. 当前现象的核心原因

Autoencoder 的重建损失只约束端到端结果：

```text
decoder(encoder(x)) ≈ x
```

但它不约束 `encoder(x)` 必须服从某个全局固定坐标系。于是不同 seed 可能学习到：

- 不同尺度的 code。
- 不同维度排列或混合方式。
- 不同 token 分布。
- decoder 内部层依赖的特定 code 统计。
- 等价但不可交换的 latent 表示。

这类不唯一性在自编码器里很常见。只要 encoder 和 decoder 成对训练，它们内部怎么
约定都可以；但一旦把不同 seed 的 encoder 和 decoder 拆开重组，约定就失效。

后验 adapter/LoRA/fc_decoder 的困难在于：

```text
它们试图在不改变 encoder 主体、不改变 decoder 主体的前提下，
把一个已经成型的 code space 翻译成另一个 decoder 期望的 code space。
```

从已有结果看，这个翻译可以学到一部分，但没有足够能力或约束恢复到原始 `-28 dB`
水平。

## 10. 结论

基于当前实验，建议停止把主要精力放在以下方向：

- 继续单纯增加 adapter 复杂度。
- 继续只调 LoRA rank。
- 继续只训练单个 decoder projection/fc 层。
- 用固定较大的 teacher code MSE 强行约束 code。

这些路线已经有比较充分的负面证据：能改善，但离目标性能差距大。

更可靠的方向是：

```text
目标 decoder 固定
encoder 在训练期间直接接入目标 decoder
用重建损失训练 encoder
```

已有 `frozen_decoder` 实验已经证明，这条路线可以稳定恢复到 `-28 dB` 级别。

## 11. 如何在训练时显式规定共同码字空间

所谓“显式规定共同码字空间”，核心是不要只优化：

```text
decoder(encoder(x)) ≈ x
```

还要额外约束 `encoder(x)` 本身落到一个可复用、可解释、跨 seed 一致的 code space。
下面按约束强度从轻到重列出可选方案。

### 方案 1：固定 Teacher Code，所有 Encoder 蒸馏到同一套码字

选一个性能强、后续希望作为标准的 teacher，例如：

```text
teacher = seed42/transnet_hybrid
teacher_code(x) = teacher.encoder(x)
```

训练任意新 encoder 时，同时优化重建损失和 teacher code 对齐损失：

```text
code_new = encoder_new(x)
recon = decoder_target(code_new)

loss = MSE(recon, x)
     + lambda(t) * MSE(code_new, teacher_code(x))
```

这就是最直接的共同码字空间定义：所有 encoder 都被拉向同一个 teacher encoder 的
输出坐标系。

注意事项：

- `lambda` 不能一开始太大。已有实验里固定 `lambda=0.1` 会明显伤害重建。
- 建议从 `1e-4 ~ 1e-3` 量级试起，而不是直接用 `0.1`。
- 推荐 warmup，让模型先学会目标 decoder 的重建，再逐步加 code 约束。

一个更稳的 schedule：

```text
epoch 1   ~ 100: lambda = 0
epoch 100 ~ 250: lambda 从 0 线性升到 1e-4
epoch 250 ~ 400: lambda = 1e-4
```

如果 `1e-4` 对 code 对齐太弱，再试：

```text
3e-4
1e-3
```

不建议一开始就回到 `0.1`。

### 方案 2：固定目标 Decoder，用 Decoder 隐式定义共同语言

这是当前实验中已经证明最有效的路线：

```text
decoder_target = seed42 decoder
freeze(decoder_target)
train encoder_new

loss = MSE(decoder_target(encoder_new(x)), x)
```

这里共同码字空间不是由 teacher code 直接定义，而是由固定 decoder 的可读输入空间
隐式定义。所有 encoder 都必须学习输出 `decoder_target` 能读懂的 code。

如果想让它更显式，可以在此基础上加入很轻的 teacher code loss：

```text
loss = MSE(decoder_target(encoder_new(x)), x)
     + lambda(t) * MSE(encoder_new(x), encoder_target(x))
```

其中 `encoder_target` 通常取与 `decoder_target` 原本配套训练的 encoder。

推荐训练顺序：

```text
stage 1: frozen decoder + encoder_new，recon-only 训练到接近 -28 dB
stage 2: 从 stage 1 checkpoint 继续，加入很小的 teacher code loss
stage 3: 检查 NMSE 是否不掉，同时 code 是否更接近 teacher
```

这样可以避免 code loss 在训练早期压过重建目标。

### 方案 3：共享 Bottleneck 规范化和分布约束

让所有 encoder 的 bottleneck 输出经过相同规范化，例如：

```text
code_raw = encoder_backbone(x)
code = LayerNorm(code_raw)
```

或：

```text
code = normalize(code_raw, dim=-1)
```

还可以加入 batch-level 分布约束：

```text
mean_loss = ||mean(code) - 0||^2
var_loss  = ||std(code) - 1||^2
cov_loss  = ||Cov(code) - I||^2

loss = recon_loss
     + alpha * mean_loss
     + beta  * var_loss
     + gamma * cov_loss
```

这类方法能减少不同 seed 之间的尺度、均值、方差和协方差漂移。它的优点是不用选
teacher；缺点是只能约束统计分布，不能保证每个维度的语义完全对齐。

适用场景：

- code space 主要存在尺度/方差漂移。
- decoder 对输入分布统计很敏感。
- 不希望强行把所有 encoder 拉到某一个 teacher encoder 的具体数值。

### 方案 4：样本级对比学习锚定

如果不希望用逐维 MSE 强行匹配 teacher code，可以改用样本级对比学习：

```text
positive: code_new(x_i), code_teacher(x_i)
negative: code_new(x_i), code_teacher(x_j), j != i
```

目标是同一个样本的 code 靠近，不同样本的 code 分开。总损失可以写成：

```text
loss = recon_loss + lambda(t) * InfoNCE(code_new, code_teacher)
```

相比 MSE，InfoNCE 更关注样本身份和局部结构，不强求每个维度数值完全一样。它适合
teacher code 存在等价旋转/尺度差异、但样本间邻域结构有价值的情况。

风险：

- 实现和调参比 MSE 复杂。
- batch size、temperature、负样本数量都会影响结果。
- 最终 decoder 是否受益需要实验验证。

### 方案 5：共享 Codebook 或离散码字

最强的共同码字空间约束是所有 encoder 都输出同一个 codebook 中的离散 index 或共享
embedding：

```text
encoder(x) -> nearest codebook vectors -> decoder
```

这样不同 seed 的 encoder 只能使用同一套码本，code space 天然统一。

优点：

- 可插拔性最强。
- 不同 encoder 的输出空间被明确限制。

缺点：

- 实现复杂度高。
- 可能牺牲 NMSE。
- 需要处理 codebook collapse、commitment loss、码本利用率等问题。

这更像长期研究方向，不建议作为当前第一优先级。

### 本项目最推荐的共同码字空间方案

结合现有实验，最建议先做这个版本：

```text
decoder_target = seed42 decoder
encoder_target = seed42 encoder
teacher_code = encoder_target(x)

freeze(decoder_target)
train(encoder_new)

loss = MSE(decoder_target(encoder_new(x)), x)
     + lambda(t) * MSE(encoder_new(x), teacher_code)
```

推荐参数：

```text
epoch 1   ~ 100: lambda = 0
epoch 100 ~ 250: lambda 从 0 线性升到 1e-4
epoch 250 ~ 400: lambda = 1e-4
```

如果 code 对齐效果太弱，再试 `3e-4` 和 `1e-3`。每次都要同时看：

- test NMSE 是否保持在 `-28 dB` 附近。
- `MSE(encoder_new(x), teacher_code)` 是否下降。
- linear/MLP probe 是否更容易把不同 encoder code 对齐。

最重要的是：**共同码字空间必须在训练闭环里定义**。只靠 autoencoder 重建损失，
模型没有理由让不同 seed 的 code 使用同一种坐标系。

## 12. 建议下一步实验

### 实验 A：加载已有独立 Encoder，解冻 Encoder 继续适配目标 Decoder

目的：判断已有独立 seed encoder 权重是否能作为初始化，而不是必须从头训练。

配置：

```text
pretrained_encoder = seed3407/transnet_hybrid/checkpoints/best_nmse.pth
pretrained_decoder = seed42/transnet_hybrid/checkpoints/best_nmse.pth
freeze decoder = true
unfreeze encoder = true
train adapter = false
```

当前代码的 `--pretrained_encoder` 会自动 freeze encoder，因此需要新增一个模式，例如：

```text
--pretrained_encoder_no_freeze
```

或增加：

```text
--unfreeze_encoder_after_load
```

预期判断：

- 如果能从 `+23.854 dB` 恢复到接近 `-28 dB`，说明已有 encoder 权重可作为初始化。
- 如果恢复慢或上不去，则从头训练 encoder 可能更干净。

### 实验 B：冻结目标 Decoder，从头训练 Encoder，加入轻量 Code Regularization

目的：在不破坏重建的情况下，让 code 更接近目标 seed 的 code 分布。

建议损失：

```text
loss = recon_loss(decoder_target(encoder_new(x)), x)
     + lambda(t) * code_loss(encoder_new(x), encoder_target(x))
```

建议：

- `lambda` 从 0 warmup 到很小值。
- 不要一开始固定 `lambda=0.1`。
- 先保证 recon loss 能到 `-28 dB`，再看 code regularization 是否改善跨模型可解释性。

### 实验 C：训练 Universal Decoder

目的：让一个 decoder 接受多个 seed/架构的 code space，而不是要求每个 encoder 都进入
单一坐标系。

训练数据：

```text
code_from_encoder_seed42
code_from_encoder_seed2026
code_from_encoder_seed3407
...
```

训练方式：

```text
decoder_universal(code_i) -> x
```

风险：

- 如果不同 code space 差异太大，单 decoder 需要更大容量。
- 需要清楚标记 code 来源，可能需要 source embedding 或 conditional adapter。

### 实验 D：做 Linear/MLP Code Probe

目的：判断两个 seed 的 code space 差异是线性变换为主，还是强非线性。

步骤：

```text
1. 导出同一批 train 数据上的 code_A = encoder_A(x)
2. 导出同一批 train 数据上的 code_B = encoder_B(x)
3. 训练 Linear(code_A -> code_B)
4. 测试 decoder_B(Linear(code_A))
5. 再训练两层 MLP(code_A -> code_B) 做对照
```

解释：

- Linear probe 能到接近 `-28 dB`：主要是线性坐标错位。
- MLP probe 明显好于 Linear：存在非线性错位。
- MLP 仍明显差：encoder_A 的 code 对 decoder_B 来说缺少必要结构，后验映射价值有限。

## 13. 当前最推荐路线

按优先级排序：

1. **继续使用 frozen decoder 训练 encoder**。这是当前唯一稳定达到 `-28 dB` 级别的
   跨 seed 方案。
2. **新增“加载 encoder 但不冻结”的适配模式**，测试已有 encoder 是否能 fine-tune
   到目标 decoder。
3. **谨慎使用 teacher code loss**，先从 recon-only 开始，再用小权重/warmup。
4. **暂停 LoRA-token-projection 方向**，除非准备扩大可训练范围到 decoder 多层或
   encoder+decoder 联合适配。
5. **做 linear/MLP probe**，用它判断 code space mismatch 的可映射性，避免继续盲目
   加模块。

## 14. 一句话总结

当前实验已经说明：不同 seed 独立训练出的 encoder/decoder 不是可插拔模块；它们的
code space 是成对约定出来的。后验补丁能缓解错配，但无法稳定恢复到原始性能。
想解决跨 seed decoder 适配，应该把目标 decoder 放进 encoder 的训练闭环，而不是训练
完以后再靠小模块翻译码字。
