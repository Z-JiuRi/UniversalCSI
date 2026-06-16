# CSI 反馈模型迁移与码字空间适配阶段性汇报

本文面向组会汇报，联合三组已有实验材料，梳理当前结果、实验困境、初步机理解释和后续解决路线。

证据来源：

- `UniversalCSI` / COST2100 in：
  - `/storage/hujiacong/zxd/Huawei/UniversalCSI/docs/cost2100_in_cross_seed_analysis.md`
  - `/storage/hujiacong/zxd/Huawei/UniversalCSI/exps/COST2100/in`
- `TransNet` / WAIRD seed797：
  - `/home/hujiacong/zxd/Huawei/TransNet/docs/waird_seed797_experiment_analysis.md`
  - `/home/hujiacong/zxd/Huawei/TransNet/exps/WAIRD/seed797`
- `TransCSI` / WAIRD seed42：
  - `/home/hujiacong/zxd/Huawei/TransCSI/WAIRD_seed796_seed797_lora_analysis.md`
  - `/home/hujiacong/zxd/Huawei/TransCSI/exps/WAIRD/seed42`

本文的核心观点：

> 现有实验不是简单说明“LoRA 不好”或“adapter 不够强”，而是在三个不同侧面反复指向同一个问题：
> CSI 自编码反馈模型的压缩码字空间不是天然可交换、可迁移、可插拔的。  
> 如果训练目标只约束端到端重建，encoder 和 decoder 会形成私有协议；跨 seed、跨场景、跨架构时，
> 这种私有协议会成为主要瓶颈。后验加小模块可以缓解，但很难替代训练时对码字空间和 decoder 输入语言的显式约束。

## 1. 三组实验分别在回答什么问题

### 1.1 COST2100 in：不同 seed 的 encoder/decoder 能否拆开重组

UniversalCSI 的 COST2100 in 实验主要考察同一数据集、同一模型族下，不同 seed 训练得到的
encoder 和 decoder 是否可插拔。

关键实验：

- 独立 seed baseline：
  - `seed42/transnet_hybrid`
  - `seed2026/transnet_hybrid`
  - `seed3407/transnet_hybrid`
  - `seed42/transnet_transnet`
  - `seed3407/transnet_transnet`
- 后验补丁：
  - adapter
  - LoRA
  - 只训练 `fc_decoder`
- 正向对照：
  - frozen decoder 后重新训练 encoder

这组实验回答的是：**同域、同任务、同结构下，仅仅 seed 不同，latent/code space 是否仍然可交换？**

答案很明确：不能。

### 1.2 TransNet WAIRD seed797：base 模型迁移到新场景后，PEFT 能否替代 full fine-tuning

TransNet 的 WAIRD seed797 实验主要考察 base 场景训练后的模型迁移到 scenario_2 不同场景：

- `01105`
- `06401`
- `09957`

关键实验：

- base 训练
- direct test
- full fine-tuning
- LoRA / PEFT
- teacher recon
- teacher code

这组实验回答的是：**跨场景迁移时，少量参数微调和 teacher 监督能否接近 full fine-tuning？**

答案是：不同场景差异很大；`01105` full fine-tuning 明显有效，但多数 LoRA 仍有明显差距；
`06401` 很难适配；teacher code 在当前设置下通常会伤害性能。

### 1.3 TransCSI WAIRD seed42：token 化结构下，压缩侧和重建侧谁更关键

TransCSI 的 WAIRD seed42 实验主要围绕 token 化压缩结构：

- `compress_token=token_size`
- `compress_token=token_num`
- direct test
- full fine-tuning
- 冻结 compression side
- 冻结 reconstruction side
- token compressor / encoder FFN LoRA

这组实验回答的是：**换成 TransCSI token 化结构后，场景迁移主要需要改压缩侧还是重建侧？**

结果显示：在 `01105` 上，full fine-tuning 最强；只训练重建侧明显优于只训练压缩侧；
只改 token compressor 或 encoder FFN 的 LoRA 仍然不足。

## 2. 表层结果：现有实验给出的直接事实

### 2.1 COST2100：独立训练模型本身正常，但跨 seed 重组严重崩溃

#### 2.1.1 独立 seed baseline

| 模型 | Seed | Best NMSE | Final/Test NMSE | 说明 |
|---|---:|---:|---:|---|
| `transnet_hybrid` | 42 | `-28.407` | `-28.407` | 独立训练正常 |
| `transnet_hybrid` | 2026 | `-28.207` | `-25.870` | best 正常，final 回退 |
| `transnet_hybrid` | 3407 | `-27.562` | `-27.562` | 独立训练正常 |
| `transnet_transnet` | 42 | `-28.126` | `-28.126` | 独立训练正常 |
| `transnet_transnet` | 2026 | `-28.180` | `-28.180` | 独立训练正常 |
| `transnet_transnet` | 3407 | `-28.520` | `-28.520` | 独立训练正常 |

说明：每个 seed 自己成对训练的 encoder+decoder 都可以达到约 `-28 dB`，不是单个模型训练失败。

#### 2.1.2 跨 seed encoder/decoder 全组合测试：直接重组完全崩溃

下表把同架构、不同 seed 的 encoder 和 decoder 拆开重组。这里不做任何微调，只测试
`decoder_j(encoder_i(x))` 的重建结果。

| Encoder | Decoder | MSE Loss | NMSE |
|---|---|---:|---:|
| `seed42_transnet_transnet encoder` | `seed2026_transnet_transnet decoder` | `4.8928e-02` | `20.8398` |
| `seed3407_transnet_transnet encoder` | `seed2026_transnet_transnet decoder` | `4.1175e-02` | `19.8961` |
| `seed2026_transnet_transnet encoder` | `seed42_transnet_transnet decoder` | `8.8851e-03` | `12.8475` |
| `seed3407_transnet_transnet encoder` | `seed42_transnet_transnet decoder` | `1.5894e-02` | `16.0782` |
| `seed2026_transnet_transnet encoder` | `seed3407_transnet_transnet decoder` | `3.6063e-02` | `19.4481` |
| `seed42_transnet_transnet encoder` | `seed3407_transnet_transnet decoder` | `3.6028e-02` | `19.1015` |
| `seed42_transnet_hybrid encoder` | `seed2026_transnet_hybrid decoder` | `3.3000e-01` | `29.4040` |
| `seed3407_transnet_hybrid encoder` | `seed2026_transnet_hybrid decoder` | `1.9973e-01` | `27.1544` |
| `seed2026_transnet_hybrid encoder` | `seed42_transnet_hybrid decoder` | `4.8094e-01` | `31.0447` |
| `seed3407_transnet_hybrid encoder` | `seed42_transnet_hybrid decoder` | `1.0977e-01` | `24.3450` |
| `seed2026_transnet_hybrid encoder` | `seed3407_transnet_hybrid decoder` | `1.4627e-01` | `25.6867` |
| `seed42_transnet_hybrid encoder` | `seed3407_transnet_hybrid decoder` | `4.4318e-02` | `20.4196` |

按架构聚合：

| 架构 | 组合数 | NMSE 范围 | 平均 NMSE | 现象 |
|---|---:|---:|---:|---|
| `transnet_transnet` | 6 | `12.8475 ~ 20.8398` | `18.0352` | 全部为正，已严重失配 |
| `transnet_hybrid` | 6 | `20.4196 ~ 31.0447` | `26.3424` | 比 transnet decoder 更敏感，崩溃更重 |
| 全部跨 seed 组合 | 12 | `12.8475 ~ 31.0447` | `22.1888` | 没有任何组合接近可用 |

对比上一节可见：成对训练的同 seed 模型能达到约 `-28 dB`，但仅仅把 encoder 和 decoder
跨 seed 重组，NMSE 立刻变成 `+12.8 ~ +31.0 dB`。这说明问题不是模型容量不足，也不是某个
seed 训练坏了，而是独立训练形成的 code space 协议完全不兼容。

#### 2.1.3 后续微调实验中的典型错配起点

后续 adapter、LoRA、`fc_decoder` 微调实验只选取了部分错配组合作为起点：

| 场景 | 组合 | 微调前 NMSE | 说明 |
|---|---|---:|---|
| adapter/hybrid | seed3407 encoder + seed42 hybrid decoder | `+23.854` | 与上表同类，评估口径略有差异 |
| fc_decoder/transnet | seed3407 encoder + seed42 transnet decoder | `+15.461` | 只训练 `decoder.fc_decoder` 的起点 |
| LoRA/hybrid | seed42 encoder + seed2026 hybrid decoder | `+28.634` | 只训练 hybrid `token_projection` LoRA 的起点 |

说明：这不是小幅分布偏移，而是 decoder 基本读不懂另一个 seed 的 code。微调实验需要先从
正 NMSE 拉回可用区间，本身已经说明后验补丁承担的是“翻译私有协议”的任务。

#### 2.1.4 后验补丁的上限

| 方法 | 微调前 NMSE | Best NMSE | Final/Test NMSE | 结论 |
|---|---:|---:|---:|---|
| adapter，无 teacher code | `+23.854` | `-20.646` | `-20.645` | 能救，但离 `-28 dB` 很远 |
| adapter，可学习 lambda | `+23.854` | `-20.393` | `-20.393` | 与无 teacher 接近 |
| adapter，code-only | `+23.854` | `-21.059` | `-21.059` | 最好仍约 `-21 dB` |
| adapter，固定 `lambda=0.1` | `+23.854` | `-0.038` | `-0.027` | 强 teacher code 明显伤害 |
| LoRA rank64 | `+28.634` | `-8.043` | `-8.042` | 只改 token projection 远远不够 |
| 只训 `fc_decoder`，recon-only | `+15.461` | `-21.627` | `-21.627` | 比 LoRA 强，但仍有天花板 |

说明：后验加模块并不是完全无效，但性能停在 `-20 ~ -21.6 dB`，与 baseline 的 `-28 dB`
相差约 `6 ~ 8 dB`。

#### 2.1.5 训练时固定目标 decoder 的结果

固定 seed42 hybrid decoder，重新训练 TransNet encoder：

| Encoder Seed | Best NMSE | Final/Test NMSE |
|---:|---:|---:|
| 0 | `-28.620` | `-28.620` |
| 1 | `-28.620` | `-28.620` |
| 2026 | `-28.616` | `-28.615` |
| 3407 | `-28.613` | `-28.613` |
| 42 | `-28.656` | `-28.656` |
| 666 | `-28.628` | `-28.628` |
| 999 | `-28.619` | `-28.618` |

说明：decoder 不是不能跨 seed 使用；关键是 encoder 必须在训练过程中进入目标 decoder
可读的码字坐标系。

### 2.2 TransNet WAIRD seed797：场景迁移差异大，full fine-tuning 仍是上限

#### 2.2.1 Base 模型稳定，backend 和 layer sharing 不是主矛盾

| Base 实验 | Final NMSE |
|---|---:|
| `base/original_independent` | `-22.533` |
| `base/original_shared` | `-22.530` |
| `base/torch_independent` | `-22.506` |
| `base/torch_shared` | `-22.516` |
| `base/torch_independent_fc_lora_true_fc_lora_rank_512` | `-18.782` |

正常 base 的差异小于 `0.03 dB`。这说明当前迁移困难不是由 torch/original backend 或
layer sharing 选择导致的。

#### 2.2.2 Direct transfer 与 full fine-tuning

| 场景 | Direct NMSE | Full Fine-tuning NMSE | 提升 |
|---|---:|---:|---:|
| `01105` | `-18.604` | `-22.006` | `+3.402 dB` |
| `06401` | `-17.823` | `-18.137` | `+0.314 dB` |
| `09957` | `-18.669` | `-19.512` | `+0.843 dB` |

说明：

- `01105` 是明显可适配场景，full fine-tuning 能基本恢复到 base 水平。
- `06401` 是低可适配场景，放开全模型也只提升 `0.314 dB`。
- `09957` 介于两者之间。

这里的重点不是“某个 LoRA 组件好不好”，而是**不同目标场景本身的可迁移性差异很大**。

#### 2.2.3 PEFT/LoRA 与 full fine-tuning 的差距

先把三个场景按 direct、full、最佳/代表性 PEFT 放在同一张表里：

| 场景 | Direct NMSE | Full Fine-tuning | 代表性最佳 PEFT/局部训练 | PEFT 相对 Direct | PEFT 距 Full | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `01105` | `-18.604` | `-22.006` | `fc_decoder LoRA=-20.872` | `+2.268 dB` | `1.134 dB` | PEFT 有效，但 full 仍明显更强 |
| `06401` | `-17.823` | `-18.137` | `freeze-only fc_decoder=-18.214` | `+0.391 dB` | `优于 full 0.077 dB` | 可适配性低，局部 decoder 训练略有效 |
| `09957` | `-18.669` | `-19.512` | `decoder_ffn LoRA=-19.193` | `+0.524 dB` | `0.319 dB` | 中等可适配，decoder-side 更优 |

如果只看新版 `lora/` 下 encoder-side LoRA 和 teacher code sweep，结果更保守：

| 场景 | 组件 | 无 teacher code | `lambda=0.001` | `lambda=0.005` | `lambda=0.01` | `lambda=0.1` | 主要现象 |
|---|---|---:|---:|---:|---:|---:|---|
| `01105` | `fc_encoder rank64` | `-19.747` | `-19.738` | `-19.685` | `-19.603` | `-18.139` | teacher code 越强越差 |
| `01105` | `encoder_ffn rank64` | `-19.604` | `-19.600` | `-19.577` | `-19.541` | `-18.484` | 小权重无益，大权重退化 |
| `06401` | `fc_encoder rank64` | `-17.829` | `-17.810` | `-17.750` | `-17.704` | `-17.292` | 本来提升极小，teacher code 加剧退化 |
| `06401` | `encoder_ffn rank64` | `-17.826` | `-17.807` | `-17.781` | `-17.761` | `-17.489` | 低可适配场景，code 监督不解决问题 |
| `09957` | `fc_encoder rank64` | `-19.046` | `-19.045` | `-19.025` | `-18.992` | `-18.620` | 有轻微 PEFT 收益，强 code 监督仍伤害 |
| `09957` | `encoder_ffn rank64` | `-19.092` | `-19.089` | `-19.078` | `-19.067` | `-18.905` | 同样呈现 lambda 增大退化 |

这张表的含义很直接：当前 teacher code 不是稳定正则项，至少在“只训练局部 LoRA、decoder
大部分保持 base 分布”的设置下，强行靠近 teacher latent space 会伤害重建。

`01105` 上：

| 方法 | NMSE |
|---|---:|
| full fine-tuning | `-22.003 ~ -22.006` |
| fc_decoder LoRA, torch independent, rank64 | `-20.872` |
| decoder_ffn LoRA, torch independent | `-20.177` |
| fc_encoder LoRA, torch independent, rank64 | `-19.854` |
| encoder_ffn LoRA, torch independent | `-19.560` |
| 新版 fc_encoder rank64 | `-19.747` |
| 新版 encoder_ffn rank64 | `-19.604` |

`06401` 上：

| 方法 | NMSE |
|---|---:|
| full fine-tuning | `-18.137 ~ -18.206` |
| freeze-only fc_decoder | `-18.214` |
| freeze-only fc_encoder | `-18.158` |
| LoRA 组件 | 约 `-17.82 ~ -17.83` |

`09957` 上：

| 方法 | NMSE |
|---|---:|
| full fine-tuning | `-19.512` |
| decoder_ffn LoRA | `-19.193` |
| fc_decoder LoRA | `-19.144` |
| encoder_ffn LoRA | `-19.088` |
| fc_encoder LoRA | `-19.063` |

说明：

- PEFT 能改善 `01105` 和 `09957`，但仍与 full fine-tuning 有差距。
- `06401` 上大多数 LoRA 几乎没有有效提升。
- decoder 侧组件在多个旧版实验中更关键，尤其 `fc_decoder`、`decoder_ffn`。

#### 2.2.4 Teacher recon 和 teacher code 的问题

当前 `teacher_recon` 实验中，传入的是原始训练 CSI：

```text
data/WAIRD/scenario_2/01105/train.pt
```

而不是 teacher 模型的输出。因此损失实际变成：

```text
MSE(student_recon, train_csi)
+ lambda * MSE(student_recon, train_csi)
= (1 + lambda) * MSE(student_recon, train_csi)
```

这不是真正蒸馏，只是缩放重建 MSE。结果也基本无收益：

| teacher_recon lambda | Final NMSE |
|---:|---:|
| 无 teacher_recon | `-22.006` |
| `0.001` | `-22.009` |
| `0.01` | `-21.752` |
| `0.05` | `-22.005` |
| `0.1` | `-22.008` |
| `0.5` | `-22.008` |

teacher code 的问题更直接：lambda 越大通常越差。

以 `01105 fc_encoder rank64` 为例：

| teacher_code lambda | Final NMSE |
|---:|---:|
| 无 teacher_code | `-19.747` |
| `0.001` | `-19.738` |
| `0.005` | `-19.685` |
| `0.01` | `-19.603` |
| `0.1` | `-18.139` |

说明：当前 teacher code 与 student 解码路径并不天然兼容。强迫 student code 靠近 full
fine-tuned teacher code，可能反而让 frozen/base decoder 更难解码。

### 2.3 TransCSI WAIRD seed42：token 化结构下，重建侧适配更关键

#### 2.3.1 Base 与 direct transfer

| 实验 | 配置 | Final NMSE |
|---|---|---:|
| `base/token_num_torch_independent` | token_num | `-0.769` |
| `base/token_size_torch_independent` | token_size | `-19.124` |
| `direct_test/scenario_2_01105_token_size_torch_independent` | base 直接测 01105 | `-17.600` |

说明：

- `token_num` 当前结果明显异常或不适合作为主线。
- 后续分析应以 `token_size_torch_independent` 为有效 base。
- 从 base 到 `01105` direct transfer，NMSE 从 `-19.124` 退化到 `-17.600`，存在明显场景迁移损失。

#### 2.3.2 Full fine-tuning、冻结侧训练与 LoRA

| 方法 | 含义 | Final NMSE | 相对 Direct 提升 | 距 Full 差距 |
|---|---|---:|---:|---:|
| direct test | base 直接测 `01105` | `-17.600` | 基线 | `3.178 dB` |
| full fine-tuning | 全模型微调 | `-20.778` | `+3.178 dB` | `0.000 dB` |
| freeze compression side | 冻结重建侧，只训练压缩侧 | `-18.820` | `+1.220 dB` | `1.958 dB` |
| freeze reconstruction side | 冻结压缩侧，只训练重建侧 | `-20.092` | `+2.492 dB` | `0.686 dB` |
| token_compressor LoRA | 只改 token compressor | `-17.668` | `+0.068 dB` | `3.110 dB` |
| encoder_ffn LoRA | 只改 encoder FFN | `-18.363` | `+0.763 dB` | `2.415 dB` |
| encoder_ffn + token_compressor LoRA | 联合改压缩侧局部模块 | `-18.370` | `+0.770 dB` | `2.408 dB` |

说明：

- full fine-tuning 最强，比 direct 提升 `+3.178 dB`。
- 只训练重建侧达到 `-20.092`，明显强于只训练压缩侧的 `-18.820`。
- 只改 token compressor 几乎没有收益，`-17.668` 只比 direct 好 `0.068 dB`。
- encoder FFN LoRA 有一定收益，但仍远低于 full fine-tuning 和重建侧训练。

这一组结果与 TransNet WAIRD seed797 互相印证：**场景迁移并不只是 encoder 压缩码字问题，decoder/reconstruction side 的映射也很关键。**

## 3. 从表层现象抽象出的共同困境

三组实验虽然任务设置不同，但困境高度一致。

### 3.1 困境一：端到端重建学到的是“私有协议”，不是公共码字语言

Autoencoder 训练目标通常是：

```text
decoder(encoder(x)) ≈ x
```

这个目标只要求 encoder 和 decoder 成对配合，不要求 `encoder(x)` 遵守任何全局标准。

因此模型可以学到：

- 不同 seed 下不同的码字尺度。
- 不同维度的旋转、置换、混合。
- 不同 token 分布。
- decoder 内部层依赖的特定 code 统计。
- 同样能重建，但互不兼容的 latent 表示。

COST2100 跨 seed 结果是这个问题最干净的证据：同一数据、同一结构、只是 seed 不同，
独立模型都能到 `-28 dB`，但拆开拼接直接变成 `+15 ~ +29 dB`。

### 3.2 困境二：后验小模块是在翻译已经成型的私有协议，容量和约束都不够

Adapter、LoRA、只训 `fc_decoder` 的共同点是：大部分 encoder/decoder 已经固定，只允许
一个小模块去补偿两个 code space 的差异。

这类方法承担的任务其实很重：

```text
把 code_A 的私有协议翻译成 decoder_B 能读的 code_B 协议。
```

如果差异只是线性尺度或旋转，小模块可能足够；但如果差异涉及 token 分布、decoder 内部层
激活统计、样本结构重排，小模块就会卡住。

这解释了 COST2100 的现象：

- adapter 可以把 `+23.854` 拉到 `-20 ~ -21`。
- `fc_decoder` 可以把 `+15.461` 拉到 `-21.627`。
- 但都回不到 `-28 dB`。

也解释了 WAIRD 的 PEFT 现象：

- `01105` 上 LoRA 能提升，但不如 full fine-tuning。
- `06401` 上 LoRA 几乎不动。
- TransCSI 里只改 token compressor 或 encoder FFN 明显不如训练重建侧或 full fine-tuning。

### 3.3 困境三：teacher code 不是天然正确监督，强行对齐可能破坏 decoder 可读性

直觉上，teacher code 来自更好的 full fine-tuned 模型，似乎应该帮助 student。但实验显示：

- COST2100 adapter 固定 `lambda=0.1` 可以从本来能到 `-20.6` 退化到接近 `0 dB`。
- TransNet WAIRD `01105 fc_encoder rank64` 中，teacher code `lambda=0.1` 从 `-19.747`
  退化到 `-18.139`。
- `06401` 和 `09957` 也普遍呈现 lambda 增大、性能变差的趋势。

原因是：teacher code 所在的 latent space 与 student 当前 decoder 路径未必兼容。

如果 student 只训练 LoRA 或局部模块，而 decoder 大部分仍保持 base 分布，那么强迫 student
code 靠近 full model code，可能会出现：

```text
code 看起来接近 teacher，
但 decoder 读不懂这个 code，
最终 reconstruction 变差。
```

所以 teacher code 不能简单当作越大越好的监督项。它必须与目标 decoder 的可读空间一致，
并且权重要小、要 warmup、要先保证重建。

### 3.4 困境四：场景迁移包含压缩侧和重建侧两类变化

如果迁移只发生在 encoder 侧，那么只改 `fc_encoder`、`encoder_ffn` 或 token compressor
应该足够。但实验不支持这个简单解释。

证据：

- TransNet `01105` 旧版实验中，`fc_decoder LoRA=-20.872` 强于 `fc_encoder rank64=-19.854`。
- TransNet `09957` 中，`decoder_ffn=-19.193`、`fc_decoder=-19.144` 也强于 `fc_encoder=-19.063`。
- TransCSI `01105` 中，只训练重建侧 `-20.092`，明显强于只训练压缩侧 `-18.820`。

这说明 scenario shift 不只是“如何压缩”的问题，也包括“如何从码字重建新场景 CSI 结构”的问题。

## 4. 更深层的解释：我们现在面对的是码字空间、数据分布、优化目标三者不一致

### 4.1 码字空间不一致

这是 COST2100 跨 seed 的主问题：

```text
encoder_A(x) 和 encoder_B(x) 都足以重建 x，
但它们不是同一种语言。
```

表现：

- 直接重组崩溃。
- 后验 adapter 有明显天花板。
- frozen decoder 训练 encoder 立刻恢复。

解决方向：

- 固定目标 decoder 训练 encoder。
- 明确共同 teacher code。
- 加 bottleneck 规范化。
- 做 linear/MLP probe 判断 code space 是否可映射。

### 4.2 数据分布不一致

这是 WAIRD 场景迁移的主问题：

```text
base 场景和 scenario_2 的 CSI 分布不同；
不同 scenario_2 场景之间可适配性也不同。
```

表现：

- `01105` full fine-tuning 可恢复。
- `06401` full fine-tuning 也提升很小。
- `09957` 中等提升。

解决方向：

- 对每个场景先做 direct/full 上限测试。
- 不要假设一个 PEFT 方法对所有场景有效。
- 对低可适配场景分析数据分布、功率分布、稀疏结构、train/test 覆盖。

### 4.3 优化目标不一致

当前训练多用 MSE，汇报多看 NMSE。MSE 与 NMSE 不完全等价，尤其 CSI 样本能量差异较大时：

```text
MSE 优化偏向高能量样本；
NMSE 关心相对误差。
```

如果 LoRA 或 teacher code 让 MSE 略降但低能量样本相对误差变大，NMSE 可能变差。

解决方向：

- 补充 NMSE loss 或 normalized MSE loss。
- 保存每样本 NMSE 分布，而不是只看均值。
- 用统一 batch size、统一 evaluator 复评 direct/base/LoRA。

## 5. 当前最有价值的正向证据

### 5.1 固定 decoder 训练 encoder 是跨 seed 可插拔的最强证据

COST2100 中 fixed seed42 hybrid decoder + 多 seed encoder 的结果稳定在 `-28.6 dB`。

这说明：

```text
decoder 可复用；
encoder 也可换 seed；
但前提是 encoder 训练时必须面对目标 decoder。
```

这给出一个比“后验 adapter”更可靠的思路：

```text
不要训练完 encoder 再翻译 code；
直接在目标 decoder 的闭环里训练 encoder。
```

### 5.2 Full fine-tuning 是 WAIRD 场景迁移的经验上限

TransNet 和 TransCSI 都显示：

- `01105`：full fine-tuning 大幅优于 direct 和 LoRA。
- `09957`：full fine-tuning 仍优于组件 LoRA。
- `06401`：full fine-tuning 提升也有限，说明场景本身难。

这给出一个实验流程原则：

```text
每个新场景先跑 direct 和 full fine-tuning，
用它们定义下限和上限，
再判断 PEFT 是否值得继续。
```

### 5.3 重建侧适配的重要性不能忽视

TransNet 和 TransCSI 都提示 decoder/reconstruction side 很关键：

- `fc_decoder`、`decoder_ffn` 往往优于单纯 encoder-side LoRA。
- TransCSI 中冻结压缩侧、训练重建侧明显优于冻结重建侧、训练压缩侧。

这说明迁移方案不应只围绕“让 encoder code 更好”，还要考虑 decoder 如何适应新场景输出结构。

## 6. 建议的解决路线：从短期可验证到长期机制化

### 6.1 短期：先把问题分型，不再盲目加模块

#### 6.1.1 对每个场景建立 direct/full/PEFT 三点标尺

每个新场景先跑：

```text
direct test
full fine-tuning
best PEFT candidate
```

判断场景类型：

| 类型 | 表现 | 代表 | 策略 |
|---|---|---|---|
| 高可适配 | direct 差，full 明显恢复 | WAIRD `01105` | 优先找接近 full 的 PEFT |
| 低可适配 | direct 差，full 也提升小 | WAIRD `06401` | 先分析数据分布和模型容量，不急着调 LoRA |
| 中间型 | full 有中等提升 | WAIRD `09957` | 比较 decoder-side / encoder-side PEFT |

#### 6.1.2 对跨 seed 问题做 linear/MLP code probe

步骤：

```text
1. 导出同一批 x 的 code_A = encoder_A(x)
2. 导出 code_B = encoder_B(x)
3. 训练 Linear(code_A -> code_B)
4. 测 decoder_B(Linear(code_A))
5. 再训练 MLP(code_A -> code_B)
```

解释：

- Linear 能接近原性能：主要是线性坐标错位。
- MLP 明显更好：非线性错位。
- MLP 也不行：后验翻译路线价值有限，应训练时对齐。

#### 6.1.3 对 LoRA 增加 epoch 0 评估

尤其 WAIRD seed797/01109 参考分析指出：LoRA 可能在前 10 epoch 内把 direct base 拉坏。

必须补：

```text
LoRA 初始化后、训练前的 test NMSE
epoch 1/5/10 的 test NMSE
best checkpoint test NMSE
```

否则无法判断是：

- LoRA 初始化就不等价于 base；
- 学习率太大早期破坏；
- 目标函数方向与 NMSE 不一致；
- 可训练模块选择错误。

### 6.2 中期：训练时显式规定共同码字空间

#### 6.2.1 固定目标 decoder，训练 encoder

这是 COST2100 已验证最强路线：

```text
decoder_target = seed42 decoder
freeze(decoder_target)
train encoder_new

loss = MSE(decoder_target(encoder_new(x)), x)
```

用于跨 seed：

- 加载已有独立 encoder，但不要冻结 encoder。
- 冻结目标 decoder。
- 全量或分阶段 fine-tune encoder。

建议新增模式：

```text
--pretrained_encoder path
--pretrained_decoder path
--freeze_decoder
--unfreeze_encoder_after_load
```

对比：

```text
from scratch encoder + frozen decoder
pretrained encoder init + frozen decoder
encoder.fc only + frozen decoder
last transformer block + frozen decoder
full encoder + frozen decoder
```

#### 6.2.2 Teacher code 只能小权重、晚启动、服务于 decoder 可读性

不要直接用 `lambda=0.1`。推荐：

```text
stage 1:
loss = recon_loss

stage 2:
loss = recon_loss + lambda(t) * code_loss
```

其中：

```text
epoch 1   ~ 100: lambda = 0
epoch 100 ~ 250: lambda 从 0 线性升到 1e-4
epoch 250 ~ 400: lambda = 1e-4
```

如果太弱，再试：

```text
3e-4
1e-3
```

teacher code 的目标应该是：

```text
在不损失 NMSE 的前提下，让多个 encoder 的 code 更容易对齐。
```

而不是单纯追求 code MSE 最小。

#### 6.2.3 加 bottleneck 规范化和分布约束

可以让所有 encoder 输出经过统一规范化：

```text
code = LayerNorm(code_raw)
```

或加入统计约束：

```text
mean_loss = ||mean(code)||^2
var_loss  = ||std(code) - 1||^2
cov_loss  = ||Cov(code) - I||^2
```

这不会保证语义维度完全对齐，但能减少尺度和协方差漂移，是比强 teacher MSE 更温和的公共空间约束。

### 6.3 中期：PEFT 不要只改单点，要围绕“压缩侧 + 重建侧”成组设计

现有结果显示单点 LoRA 经常不够。更合理的 PEFT 组合：

```text
encoder_ffn + decoder_ffn
fc_encoder + fc_decoder
token_compressor + token_decompressor
encoder_last_block + decoder_first_projection
```

对 TransNet：

- `fc_decoder`、`decoder_ffn` 已经显示价值。
- 不应只做 `fc_encoder` 或 `encoder_ffn`。

对 TransCSI：

- `token_compressor` 单独几乎没收益。
- 重建侧训练更有效，应考虑 `token_decompressor`、decoder FFN、decoder attention。

建议按参数量分层：

| 粒度 | 可训练模块 | 目的 |
|---|---|---|
| 极小 | 单个 LoRA 组件 | 快速定位敏感层 |
| 小 | encoder-side + decoder-side 成对 LoRA | 同时适配码字生成和读取 |
| 中 | 冻结一侧、训练另一侧 | 判断迁移主要发生在哪一侧 |
| 大 | full fine-tuning | 上限 |

### 6.4 长期：训练 universal decoder 或共享 codebook

如果目标是“多个 encoder 可插拔接一个 decoder”，需要从训练目标上改变设定。

#### Universal decoder

训练一个 decoder 接受多个来源的 code：

```text
decoder_universal(code_from_seed42) -> x
decoder_universal(code_from_seed2026) -> x
decoder_universal(code_from_seed3407) -> x
```

必要时加入 source embedding：

```text
decoder_universal(code, source_id) -> x
```

优点：

- decoder 学会多种 code space。
- 比后验 adapter 更系统。

风险：

- decoder 容量需求更高。
- 需要防止 decoder 只记 source-specific shortcut。

#### 共享 codebook

更强约束是引入离散或共享 codebook：

```text
encoder(x) -> shared codebook index/embedding -> decoder
```

优点：

- 码字空间天然统一。
- 更适合“可插拔 encoder”目标。

风险：

- 实现复杂。
- 可能损失 NMSE。
- 需要处理 codebook collapse 和利用率。

## 7. 建议组会重点汇报的主线

### 7.1 一句话主线

> 我们当前遇到的核心问题不是某个 LoRA rank 或 adapter 容量没有调好，而是 CSI 反馈自编码模型的码字空间缺少公共约束。  
> 在跨 seed 和跨场景迁移中，encoder 和 decoder 的私有协议会破坏可插拔性；后验微调只能部分修复。  
> 下一步应从训练闭环中定义 decoder 可读的共同码字空间，并把 PEFT 从单点微调升级为压缩侧与重建侧协同适配。

### 7.2 建议展示顺序

1. **先展示 COST2100 跨 seed 的强对照**  
   独立模型都 `-28 dB`，重组直接 `+15 ~ +29 dB`，后验补丁只到 `-21 dB`，frozen decoder 训练 encoder 回到 `-28.6 dB`。

2. **再展示 WAIRD 场景迁移的复杂性**  
   `01105` full fine-tuning 大幅有效，`06401` full 也难提升，说明场景有不同可适配类型。

3. **再展示 TransCSI 的侧面证据**  
   token 化结构下，只训重建侧明显优于只训压缩侧，说明 decoder/reconstruction side 是迁移关键。

4. **最后给出统一解释**  
   码字空间私有协议 + 场景分布偏移 + MSE/NMSE 目标不一致。

5. **给出下一步计划**  
   frozen decoder + unfreeze encoder、linear/MLP probe、成组 LoRA、teacher code warmup、universal decoder。

## 8. 下一阶段具体实验清单

### 8.1 必做实验

1. **COST2100：已有 encoder 初始化 + frozen decoder + 解冻 encoder**
   - 目标：判断独立 encoder 权重能否作为适配初始化。
   - 对照：from scratch encoder + frozen decoder。

2. **COST2100：linear/MLP code probe**
   - 目标：判断跨 seed code space 差异是否可映射。

3. **WAIRD/TransNet：LoRA epoch 0 评估**
   - 目标：确认 LoRA 是初始化不等价还是早期训练破坏。

4. **WAIRD/TransNet：decoder-side + encoder-side 成组 LoRA**
   - 例如 `fc_encoder + fc_decoder`、`encoder_ffn + decoder_ffn`。

5. **WAIRD/TransCSI：补 token_decompressor / decoder-side LoRA**
   - 目标：验证 TransCSI 中重建侧更关键的判断。

### 8.2 建议实验

1. **Teacher code warmup**
   - `lambda=0 -> 1e-4 -> 3e-4 -> 1e-3`
   - 只在 recon-only 已经稳定后加入。

2. **Bottleneck LayerNorm / code distribution regularization**
   - 目标：减少跨 seed code 统计漂移。

3. **按场景建立可适配性画像**
   - 对 `01105/06401/09957/01109` 比较 direct、full、PEFT、数据统计。

4. **统一 evaluator 复评**
   - 相同 batch size。
   - 样本级 NMSE 分布。
   - 避免 batch-level dB 平均造成口径差异。

## 9. 风险和注意事项

1. **不要把 teacher code 当作默认正确答案**  
   teacher code 必须和目标 decoder 可读空间一致，否则会伤害重建。

2. **不要只看 final，需要看 best 和训练早期**  
   LoRA 可能很早破坏 base，final 不能解释退化发生在哪里。

3. **不要混用不同维度口径**  
   WAIRD seed796/seed797 参考分析提示 32x32 与 64x64 可能混用。后续所有报告必须显式写清 `nt/nc`。

4. **不要只比较方法名，要比较可训练范围和参数量**  
   “LoRA”不是一个统一方法；单组件 rank8 与成组 LoRA 或 full fine-tuning 不是同一量级。

5. **不要只看平均 NMSE**  
   需要看每样本 NMSE 分布，尤其低能量样本可能主导退化。

## 10. 最终结论

现有三组实验共同说明：

1. **COST2100 跨 seed问题**揭示了自编码器码字空间的非唯一性。  
   独立训练的 encoder/decoder 形成私有协议，后验 adapter/LoRA/fc_decoder 只能部分翻译；
   fixed decoder 训练 encoder 才能稳定恢复。

2. **TransNet WAIRD 场景迁移问题**揭示了场景分布偏移和 PEFT 天花板。  
   full fine-tuning 是当前上限；LoRA 在可适配场景有效但不够，在困难场景几乎无效；
   teacher code 当前设置下经常伤害性能。

3. **TransCSI WAIRD token 化实验**进一步说明迁移不只是压缩侧问题。  
   重建侧训练明显优于压缩侧训练，单改 token compressor 不够。

统一起来看，下一步不应继续盲目堆单点模块，而应围绕三个问题设计实验：

```text
1. 目标 decoder 的可读码字空间如何定义？
2. 新 encoder 如何在训练闭环中进入这个空间？
3. 新场景的变化主要落在压缩侧、重建侧，还是两者都需要协同适配？
```

真正值得推进的方向是：

```text
固定目标 decoder / 显式共同码字空间
+ encoder 训练或适配
+ decoder-side 与 encoder-side 成组 PEFT
+ 谨慎、小权重、晚启动的 teacher/code 正则
+ 每场景可适配性画像
```

这比继续在已经成型的私有 latent space 后面加一个小模块，更接近问题本质。
