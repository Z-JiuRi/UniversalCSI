# decoder_lora 的 LoRA 参数量与训练步数优化方法调研

日期：2026-07-08

本文目标是为当前 `decoder_lora` 实验选择更省参数、更快收敛、或更稳定的 LoRA 方案。当前代码的 LoRA 注入点主要是：

- `decoder.fc_decoder`: `512 -> 2048`
- `decoder.decoder.layers.{0,1}.linear1`: `64 -> 2048`
- `decoder.decoder.layers.{0,1}.linear2`: `2048 -> 64`

因此 only-LoRA 的参数量可以近似写成：

```text
LoRA params = 2560 * fc_rank + 8448 * ffn_rank
```

例如 `fc_rank=256, ffn_rank=16` 时是 `790,528` 参数；`fc_rank=64, ffn_rank=8` 时是 `231,424` 参数。

## 结论摘要

最适合当前仓库优先尝试的方向：

1. 非均匀 rank：保留较高 `fc_decoder` rank，显著降低 FFN rank。
   当前任务的 code 坐标对齐主要发生在 `512 -> 2048` 的 decoder 输入展开处，FFN 更像后续重建细化。优先扫 `fc_rank=64/128/256` 与 `ffn_rank=4/8/16`，比统一 rank 更合理。

2. rsLoRA 缩放：把 scaling 从 `alpha / rank` 改成 `alpha / sqrt(rank)`。
   Rank-Stabilized LoRA 指出原始 `alpha/r` 会让高 rank 学习变慢。当前 sweep 中大量使用高 `fc_rank`，这个改动可能直接缩短训练步数或降低 alpha sweep 成本。

3. LoRA+ 优化器：给 `lora_up` 和 `lora_down` 使用不同学习率。
   LoRA+ 报告同等计算下可提升性能并最高约 2x 加快微调。当前实现里 `lora_down` 随机初始化、`lora_up` 零初始化，很适合做分组学习率实验。

4. 数据驱动初始化：用 affine 后的 `z0` 输入和 target CSI/reconstruction 残差做一次 warm-start。
   PEFT 中的 EVA/PiSSA/OLoRA 都说明初始化会明显影响收敛。对本项目，最直接的变体是对每个 LoRA 目标层采样前向 activation，做低秩初始化或最小二乘初始化，而不是纯随机 down + zero up。

5. 早停与低成本代理评估：不要每个组合固定跑满 400/1000 epoch。
   建议先用 `eval_decoder_max_samples=10000` 或更小样本每 10 epoch 评估，若 50-100 epoch 内 true decoder NMSE 明显落后，就提前停止。

不建议优先做的方向：

- QLoRA：主要节省冻结大模型的显存。当前 decoder 只有约 3.26M 参数，瓶颈更可能是实验数量和 epoch 数，不是 base model 显存。
- VeRA/VB-LoRA：参数量极省，但实现复杂度高，且当前模型只有少数线性层，收益不一定超过简单 rank 降低。
- DoRA：低 rank 性能可能更好，但会引入额外 magnitude 参数和实现复杂度；可作为第二阶段实验。

## 方法调研

### 1. 非均匀 rank / 动态 rank

AdaLoRA 提出不要给所有矩阵均匀分配低秩预算，而是按重要性自适应分配参数预算；它通过类似 SVD 的参数化和奇异值重要性裁剪，把预算留给更重要的更新矩阵。论文报告低预算场景下比固定 rank baseline 更好。

ALoRA 和 ARD-LoRA 也是同一方向：训练中估计不同 rank 或不同模块的重要性，逐步剪掉无效 rank，把预算转给更需要的模块。ARD-LoRA 还把 rank 分配做成可微优化，并加入稀疏正则。

对当前项目的落地建议：

- 先不用完整实现 AdaLoRA。当前 decoder 目标层很少，可以用手工非均匀 rank 达到大部分收益。
- 重点比较：

```text
fc_rank=64,  ffn_rank=4    params=197,632
fc_rank=64,  ffn_rank=8    params=231,424
fc_rank=128, ffn_rank=8    params=395,264
fc_rank=128, ffn_rank=16   params=462,848
fc_rank=256, ffn_rank=8    params=723,456
fc_rank=256, ffn_rank=16   params=790,528
```

- 若 `fc_rank=128, ffn_rank=8` 接近 `256/16`，说明 FFN rank 可以降。
- 若 `fc_rank=64, ffn_rank=16` 明显弱，而 `128/8` 强，说明主要瓶颈在 `fc_decoder`。

进一步可实现“训练后 rank 压缩”：训练较大 rank 后，计算每个 LoRA 层的 `Delta W = up @ down * scaling`，对 `Delta W` 做 SVD 截断到较小 rank，再继续短训。这是比完整 AdaLoRA 简单很多的后处理压缩。

### 2. rsLoRA：更适合高 rank 的缩放

标准 LoRA 使用：

```text
Delta W = (alpha / rank) * B @ A
```

Rank-Stabilized LoRA 建议改为：

```text
Delta W = (alpha / sqrt(rank)) * B @ A
```

Hugging Face PEFT 文档也把 `use_rslora=True` 解释为使用 `lora_alpha / math.sqrt(r)`，并指出这是被证明更好的缩放方式。

对当前项目的意义：

- 你现在 sweep 里有 `fc_rank=128/256` 这类高 rank。原始 `alpha/r` 对高 rank 可能学习偏慢，所以才需要较大的 alpha，例如 `fc_alpha=2048`。
- rsLoRA 可以减少 alpha 网格搜索维度。可以把 alpha 设成较温和的值，再由 `sqrt(rank)` 缩放保持更新强度。

建议实现：

- 在 `LoRALinear` 增加 `scale_mode` 参数：

```text
scale_mode=vanilla: scaling = alpha / rank
scale_mode=rslora:  scaling = alpha / sqrt(rank)
```

- 实验优先级：

```text
fc_rank=128, ffn_rank=8,  scale_mode=vanilla vs rslora
fc_rank=128, ffn_rank=16, scale_mode=vanilla vs rslora
fc_rank=256, ffn_rank=16, scale_mode=vanilla vs rslora
```

判断标准：同等 epoch 下 true decoder NMSE 更快接近平台值，或相同 NMSE 所需 epoch 更少。

### 3. LoRA+：不同矩阵使用不同学习率

LoRA+ 的核心观点是 LoRA 的 A/B 两个矩阵使用同一学习率并不理想，尤其在较大宽度模型上会限制有效特征学习。它建议给两个矩阵设置不同学习率比例，论文报告同等计算下有性能收益，并且微调速度最高约 2x。

当前实现中：

```text
lora_down = A: random init
lora_up   = B: zero init
```

这意味着训练早期只有 `lora_up` 直接收到有效梯度，`lora_down` 的有效学习会滞后。分组学习率是低成本改动：

```text
lora_up:   lr
lora_down: lr / 8 或 lr / 16
```

或反向测试：

```text
lora_up:   lr * 2
lora_down: lr
```

建议先实现为 optimizer 分组参数，而不是改模块结构。可加 CLI：

```text
--lora_down_lr_ratio 0.125
--lora_up_lr_ratio 1.0
```

优先 sweep：

```text
down/up = 1.0/1.0
down/up = 0.125/1.0
down/up = 0.25/1.0
down/up = 1.0/2.0
```

### 4. LoRA-FA：冻结 A，只训练 B

LoRA-FA 冻结投影 down 矩阵 A，只训练 up 矩阵 B。论文强调这能降低 activation memory，并保持接近 LoRA 的效果。对当前小 decoder，显存收益不是关键，但参数量会直接下降一半：

```text
fc_decoder 原 LoRA 参数: rank * (512 + 2048)
LoRA-FA 只训 up:        rank * 2048

FFN linear1 原参数: rank * (64 + 2048)
只训 up:           rank * 2048

FFN linear2 原参数: rank * (2048 + 64)
只训 up:           rank * 64
```

注意：对 `linear2: 2048 -> 64`，只训练 up 的参数很少，表达能力也可能明显不足；所以更实际的变体是：

- `fc_decoder` 和 `linear1` 使用 LoRA-FA。
- `linear2` 保持普通 LoRA，或直接不注入。

当前任务可作为低参数 baseline：

```text
fc_rank=128, ffn_rank=8, freeze_lora_down=1
fc_rank=256, ffn_rank=16, freeze_lora_down=1
```

如果 NMSE 损失很小，说明 down 矩阵的自由度不是必要的。

### 5. DoRA：低 rank 下增强表达能力

DoRA 把权重更新分为 magnitude 和 direction：direction 仍由 LoRA 处理，magnitude 用额外参数学习。论文称它提升 LoRA 的学习能力和训练稳定性，尤其低 rank 时有优势；PEFT 文档也说明 DoRA 可能提升低 rank 表现，但训练/推理开销比纯 LoRA 更大，建议推理前 merge。

对当前项目的意义：

- 如果目标是“同等参数量下更强”，DoRA 值得试。
- 如果目标是“最少参数”，DoRA 未必优先，因为它会为每个输出维度增加 magnitude 参数。

建议第二阶段做：

```text
fc_rank=32/64, ffn_rank=4/8 + DoRA
```

看它能否接近普通 LoRA 的 `fc_rank=128, ffn_rank=16`。

### 6. 数据驱动初始化：EVA / PiSSA / OLoRA 思路

PEFT 的 LoRA 配置已经支持多种初始化，包括 EVA、OLoRA、PiSSA、CorDA、LoftQ 和 orthogonal。PEFT 文档说明：

- EVA 基于 finetuning 数据的 layer input activations 做 SVD 初始化，并可按解释方差重新分配 rank。
- PiSSA 使用主奇异值和奇异向量初始化，目标是更快收敛并提升最终效果。
- CorDA 在某些模式下比 PiSSA 收敛更快。

当前项目不一定要完整复刻这些方法，但可以实现更贴合 CSI decoder 的版本：

1. 先用 affine 得到 `z0`。
2. 用 frozen seed42 decoder 前向，记录每个 LoRA 目标层输入 activation。
3. 目标残差可以设为：

```text
teacher_recon - base_decoder(z0)
```

或各层 feature residual。
4. 对每个线性层拟合低秩 `Delta W`，初始化 `lora_down/up`。

这会把训练从“从 0 学 residual”改成“从闭式近似开始微调”，很可能减少 epoch 数。它的工程复杂度比 LoRA+ 高，但比完整 AdaLoRA 更可控。

### 7. QLoRA / LoftQ

QLoRA 的核心是冻结 base model 做 4-bit 量化，并通过 LoRA 反传，配合 NF4、double quantization 和 paged optimizer 大幅降低大模型微调显存。PEFT 文档中的 LoftQ 则用于量化 backbone 并初始化 LoRA，使量化误差更小。

对当前项目不建议优先做，原因是：

- 当前 fixed decoder 只有约 3.26M 参数，量化节省有限。
- 主要成本是跑很多配置和 epoch，不是单模型显存。
- 量化可能影响 CSI 重建 NMSE，收益风险比不高。

只有当你后续把 decoder 扩大到明显更大的 Transformer，或需要同时并行很多任务时，再考虑 QLoRA/LoftQ。

### 8. VeRA / VB-LoRA：共享参数极限压缩

VeRA 用全局共享的随机低秩矩阵，只学习小的 scaling vectors；VB-LoRA 用共享 vector bank 组合低秩矩阵。它们都能极大减少每任务存储参数，VB-LoRA 论文报告在 Llama2-13B 上只用 LoRA 存储参数的 0.4%。

对当前项目的判断：

- 如果你要为大量 seed/encoder 训练很多 adapter，长期看有价值。
- 但当前 decoder 只有 5 个 LoRA 目标线性层，普通 LoRA 参数已经在 0.1M-0.9M 量级；共享矩阵方案的工程复杂度可能不划算。

建议暂时只作为第三阶段方向。

## 当前项目的推荐实验路线

### 阶段 A：不改代码，仅靠已有 run_tasks sweep

先完成 only-LoRA rank/alpha sweep，并记录：

```text
summary.tsv:
label, fc_rank, fc_alpha, ffn_rank, ffn_alpha,
est_lora_params, log_lora_params,
best_loss_nmse_db, best_loss_epoch,
best_nmse_db, best_nmse_epoch
```

优先看参数量与 NMSE 的 Pareto 前沿：

- 如果 `231k` 参数组接近 `790k` 参数组，后续主线用小 rank。
- 如果 alpha scale 差异很小，后续固定 alpha/rank 比例即可。
- 如果高 rank 只有训练后期才追上，优先做 rsLoRA 或 LoRA+。

### 阶段 B：小改代码，优先实现 rsLoRA + LoRA+

建议按以下顺序：

1. 增加 `--lora_scale_mode vanilla|rslora`。
2. 增加 optimizer 参数组：

```text
--lora_down_lr_ratio
--lora_up_lr_ratio
```

3. 对 Pareto 前沿上的 2-3 个 rank 组合复跑短训：

```text
epochs=100
eval_decoder_every=10
eval_decoder_max_samples=10000
```

如果 100 epoch 下 rslora/loraplus 明显领先，再跑 400 epoch。

### 阶段 C：中等改动，做 LoRA-FA 和训练后 SVD 压缩

LoRA-FA 改动：

- `LoRALinear` 增加 `freeze_down=True`。
- `mark_only_lora_trainable()` 排除 `lora_down`。
- optimizer 只更新 `lora_up`。

训练后 SVD 压缩：

- 对每层 `Delta W = up @ down * scaling` 做 `torch.linalg.svd`。
- 截断 rank。
- 重新构造较小的 LoRA down/up。
- 用较小学习率短训 20-50 epoch。

这条路线能直接回答：“大 rank 训练得到的能力能否压缩到小 rank”。

### 阶段 D：较大改动，数据驱动初始化

实现成本更高，但最可能减少训练步数。建议先只对 `fc_decoder` 做，因为它占参数量最大、最贴近 code 坐标变换：

```text
fc_decoder LoRA params = 2560 * fc_rank
```

如果 `fc_decoder` 的 warm-start 能显著减少 epoch，再推广到 FFN。

## 建议优先级

| 优先级 | 方法 | 主要收益 | 实现成本 | 风险 |
|---|---|---:|---:|---|
| P0 | 非均匀 rank sweep | 降参数 | 无需改代码 | 需要跑实验 |
| P0 | 早停/小样本 true NMSE | 降训练步数 | 低 | 小样本指标可能有噪声 |
| P1 | rsLoRA scaling | 高 rank 更快更稳 | 低 | alpha 需重扫 |
| P1 | LoRA+ LR 分组 | 更快收敛 | 低 | LR ratio 需调 |
| P2 | LoRA-FA | 减半可训练参数/状态 | 中 | 表达能力下降 |
| P2 | 训练后 SVD 压缩 | 压缩已训练 adapter | 中 | 需短训恢复 |
| P3 | DoRA | 低 rank 性能 | 中 | 额外参数与开销 |
| P3 | EVA/PiSSA 类初始化 | 降 epoch | 高 | 实现复杂 |
| P4 | VeRA/VB-LoRA | 极限存储压缩 | 高 | 对小 decoder 未必划算 |
| P4 | QLoRA/LoftQ | 降 base 显存 | 中高 | 当前模型太小，收益低 |

## 参考资料

- LoRA: Low-Rank Adaptation of Large Language Models, arXiv: https://arxiv.org/abs/2106.09685
- AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning, arXiv: https://arxiv.org/abs/2303.10512
- QLoRA: Efficient Finetuning of Quantized LLMs, arXiv: https://arxiv.org/abs/2305.14314
- LoRA-FA: Memory-efficient Low-rank Adaptation, arXiv: https://arxiv.org/abs/2308.03303
- VeRA: Vector-based Random Matrix Adaptation, arXiv: https://arxiv.org/abs/2310.11454
- Rank-Stabilized LoRA, arXiv: https://arxiv.org/abs/2312.03732
- DoRA: Weight-Decomposed Low-Rank Adaptation, arXiv: https://arxiv.org/abs/2402.09353
- LoRA+: Efficient Low Rank Adaptation of Large Models, arXiv: https://arxiv.org/abs/2402.12354
- ALoRA: Allocating Low-Rank Adaptation, arXiv: https://arxiv.org/abs/2403.16187
- VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks, arXiv: https://arxiv.org/abs/2405.15179
- Hugging Face PEFT LoRA documentation: https://huggingface.co/docs/peft/v0.19.0/package_reference/lora
