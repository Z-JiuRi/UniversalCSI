# mapper 下一阶段 decoder-aware loss 设计分析

本文总结在当前 `mapper` 实验基础上，下一步如果加入 decoder-aware loss，应该如何加入、为什么这样加入，以及每个判断对应的实验证据。

## 1. 结论

当前最值得做的不是从头训练一个更复杂的 mapper，也不是把 decoder reconstruction loss 直接作为主损失强行训练，而是：

```text
第一阶段：保持当前最强的 affine + hybrid residual mapper 做 code-space 粗对齐
第二阶段：加载第一阶段 checkpoint，用小学习率做 decoder-aware finetune
```

第二阶段推荐的主 loss：

```text
L =
  λ_code * MSE(z_a, z_t)
+ λ_recT * MSE(D_t(z_a), D_t(z_t))
+ λ_fc   * MSE(fc_t(z_a), fc_t(z_t))
+ λ_rec  * MSE(D_t(z_a), x)
+ λ_tail * TailMSE(D_t(z_a), x)
```

第一批最推荐的配置：

```text
λ_code = 0.3
λ_recT = 1.0
λ_fc   = 1e-2
λ_rec  = 0.0 或 0.1
λ_tail = 0.0 或 0.05
```

训练策略：

```text
从当前 alignaffine + hybrid 的 best checkpoint 继续训练
lr = 5e-5 或 1e-4
epochs = 100~200
eval_decoder_every = 5 或 10
checkpoint 选择以 true fixed-decoder NMSE 为主，同时保留 best code MSE
```

## 2. 当前最强基线已经不是旧 mapper

旧分析报告中，最好 mapper 大约是：

```text
old hybrid / seed2026 transnet:
fixed decoder NMSE ≈ -25.39 dB
teacher fixed decoder NMSE ≈ -29.10 dB
gap ≈ 3.72 dB
```

见：

```text
mapper/reports/mapper_exps_full_analysis.md
mapper/reports/mapper_loss_design_analysis.md
```

但最新全量测试 `mapper/reports/affine_true_nmse/affine_code_nmse.csv` 显示，当前最强路线已经变成：

```text
align_mode=affine
residual_mapping=1
residual_condition=start
mapper=hybrid
```

全量 train CSI 上，固定 seed42 transnet decoder 的真实 NMSE：

| source | true fixed-decoder NMSE | decoder MSE |
|---|---:|---:|
| seed2026/transnet | -28.112 dB | 6.972890e-07 |
| seed3407/transnet | -27.915 dB | 7.296136e-07 |
| seed2026/clnet | -27.665 dB | 7.728927e-07 |
| seed2026/crnet | -27.643 dB | 7.766889e-07 |
| seed3407/clnet | -27.609 dB | 7.828643e-07 |
| seed3407/csinet | -27.403 dB | 8.209124e-07 |
| seed3407/crnet | -27.142 dB | 8.716429e-07 |
| seed2026/csinet | -26.864 dB | 9.292841e-07 |

去重后按 mapper 类型统计：

| mapper | n | best | mean | median | worst | std |
|---|---:|---:|---:|---:|---:|---:|
| delta_mlp | 1 | -24.057 | -24.057 | -24.057 | -24.057 | 0.000 |
| flow | 4 | -22.335 | -21.641 | -21.624 | -20.981 | 0.480 |
| hybrid | 8 | -28.112 | -27.544 | -27.626 | -26.864 | 0.376 |
| mlp | 8 | -26.117 | -24.802 | -25.192 | -23.158 | 0.920 |

实证结论：

1. `affine + hybrid` 已经系统性优于 MLP、flow 和 delta_mlp。
2. 当前问题已经不是“完全没有对齐”，而是“离 teacher 还差最后约 1~2 dB”。
3. 下一步 loss 设计应服务于精修 fixed decoder 敏感方向，而不是重新承担全局坐标系对齐。

## 3. 为什么不能直接从头加 decoder-aware loss

已有旧 decoder-aware 实验：

```text
mapper/exps_decoder_aware
mapper/exps_combined_losses
```

其中最好的结果大约是：

| 实验目录 | fixed decoder NMSE | code/selection MSE |
|---|---:|---:|
| mapper/exps_combined_losses/hybrid/smooth_tail_recT_fc/seed2026_transnet... | -25.497 dB | 0.001810 |
| mapper/exps_combined_losses/hybrid/smooth_tail_white_recT_fc/seed2026_transnet... | -25.447 dB | 0.001831 |
| mapper/exps_combined_losses/hybrid/smooth_tail_white_recT/seed2026_transnet... | -25.444 dB | 0.001832 |
| mapper/exps_decoder_aware/hybrid/recT/seed2026_transnet... | -25.377 dB | 0.001861 |
| mapper/exps_decoder_aware/hybrid/recT_rec/seed2026_transnet... | -25.366 dB | 0.001866 |

这些明显弱于最新：

```text
affine + hybrid / seed2026 transnet = -28.112 dB
```

这不是证明 decoder-aware loss 没用，而是说明旧训练方式不合适。旧实验多是从 source code 直接训练 mapper，同时加入 decoder-aware 项。这样 decoder-aware loss 需要同时解决：

```text
1. 不同 seed/架构之间的大坐标系旋转、混合、尺度、偏置
2. fixed decoder 敏感方向的细修
```

但 decoder-aware loss 最擅长的是第二件事，不擅长第一件事。

最新 affine 实验证明，全局坐标系问题应先由闭式 affine 和 code-space residual mapper 解决。decoder-aware loss 应该作为第二阶段微调项，而不是第一阶段主力。

## 4. 为什么 code MSE 不够

最终目标不是：

```text
z_a ≈ z_t
```

而是：

```text
D_t(z_a) ≈ x
```

其中：

```text
z_s = source encoder code
z_t = seed42 teacher encoder code
z_a = mapper(z_s)
D_t = seed42 fixed decoder
x   = 原始 CSI
```

在 teacher code 附近，一阶近似为：

```text
D_t(z_a) - D_t(z_t) ≈ J_D(z_t) (z_a - z_t)
```

普通 code MSE 优化的是：

```text
||z_a - z_t||^2
```

fixed decoder 真正在意的是：

```text
||J_D(z_t)(z_a - z_t)||^2
```

因此，如果残差落在 decoder 的高增益方向，即使 code MSE 较低，重建 NMSE 也会差。

已有实证：

1. `mapper/reports/mapper_exps_full_analysis.md` 中，旧 hybrid mapper 的 all code MSE 已到 `3.117e-3`、cosine 到 `0.99767`，但 fixed decoder NMSE 仍只有 `-25.39 dB`，离 teacher `-29.10 dB` 差 `3.72 dB`。
2. `mapper/reports/generative_code_mapping_feasibility.md` 中，teacher code 加噪声实验显示，fixed decoder NMSE 掉 `1 dB` 时，code MSE 约为 `7.54e-4`。旧最好 code MSE `2.093e-3` 仍是该阈值的约 `2.78x`。
3. 同一报告中，fixed decoder 的 `fc_decoder.weight` 奇异值最大约 `5.112`，中位数约 `1.310`，最小约 `0.779`。仅第一层就已经不是等距映射。

所以 decoder-aware loss 的作用不是简单让 code 更像 teacher，而是让 residual 避开 fixed decoder 的高敏感方向。

## 5. 为什么先用 recT，再小心加入 rec

### 5.1 recT 的定义

```text
L_recT = MSE(D_t(z_a), D_t(z_t))
```

它要求 mapped code 经过 fixed decoder 后，输出接近 teacher code 经过同一个 fixed decoder 后的输出。

实证和理论依据：

1. 一阶近似下，`L_recT` 等价于 decoder-Jacobian 加权 code loss。
2. target 是 `D_t(z_t)`，属于 fixed decoder 在自然 teacher code 上的输出，训练稳定。
3. 它不会要求 mapper 超越 teacher encoder/decoder 本身，只要求行为接近 teacher。
4. 旧 decoder-aware 实验中，`recT` 类实验是旧体系里较好的分支之一，例如 `mapper/exps_decoder_aware/hybrid/recT/seed2026_transnet...` 达到 `-25.377 dB`，虽不如新 affine+hybrid，但说明 recT 本身方向合理。

推荐：

```text
λ_recT = 1.0
```

### 5.2 rec 的定义

```text
L_rec = MSE(D_t(z_a), x)
```

它直接优化最终重建。

优点：

```text
和最终 fixed decoder NMSE 最一致
可以修正 teacher reconstruction 本身不完美的部分
```

风险：

```text
如果 λ_rec 太大，mapper 可能离开 teacher code manifold
在 train 上变好，但 test 或跨样本稳定性变差
```

原因是 fixed decoder 训练时主要见过 teacher encoder 产生的 code 分布。如果 mapper 为了训练集重建找到一些非自然 code，短期 train NMSE 可能更好，但不一定稳。

因此第一轮建议：

```text
λ_rec = 0.0 或 0.1
```

确认 `recT + fc` 有稳定收益后，再试：

```text
λ_rec = 0.3
```

不建议一开始：

```text
λ_rec = 1.0
λ_code = 0
```

## 6. 为什么要加 fc loss

定义：

```text
L_fc = MSE(fc_t(z_a), fc_t(z_t))
```

其中 `fc_t` 是 seed42 fixed decoder 的 `fc_decoder`。

实证依据来自 `mapper/reports/generative_code_mapping_feasibility.md`：

```text
decoder.fc_decoder.weight: (2048, 512)
max singular value    ≈ 5.112
p95 singular value    ≈ 1.744
median singular value ≈ 1.310
min singular value    ≈ 0.779
```

这说明 code 残差在进入 decoder 第一层时已经会被各向异性放大。普通 code MSE 对 512 维方向等权，而 fixed decoder 对这些方向不等权。

`fc loss` 的优点：

1. 比完整 decoder reconstruction 更接近 code 侧，梯度路径短。
2. 比普通 code MSE 更接近 decoder 感知距离。
3. 可以提前压住 `fc_decoder` 高奇异方向上的残差。

推荐：

```text
λ_fc = 1e-2
```

可追加一组：

```text
λ_fc = 3e-2
```

不建议第一轮开太大，因为 `fc_decoder` 输出是 2048 维，过强可能让训练过度贴第一层特征，而不是最终 reconstruction。

## 7. 为什么需要保留 code loss

定义：

```text
L_code = MSE(z_a, z_t)
```

实证依据：

1. `mapper/reports/generative_code_mapping_feasibility.md` 显示，mapped code 的全局分布已经非常接近 teacher，包括 std、norm、effective rank、top50 PCA energy。这说明 teacher code manifold 是有效目标。
2. 同报告指出，当前失败不是全局分布不像 teacher，而是每个样本的 pairwise residual 仍然会被 fixed decoder 放大。
3. 如果去掉 code loss，只靠 `rec`，可能得到 fixed decoder 能在 train 上重建的非自然 code，破坏 teacher manifold。

所以 decoder-aware finetune 不是替代 code loss，而是在 code loss 的约束下修 decoder-sensitive residual。

推荐：

```text
λ_code = 0.3
```

如果非常保守：

```text
λ_code = 1.0
```

如果想更激进冲 train NMSE：

```text
λ_code = 0.1
```

不建议：

```text
λ_code = 0
```

除非作为 ablation。

## 8. 为什么要考虑 tail loss

已有报告 `mapper/reports/mapper_loss_design_analysis.md` 给出旧 mapper 的尾部证据：

```text
teacher p95 sample NMSE = -24.57 dB
mapped  p95 sample NMSE = -21.48 dB
mapped  p99 sample NMSE = -17.89 dB
mapped 超过 teacher p95 的样本比例 = 19.5%
```

同一报告还指出：

```text
mapped residual: Laplace better than Normal = 10 / 10
residual 是尖峰重尾，不是普通高斯小噪声
```

`mapper/reports/generative_code_mapping_feasibility.md` 也显示，`smooth+tail+white` 虽然降低了样本尾部，但 sample p95/p99 仍高于 1 dB 噪声阈值：

```text
smooth+tail+white sample p95 = 4.471e-3，是 7.54e-4 阈值的 5.93x
smooth+tail+white sample p99 = 7.449e-3，是 7.54e-4 阈值的 9.88x
```

因此平均 MSE 不足以描述问题。对 decoder-aware 阶段，可以加入：

```text
e_i = mean((D_t(z_a_i) - x_i)^2)
L_tail = mean(topk(e_i, k = tail_ratio * batch_size))
```

推荐：

```text
tail_ratio = 0.2
λ_tail = 0.05
```

第一轮不要太大。tail loss 太强可能牺牲大多数普通样本，只修少数难样本。

## 9. 推荐实验矩阵

先在三个 source 上验证：

```text
seed2026/transnet_transnet  # 当前最好，同架构，检验能否逼近 teacher
seed2026/clnet_transnet     # 跨架构较好，检验泛化
seed2026/csinet_transnet    # 跨架构最难，检验下限
```

基线是最新 affine+hybrid：

| source | baseline true NMSE |
|---|---:|
| seed2026/transnet | -28.112 dB |
| seed2026/clnet | -27.665 dB |
| seed2026/csinet | -26.864 dB |

### 配置 A：稳健 recT

```text
λ_code = 0.3
λ_recT = 1.0
λ_fc   = 0
λ_rec  = 0
λ_tail = 0
```

目的：验证 decoder-Jacobian 加权 code loss 是否能稳定提升真实 NMSE。

预期：最稳，不容易破坏 code manifold。

### 配置 B：recT + fc

```text
λ_code = 0.3
λ_recT = 1.0
λ_fc   = 1e-2
λ_rec  = 0
λ_tail = 0
```

目的：约束 `fc_decoder` 后的 2048 维特征，压第一层高敏感方向。

预期：如果 B 明显优于 A，说明第一层 feature mismatch 是主要瓶颈。

### 配置 C：recT + fc + 小 rec

```text
λ_code = 0.3
λ_recT = 1.0
λ_fc   = 1e-2
λ_rec  = 0.1
λ_tail = 0
```

目的：在 teacher decoder consistency 基础上，直接优化最终 CSI reconstruction。

预期：最可能成为综合最好配置。

### 配置 D：recT + fc + 小 rec + tail

```text
λ_code = 0.3
λ_recT = 1.0
λ_fc   = 1e-2
λ_rec  = 0.1
λ_tail = 0.05
tail_ratio = 0.2
```

目的：修复样本尾部。

预期：平均 NMSE 可能提升不大，但 p95/p99 应改善。

### 配置 E：更强 rec

```text
λ_code = 0.1
λ_recT = 1.0
λ_fc   = 1e-2
λ_rec  = 0.3
λ_tail = 0.05
```

目的：更直接冲 fixed decoder train NMSE。

风险：code manifold 可能被破坏，必须同时看 code MSE 和 test NMSE。

### 配置 F：只 rec ablation

```text
λ_code = 0.1
λ_recT = 0
λ_fc   = 0
λ_rec  = 1.0
λ_tail = 0
```

目的：验证直接 reconstruction loss 是否会投机。

预期：如果 train NMSE 好但 code MSE/test 差，说明必须保留 teacher manifold 约束。

## 10. 训练设置

建议第二阶段设置：

```text
resume/checkpoint = 第一阶段 affine+hybrid best_mse 或 best_nmse
lr = 5e-5
eta_min = 1e-5
epochs = 100~200
batch_size = 1024 或 2048
scheduler = warmup cosine
eval_decoder_every = 5 或 10
eval_decoder_max_samples = 0
```

为什么小学习率：

1. 当前 `affine + hybrid` 已经能达到 `-28.112 dB`，离 teacher 很近。
2. 大学习率容易破坏已经对齐的 teacher-like code manifold。
3. decoder-aware 阶段要修的是小 residual，不是重新学习全局映射。

## 11. 必须监控的指标

只看训练 loss 不够。每个 epoch 或每隔若干 epoch 至少记录：

```text
true_decoder_nmse
code_mse = MSE(z_a, z_t)
recT = MSE(D_t(z_a), D_t(z_t))
rec = MSE(D_t(z_a), x)
fc = MSE(fc_t(z_a), fc_t(z_t))
sample reconstruction p95/p99
dim RMSE max 或 top-k dim error
```

判断标准：

```text
强成功：
  seed2026/transnet 从 -28.112 提到 -28.5~-28.8 dB
  跨架构多数达到 -28 dB 附近

中等成功：
  每个 source 提升 0.3~0.5 dB
  code MSE 不明显恶化

失败：
  true NMSE 不提升
  或 true NMSE 只在 train 提升但 code MSE/test 明显变差
```

checkpoint 建议同时保存：

```text
best_mse.pth       # code MSE 最优
best_nmse.pth      # true fixed-decoder NMSE 最优
best_recT.pth      # recT 最优，可选
```

最终比较时以 `best_nmse.pth` 导出的 mapped code 为主，但必须同时检查 code MSE 和 test NMSE。

## 12. 不建议的方向

基于当前实证，不建议下一步优先做：

1. 从头训练 decoder-aware mapper，跳过 affine/code-only 预训练。
2. `λ_rec` 很大且 `λ_code=0`。
3. 继续扩大 flow 参数量。最新 affine 测试中，flow 的真实 NMSE 只有 `-20.981~-22.335 dB`，明显弱于 hybrid。
4. 只按 code MSE 选 checkpoint。已有分析说明 code MSE 与 fixed decoder NMSE 相关但不等价。
5. 只看 train loss，不做周期 fixed decoder NMSE。

## 13. 最终建议

最优先实验：

```text
架构：alignaffine + hybrid residual
训练方式：加载第一阶段 best checkpoint 二阶段 finetune

L =
  0.3  * MSE(z_a, z_t)
+ 1.0  * MSE(D_t(z_a), D_t(z_t))
+ 1e-2 * MSE(fc_t(z_a), fc_t(z_t))
+ 0.1  * MSE(D_t(z_a), x)

lr = 5e-5
epochs = 100~200
eval_decoder_every = 5
```

如果第一轮要更稳：

```text
L =
  0.3  * MSE(z_a, z_t)
+ 1.0  * MSE(D_t(z_a), D_t(z_t))
+ 1e-2 * MSE(fc_t(z_a), fc_t(z_t))
```

如果确认平均 NMSE 有提升但尾部仍差，再加：

```text
+ 0.05 * TailMSE(D_t(z_a), x), tail_ratio=0.2
```

一句话总结：

```text
affine + hybrid 已经解决了大部分坐标系问题；
decoder-aware loss 的正确角色是第二阶段小步精修 fixed decoder 敏感方向；
核心优先级是 recT，其次 fc，再小权重 rec 和 tail。
```

