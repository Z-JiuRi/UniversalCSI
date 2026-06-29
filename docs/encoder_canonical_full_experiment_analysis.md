# COST2100/in encoder_canonical 实验综合分析

本文分析 `exps/COST2100/in/encoder_canonical` 下当前已有的全部实验结果、日志和码字。统计对象包括 78 个 `run.log`、78 个 `args.json` 和 78 个 `codewords/train_code.pt`。

码字统计说明：

- `best NMSE` 来自日志中的 `Best NMSE: ... (epoch=...)`，数值越小越好。
- `final` 来自训练脚本最后的 `Final test`；部分实验曾在训练结束导出 best checkpoint 时崩溃，因此没有 final 行。
- `cross-seed cos` 是同一方案内不同 seed、同一样本 codeword 的平均余弦相似度，使用每个码字文件前 20000 个训练样本统计。
- `offdiag` 是样本协方差非对角 Frobenius 范数与对角 Frobenius 范数之比，越大说明维度间相关越强。
- `eff_rank` 是协方差谱的 participation ratio，近似表示有效维度数。

## 总结结论

当前最成功的 encoder 约束方案是 `aux_pca_1e-3`。它在 5 个 seed 上平均 best NMSE 为 `-29.508 dB`，明显好于 canonical baseline 的 `-27.306 dB`；同时跨 seed 同样本余弦从 baseline 的 `-0.006` 提升到 `0.778`。这说明 PCA auxiliary target 确实把不同 seed 的 code 拉向了更接近的公共坐标系，而且没有牺牲重建能力。

但是，`aux_pca_1e-3` 还没有做到真正的“同一坐标系”。`cross-seed cos=0.778` 只是显著改善，不是完全对齐。它保留了约 `eff_rank=95.6` 的有效维度，码字分布没有明显坍缩，这是它优于其他强约束方案的关键。

`code_reg`、`codebook`、`DCT` 这类方案能把跨 seed 余弦进一步推高，但代价通常是码字表达退化。例如 `aux_pca_5e-3_code_reg` 的 cross-seed cos 达到 `0.9963`，但有效秩只有 `3.2`，best NMSE 降到 `-25.473 dB`。这不是更好的 canonical code，而是把不同 encoder 压到了少数公共主方向上。

adapter 实验里，最好的仍是 `gated_lowrank_affine_mlp`，尤其是 `rank32_hidden2048_gate0.1_code1e-3_fc1e-2`。当前 best 为 `-27.601 dB @ epoch 1200`。它能把 adapter 后 code 几乎完全贴近 seed42 teacher code，但测试 NMSE 仍比 seed42 自己的 `aux_pca_1e-3` 自编码性能差约 2 dB，说明训练集 code 对齐本身已经不是唯一瓶颈，泛化和 decoder 敏感性仍然存在。

## 全方案结果概览

| 方案 | 实验数 | 有 final | 日志崩溃 | best 平均 | best 最好 | best 最差 | code std | offdiag | eff_rank | dim_var_cv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 5 | 5 | 0 | -27.306 | -28.180 | -26.211 | 0.7452 | 2.828 | 52.1 | 0.31 |
| `aux_pca_1e-3` | 5 | 5 | 0 | -29.508 | -29.889 | -29.181 | 0.5305 | 0.099 | 95.6 | 2.07 |
| `aux_dct_1e-3` | 5 | 5 | 0 | -24.612 | -25.072 | -24.306 | 1.3313 | 2.928 | 51.2 | 0.21 |
| `fixed_q` | 5 | 5 | 0 | -20.888 | -21.671 | -20.025 | 0.8086 | 2.377 | 74.9 | 0.18 |
| `fixed_q_rank16` | 5 | 5 | 0 | -22.208 | -22.846 | -21.863 | 0.4999 | 2.342 | 77.3 | 0.15 |
| `fixed_q_rank16_code_reg` | 5 | 5 | 0 | -22.305 | -22.741 | -22.132 | 0.9907 | 1.355 | 136.7 | 0.57 |
| `fixed_q_rank16_pca` | 5 | 5 | 0 | -13.073 | -13.123 | -12.944 | 0.2260 | 0.296 | 30.9 | 3.78 |
| `codebook1024` | 5 | 5 | 0 | -24.874 | -25.308 | -23.838 | 0.0121 | 4.612 | 23.4 | 0.38 |
| `codebook1024_dct_reg` | 5 | 5 | 0 | -14.910 | -15.073 | -14.799 | 0.0052 | 2.990 | 49.2 | 0.22 |
| `aux_pca_1e-3_code_reg` | 3 | 3 | 0 | -24.698 | -24.833 | -24.516 | 0.9741 | 9.605 | 4.2 | 0.56 |
| `aux_pca_5e-3_code_reg` | 3 | 3 | 0 | -25.473 | -25.617 | -25.311 | 0.9658 | 11.192 | 3.2 | 0.53 |
| `aux_pca_1e-2_code_reg` | 3 | 3 | 0 | -23.773 | -25.308 | -20.717 | 0.9631 | 11.686 | 2.9 | 0.52 |
| `adapter/aux_pca_1e-3/gated_lowrank_affine_mlp` | 9 | 4 | 4 | -26.985 | -27.601 | -25.879 | 0.4972 | 0.098 | 95.9 | 2.06 |
| `adapter/aux_pca_1e-3/lowrank_affine_mlp` | 1 | 1 | 0 | -26.573 | -26.573 | -26.573 | 0.4968 | 0.097 | 95.9 | 2.06 |
| `adapter/aux_pca_1e-3/mlp` | 9 | 9 | 0 | -23.272 | -24.884 | -12.571 | 0.9510 | 0.242 | 88.9 | 2.07 |
| `adapter/aux_pca_1e-3/gated_lowrank_affine_linear` | 2 | 0 | 2 | -21.061 | -21.935 | -20.187 | 0.4975 | 0.098 | 96.0 | 2.06 |
| `adapter/aux_pca_1e-3/diag_affine` | 1 | 1 | 0 | -12.535 | -12.535 | -12.535 | 0.4530 | 0.078 | 77.0 | 2.36 |
| `adapter/aux_pca_1e-3/lowrank_affine` | 1 | 1 | 0 | -11.259 | -11.259 | -11.259 | 0.5533 | 0.143 | 116.1 | 1.81 |
| `adapter/aux_pca_5e-3_code_reg/mlp` | 1 | 1 | 0 | -16.703 | -16.703 | -16.703 | 0.9753 | 11.323 | 3.1 | 0.53 |

注意：`gated_lowrank_affine_mlp` 和 `gated_lowrank_affine_linear` 中有几组 `run.log` 记录了导出阶段的历史崩溃，因此 final 为空；当前 `train_code.pt` 文件已经存在，本文的码字统计按现有文件计算。

## 跨 seed 码字坐标系对齐

| 方案 | pair 数 | cross-seed cos 平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| `baseline` | 10 | -0.0060 | -0.0173 | 0.0030 |
| `aux_pca_1e-3` | 10 | 0.7783 | 0.5842 | 0.9998 |
| `aux_dct_1e-3` | 10 | 0.9877 | 0.9789 | 0.9995 |
| `aux_pca_1e-3_code_reg` | 3 | 0.9965 | 0.9957 | 0.9973 |
| `aux_pca_5e-3_code_reg` | 3 | 0.9963 | 0.9959 | 0.9970 |
| `aux_pca_1e-2_code_reg` | 3 | 0.9969 | 0.9965 | 0.9973 |
| `fixed_q` | 10 | 0.0704 | -0.1131 | 0.1992 |
| `fixed_q_rank16` | 10 | 0.2626 | 0.1468 | 0.3578 |
| `fixed_q_rank16_code_reg` | 10 | 0.2625 | 0.2148 | 0.3334 |
| `fixed_q_rank16_pca` | 10 | 0.8853 | 0.8713 | 0.9034 |
| `codebook1024` | 10 | 0.9019 | 0.8927 | 0.9140 |
| `codebook1024_dct_reg` | 10 | 0.9921 | 0.9908 | 0.9934 |

这个表要和 NMSE、有效秩一起看。`baseline` 的重建不差，但跨 seed code 几乎正交，说明不同 seed 的 encoder/decoder 自发学出了互不兼容的坐标系。`aux_pca_1e-3` 把这个问题明显缓解，同时保持了最好的重建性能。

`aux_dct_1e-3`、`codebook1024_dct_reg`、PCA + code_reg 的跨 seed 余弦都很高，但 NMSE 显著变差。原因不是坐标系问题已经完美解决，而是表达空间被强行压成了公共但不适合 decoder 重建的形状。

## 关键 encoder 方案分析

### baseline

baseline 平均 best NMSE 为 `-27.306 dB`，单 seed 最好 `-28.180 dB`。它的问题不在单模型重建，而在坐标系完全不统一：cross-seed cos 为 `-0.0060`，基本等于随机正交。

从数学上看，自编码器目标只约束 `D_s(E_s(x)) ≈ x`。如果存在一个可逆变换 `R`，那么 `D_s R^{-1}` 和 `R E_s` 可以得到几乎相同的重建误差。这就是 encoder code 坐标系不唯一的根源。baseline 没有任何公共锚点，所以不同 seed 会选择不同的 `R`。

### aux_pca_1e-3

`aux_pca_1e-3` 是目前最好的折中：

| 实验 | best NMSE | final | code std | offdiag | eff_rank |
|---|---:|---:|---:|---:|---:|
| `aux_pca_1e-3/seed3407_transnet_transnet` | -29.889 | -29.889 | 0.5026 | 0.090 | 95.3 |
| `aux_pca_1e-3/seed2026_transnet_transnet` | -29.620 | -29.620 | 0.5918 | 0.104 | 95.5 |
| `aux_pca_1e-3/seed42_transnet_transnet` | -29.528 | -29.528 | 0.4969 | 0.098 | 95.8 |
| `aux_pca_1e-3/seed1024_transnet_transnet` | -29.323 | -29.323 | 0.5157 | 0.094 | 95.7 |
| `aux_pca_1e-3/seed796_transnet_transnet` | -29.181 | -29.181 | 0.5455 | 0.111 | 95.7 |

它比 baseline 更好，说明 PCA auxiliary target 没有只是“牺牲重建换对齐”，反而可能给 encoder 提供了更稳定的低频/主成分结构先验。它的 offdiag 只有 `0.099`，比 baseline 的 `2.828` 小很多；有效秩约 `96`，也没有掉到少数维度。

但它的 cross-seed cos 只有 `0.7783`，说明 PCA anchor 只给了一个软坐标偏好。网络仍然可以在满足重建和弱 PCA loss 的前提下保留 seed 特有自由度。

### PCA + code regularization

`aux_pca_1e-3_code_reg`、`aux_pca_5e-3_code_reg`、`aux_pca_1e-2_code_reg` 都有一个共同现象：cross-seed cos 约 `0.996`，但 NMSE 掉到 `-23.8 ~ -25.5 dB`，有效秩只有 `2.9 ~ 4.2`。

这说明当前 code regularization 并没有得到“高维统一坐标系”，而是把大部分样本压到了少数公共方向。`offdiag` 达到 `9 ~ 12`，表示协方差虽然在边缘统计上可能被某些 loss 拉住，但整体仍是低秩且强相关的 dense covariance。

结论是：当前这组 code regularization 不能作为主方案。它可以提高跨 seed 表面相似度，但损害了 CSI 重建所需的信息容量。

### fixed Q / fixed Q + low-rank

`fixed_q` 的 cross-seed cos 只有 `0.0704`，NMSE 也只有 `-20.888 dB`。这说明固定随机正交投影 Q 本身不能强制 code 进入统一坐标系。

原因是 Q 固定在 encoder 末端，但 Q 前面的 feature 是可学习的。不同 seed 仍可以学习不同的 feature 坐标，再经过同一个 Q 得到不同的 code 分布。也就是说，固定 Q 只固定了最后一层线性读出，不固定 feature 空间本身。

`fixed_q_rank16` 加了低秩可学习残差后，NMSE 提升到 `-22.208 dB`，cross-seed cos 提到 `0.2626`，但仍远不够。`fixed_q_rank16_code_reg` 的有效秩提高到 `136.7`，但 cross-seed cos 仍只有 `0.2625`，说明 marginal code regularization 对坐标方向约束很弱。

`fixed_q_rank16_pca` 的 cross-seed cos 达到 `0.8853`，但 NMSE 只有 `-13.073 dB`。这属于约束冲突：它确实把不同 seed 拉近了，但损坏了 decoder 可用的信息结构。

### fixed codebook

`codebook1024` 的 cross-seed cos 为 `0.9019`，但平均 best NMSE 只有 `-24.874 dB`。更关键的是 code std 只有 `0.0121`，平均 sample norm 约 `0.27`，说明 codebook convex combination 的动态范围非常小。

固定 codebook 的问题在于：如果 encoder 只能输出 assignment logits，再由固定 codebook 做 convex combination，那么 code 的可表达区域被限制在固定点集的凸包内。这个凸包可以提供公共坐标锚点，但表达能力和尺度都受限。decoder 可以适应一部分小尺度 code，但高精度 CSI 重建需要更细粒度的连续信息。

`codebook1024_dct_reg` 进一步把 cross-seed cos 推到 `0.9921`，但 NMSE 掉到 `-14.910 dB`，code std 只有 `0.0052`。这基本是强公共约束造成的低幅值退化。

### DCT auxiliary target

`aux_dct_1e-3` 的 cross-seed cos 为 `0.9877`，但平均 best NMSE 只有 `-24.612 dB`。DCT target 比 PCA 更固定、更物理，但它不一定匹配 TransNet encoder 最自然的重建特征。

PCA target 是从原始 CSI 数据分布得到的公共 basis，仍然贴近数据主变化方向；DCT target 是手工频率 basis，对齐更强但任务适配更差。因此 DCT 更像强几何约束，不适合作为当前主训练目标。

## adapter 实验分析

adapter 目标是用 `enc_seed2026` 的 encoder 加 adapter 去适配 `dec_seed42` 的 decoder。当前最好的结构是：

```text
gated_lowrank_affine_mlp
rank = 32
hidden = 2048
gate_init = 0.1
lambda_code = 1e-3
lambda_fc = 1e-2
```

Top adapter 结果：

| 实验 | best NMSE | final | 历史崩溃 | cos(adapter, teacher42) | MSE(adapter, teacher42) | cos(adapter, source2026) |
|---|---:|---:|---:|---:|---:|---:|
| `gated_lowrank_affine_mlp/rank32_hidden2048_gate0.1_code1e-3_fc1e-2_lr5e-4_ep3000` | -27.601 |  | False | 1.0000 | 1.91e-05 | 0.6586 |
| `gated_lowrank_affine_mlp/rank32_hidden2048_gate0.1_code1e-3_fc1e-2_lr5e-4_ep1000` | -27.478 | -27.425 | False | 1.0000 | 2.01e-05 | 0.6595 |
| `gated_lowrank_affine_mlp/rank32_hidden2048_gate0.1_code1e-3_fc1e-2_lr5e-4_ep400` | -27.305 | -27.303 | False | 1.0000 | 3.31e-05 | 0.6598 |
| `gated_lowrank_affine_mlp/rank32_hidden2048_gate0.1_code1e-3_fc1e-3_lr5e-4_ep400` | -27.024 |  | True | 1.0000 | 3.17e-05 | 0.6589 |
| `gated_lowrank_affine_mlp/rank32_hidden1024_gate0.1_code1e-3_fc1e-3_lr5e-4_ep400` | -26.907 |  | True | 0.9999 | 4.73e-05 | 0.6595 |
| `gated_lowrank_affine_mlp/rank32_hidden1024_gate0.1_code1e-3_fc1e-2_lr5e-4_ep400` | -26.818 | -26.818 | False | 0.9999 | 6.88e-05 | 0.6590 |
| `lowrank_affine_mlp/rank32_hidden2048_code1e-3_fc1e-2_lr2e-4` | -26.573 | -26.490 | False | 1.0000 | 3.39e-05 | 0.6595 |
| `gated_lowrank_affine_linear/rank32_gate0.1_code1e-3_fc1e-3_lr5e-4_ep400` | -21.935 |  | True | 0.9987 | 8.79e-04 | 0.6589 |
| `diag_affine/rank32_code1e-3_fc1e-2_lr2e-4` | -12.535 | -11.771 | False | 0.9887 | 8.96e-03 | 0.6351 |
| `lowrank_affine/rank32_code1e-3_fc1e-2_lr2e-4` | -11.259 | -11.091 | False | 0.9864 | 1.33e-02 | 0.6757 |

几个关键判断：

1. `lambda_fc=1e-2` 比 `lambda_fc=1e-3` 更可靠。即使 code-space MSE 已经很小，decoder 的 `fc_decoder` 后特征仍然敏感；只对齐 raw code 不足以保证 decoder 内部特征对齐。
2. `hidden2048` 仍是当前最稳配置。`hidden1024` 有一定潜力，但现有结果没有超过 hidden2048；`hidden512` 明显容量不足。
3. `gated_lowrank_affine_linear` 不能替代 MLP。它的 code 分布形状看起来正常，甚至 cos teacher 很高，但 NMSE 只有 `-20 ~ -22 dB`。这说明 decoder 对结构化小误差非常敏感，线性残差无法表达跨 seed encoder 之间的非线性差异。
4. 纯 `diag_affine` 和纯 `lowrank_affine` 基本失败。它们能做尺度、偏置、低秩旋转，但不能完成高维非线性的局部纠正。
5. adapter 后 code 与 teacher42 的训练集余弦接近 1，但测试 NMSE 仍未接近 teacher 自编码器的 `-29.528 dB`。因此后续优化重点不是继续把 train code MSE 压到更低，而是提升 adapter 的泛化、平滑性和 decoder 内部特征一致性。

## 为什么“坐标系统一”和“重建性能”会冲突

自编码器的 code 不是自然物理量，而是 encoder 和 decoder 共同协商出的隐变量。重建损失只要求：

```text
D(E(x)) ≈ x
```

对任意可逆变换 `R`，都可以构造：

```text
E'(x) = R E(x)
D'(z) = D(R^{-1} z)
```

于是 `D'(E'(x))` 和原模型等价。这就是不同 seed 产生不同 code 坐标系的数学原因。

要打破这个自由度，必须引入公共锚点。但锚点过弱，坐标系仍然漂移；锚点过强，encoder 会牺牲 CSI 重建所需的自由度。当前实验正好验证了这个权衡：

- `baseline`：重建可以，坐标系完全不统一。
- `aux_pca_1e-3`：重建更好，坐标系明显更接近，但不完全统一。
- `code_reg / DCT / codebook`：坐标更统一，但表达能力明显受损。

所以目标不应该是盲目最大化 cross-seed cos，而是找到“公共坐标约束”和“任务信息容量”的平衡点。

## 后续建议

1. 主线保留 `aux_pca_1e-3`，不要把当前 code_reg 作为默认方案。它是唯一同时满足高 NMSE、较好跨 seed 对齐、有效秩不塌的方案。
2. adapter 主线继续用 `gated_lowrank_affine_mlp`，优先比较 `hidden2048` 和 `hidden1024`，并固定 `lambda_fc=1e-2` 做更干净的对照。
3. 对 adapter 的优化不要再只盯 `MSE(z_adapter, z_teacher)`。现有 best adapter 的 train code 已经几乎贴近 teacher，后续应更重视 `fc_decoder` feature、teacher reconstruction、一致性正则和测试集泛化。
4. 不建议继续投入 `fixed_q`、`fixed codebook`、`DCT` 强约束作为主路线。它们可以作为理论对照，但当前日志和码字都显示重建代价太高。
5. 如果继续研究 encoder 侧约束，更合理的方向是保留 `aux_pca_1e-3` 的软锚点，同时设计不会低秩坍缩的分布约束，例如按 batch 做轻量 whitening、Barlow Twins 风格 cross-correlation 约束，或者只约束前若干稳定 PCA 维度而不是全 code。

## 当前最可信的结论

`aux_pca_1e-3` 已经证明“公共数据 basis 可以减轻不同 seed 的 code 坐标系旋转问题”，但还没有彻底解决。强行提高坐标一致性会导致 code 退化，最终伤害 NMSE。adapter 能进一步把一个 seed 的 encoder code 映射到另一个 seed decoder 可用的空间，但目前瓶颈已经从训练集 code 对齐转向泛化和 decoder 内部特征敏感性。

