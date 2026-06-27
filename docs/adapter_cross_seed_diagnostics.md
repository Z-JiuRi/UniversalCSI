# Cross-seed adapter 诊断分析

本文专门分析 `aux_pca_1e-3` 规范化 encoder/decoder 上的 cross-seed adapter 问题：

```text
encoder: aux_pca_1e-3 seed2026
decoder: aux_pca_1e-3 seed42
adapter: mlp
```

目标是回答：

1. seed42 teacher code、seed42 decoder checkpoint、train 数据顺序和导出逻辑是否严格一致。
2. seed42 decoder 输入 teacher code 和 adapter 后 code 时，重建差异有多大。
3. adapter 后 code 与 teacher code 的逐维 mean/var/cov 和误差谱是什么样。
4. decoder feature / reconstruction consistency 作为 adapter loss 是否可行。
5. adapter 结构如何优化，是否有可能接近 baseline 水平。
6. 还有哪些分析能辅助判断当前 adapter 的瓶颈。

所有新计算均使用全量 train 数据 `N=100000`，中间结果只写入 `/tmp`：

```text
/tmp/adapter_deep_analysis_results.json
/tmp/adapter_full_distribution_results.json
/tmp/adapter_decoder_error_decomposition.json
```

没有写入或污染现有实验目录。

## 1. 一致性核对

### 1.1 实验路径

seed42 teacher 来源：

```text
exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed42_transnet_transnet
```

adapter 实验中的路径来自各自 `args.json`：

```text
pretrained_encoder = exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed2026_transnet_transnet/checkpoints/best_nmse.pth
pretrained_decoder = exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed42_transnet_transnet/checkpoints/best_nmse.pth
teacher_code       = exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed42_transnet_transnet/codewords/train_code.pt
```

因此，所有带 `lambda_code > 0` 的 adapter 实验确实使用 seed42 的 decoder checkpoint 和 seed42 的 teacher code。`lambda_code=0` 的实验没有加载 teacher code，这是预期行为。

### 1.2 checkpoint 与 teacher code 是否来自同一个模型

seed42 aux PCA 日志显示：

```text
Best NMSE: -2.9528e+01 (epoch=400)
Saved index-aligned encoder outputs (100000, 512)
```

也就是说，seed42 的 best epoch 是最终 epoch 400，训练结束后保存 codeword 时使用的是最终模型；最终模型就是 best checkpoint 对应的模型。

我用 seed42 `best_nmse.pth` 重新构建完整模型，并在同一份 train 数据上重新计算 `model.encode(x)`，与已保存的 teacher code 比较：

| 检查项 | 数值 |
|---|---:|
| code shape | `(100000, 512)` |
| recomputed vs saved code MSE | `3.23e-13` |
| recomputed vs saved mean L2 | `1.28e-5` |
| recomputed vs saved max abs | `1.48e-5` |

这个误差只有浮点级别，证明：

```text
seed42 teacher code 与 seed42 best checkpoint 严格一致。
```

### 1.3 数据顺序是否一致

代码路径：

- `main.py` 构建 `MyDataLoader(..., return_indices=True)`。
- `dataloader/dataloader.py` 中 train loader 虽然 `shuffle=True`，但返回 `(data, index)`。
- `Trainer.save_codewords()` 会收集 index，并把输出重新排列回 `0..N-1` 的原始样本顺序。

因此即使 train loader shuffle，保存出的 `train_code.pt` 仍然是 index-aligned。seed42 日志也明确打印了：

```text
Saved index-aligned encoder outputs (100000, 512)
```

当前函数名已经改为 `save_codewords()`，并保存 `model.encode(x)`。旧日志里的 `encoder outputs` 是旧文案，不影响这次一致性判断。

## 2. seed42 decoder 直接解码对比

我固定 seed42 decoder，分别输入：

- seed42 teacher code。
- 各 adapter 实验重新导出的 adapter 后 code。
- 原始 seed2026 encoder code 作为参考。

这里的 NMSE 是在 train 全量数据上重新计算的，和日志里的 test NMSE 不是同一个 split，因此数值不要求完全相同。

### 2.1 teacher code 上限

| 输入 | train NMSE |
|---|---:|
| seed42 decoder + saved teacher code | `-29.8357 dB` |
| seed42 full model `encoder -> decoder` | `-29.8357 dB` |

两者完全一致。这再次说明 teacher code、decoder checkpoint 和数据顺序是匹配的。

### 2.2 adapter code 直接输入 seed42 decoder

| adapter 实验 | log best test NMSE | decoder42 + adapter code train NMSE | code L2 to teacher | code cos to teacher |
|---|---:|---:|---:|---:|
| `recon0.0_code1.0_lr1e-3` | -24.507 | -26.7854 | 0.3428 | 0.9997 |
| `recon1.0_code0.0_lr2e-4` | -12.571 | -13.1767 | 101.1642 | 0.2086 |
| `recon1.0_code1e-1_lr1e-3` | -24.342 | -26.7319 | 0.3436 | 0.9997 |
| `recon1.0_code1e-2_lr1e-3` | -24.084 | -26.1048 | 0.3997 | 0.9996 |
| `recon1.0_code1e-3_lr2e-4` | **-24.753** | **-27.1322** | **0.3414** | **0.9998** |

结论：

- 不加 code loss 的 adapter 完全不可靠，adapter 后 code 被推到异常分布，code L2 到 `101`，train NMSE 只有 `-13.18 dB`。
- 加 code loss 后，adapter 后 code 已经非常接近 seed42 teacher code。
- 但即使 code L2 只有约 `0.34`，decoder42 的 train NMSE 也只能到 `-27.13 dB`，距离 teacher code 的 `-29.84 dB` 仍有约 `2.70 dB` 差距。

## 3. 为什么 0.34 code L2 残差仍然致命

把 adapter 重建误差分解为：

```text
adapter_recon - gt
= (teacher_recon - gt) + (adapter_recon - teacher_recon)
```

其中：

- `teacher_recon - gt` 是 seed42 decoder 自身的重建误差。
- `adapter_recon - teacher_recon` 是 adapter code 残差经过 seed42 decoder 后引入的额外输出扰动。

全量 train 结果：

| adapter 实验 | teacher err NMSE | adapter-vs-teacher output NMSE | adapter total NMSE | mean code L2 | mean fc L2 | mean output delta L2 |
|---|---:|---:|---:|---:|---:|---:|
| `recon0.0_code1.0_lr1e-3` | -29.8357 | -29.7521 | -26.7854 | 0.3428 | 0.5268 | 0.0290 |
| `recon1.0_code0.0_lr2e-4` | -29.8357 | -13.2458 | -13.1767 | 101.1642 | 112.6851 | 0.2046 |
| `recon1.0_code1e-1_lr1e-3` | -29.8357 | -29.6416 | -26.7319 | 0.3436 | 0.5675 | 0.0293 |
| `recon1.0_code1e-2_lr1e-3` | -29.8357 | -28.4601 | -26.1048 | 0.3997 | 0.7242 | 0.0333 |
| `recon1.0_code1e-3_lr2e-4` | -29.8357 | **-30.2397** | **-27.1322** | **0.3414** | **0.5245** | **0.0267** |

关键点：

```text
teacher 自身误差约 -29.84 dB
adapter code 残差引入的输出扰动约 -30.24 dB
```

这两个误差能量是同一量级。即使它们近似不相关，能量相加后也会带来约 `3 dB` 的退化：

```text
两个 -30 dB 级别误差相加 -> 约 -27 dB
```

这正是观察到的 `-27.13 dB`。因此：

```text
0.34 的 code L2 残差虽然在 code cosine 上看很小，
但经过 decoder 后产生的输出扰动已经和 teacher 自身重建误差同量级，
足以形成 -24~-27 dB 的性能上限。
```

这也说明 adapter 要接近 seed42 baseline，不是把 code cosine 提到 0.999 就够，而是要把 decoder 输出扰动压到远低于 teacher 自身误差。粗略说，若希望总误差只比 teacher 差很小，`adapter_recon - teacher_recon` 应该到 `-40 dB` 量级，而不是现在的 `-30 dB` 量级。

## 4. fc_decoder 对 code 残差的放大

seed42 `TransNetDecoder` 的入口是：

```text
code(512) -> fc_decoder -> feature(2048) -> TransformerDecoder -> CSI
```

`fc_decoder.weight` 的奇异值统计：

| 指标 | 数值 |
|---|---:|
| max singular value | 4.4013 |
| p95 singular value | 1.6708 |
| median singular value | 1.1826 |
| mean singular value | 1.2139 |
| min singular value | 0.7415 |

对最佳 adapter：

```text
mean code L2 = 0.3414
mean fc feature L2 = 0.5245
rms fc/code amplification = 1.6353
mean output delta L2 = 0.0267
```

所以 code 残差在 decoder 入口的 `fc_decoder` 阶段已经被放大到更高维特征空间。后续 TransformerDecoder 再把这个扰动映射到 CSI 输出。当前瓶颈不是单纯“code 空间看起来近不近”，而是：

```text
code 残差是否落在 decoder 低敏感方向上。
```

普通 MSE code loss 对所有 code 维度等权，但 decoder 实际敏感度由 `fc_decoder` 和后续非线性决定；这解释了为什么 raw L2/cosine 很好看，重建仍然不够。

## 5. 全量 mean/var/cov 与误差谱

### 5.1 原始 seed2026 code 到 seed42 teacher code

作为参考，未经过 adapter 的 seed2026 code 与 seed42 teacher code：

| 指标 | 数值 |
|---|---:|
| raw L2 | 10.6111 |
| raw cosine | 0.6605 |
| mean L2 | 10.3866 |
| var L2 | 5.3721 |
| cov relative Frobenius | 0.4207 |
| error PCA top10 energy | 0.5850 |
| error effective rank | 28.91 |

说明原始跨 seed code 差异主要集中在较低维方向，且均值差异很大。

### 5.2 adapter 后 code 到 teacher code

| adapter 实验 | raw L2 | raw cos | mean L2 | var L2 | cov rel Fro | error top1 | error top10 | error top50 | eff rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `recon0.0_code1.0_lr1e-3` | 0.3428 | 0.9997 | 0.1077 | 0.0695 | 0.0145 | 0.0523 | 0.1690 | 0.4746 | 224.77 |
| `recon1.0_code0.0_lr2e-4` | 101.1642 | 0.2086 | 98.9929 | 80.7523 | 11.8336 | 0.2369 | 0.5667 | 0.8267 | 48.16 |
| `recon1.0_code1e-1_lr1e-3` | 0.3436 | 0.9997 | 0.1286 | 0.1211 | 0.0154 | 0.0445 | 0.1749 | 0.4887 | 216.32 |
| `recon1.0_code1e-2_lr1e-3` | 0.3997 | 0.9996 | 0.0759 | 0.1446 | 0.0218 | 0.1514 | 0.2845 | 0.5974 | 140.97 |
| `recon1.0_code1e-3_lr2e-4` | **0.3414** | **0.9998** | **0.0814** | 0.1214 | **0.0110** | 0.0949 | 0.2275 | 0.5358 | 182.35 |

解释：

- 加 code loss 后，adapter 已经极大降低 mean/var/cov 差异。
- 最佳 `1e-3` 的 covariance 相对误差最低，只有 `0.0110`。
- 但 error spectrum 仍然不是纯随机白噪声。最佳 `1e-3` 的 top10 主方向占 `22.75%`，top50 占 `53.58%`。这些残差方向很可能与 decoder 敏感方向重叠，因此会造成可见重建损失。

最佳 `1e-3` 的误差最大维度集中在低维 PCA/code 轴：

```text
top mean abs error dims:
0, 1, 2, 3, 4, 5, 6, 12, 17, 11
```

这很重要。aux PCA target 本身按主成分组织 code，低维往往承载更大能量和更重要的物理/统计信息。adapter 在这些维度上残留的绝对误差虽然数值不大，但对 decoder 可能更敏感。

## 6. 对 decoder feature / reconstruction consistency loss 的可行性分析

### 6.1 fc feature consistency

可以引入：

```text
L_fc = || fc_decoder(adapter_code) - fc_decoder(teacher_code) ||^2
```

对于 TransNet decoder，这相当于在 decoder 入口特征空间对齐，而不是在 code 空间等权对齐。

数学上，若 `W = fc_decoder.weight`，则：

```text
L_fc = || W (z_adapter - z_teacher) ||^2
     = (z_adapter - z_teacher)^T W^T W (z_adapter - z_teacher)
```

这就是一个 decoder-aware code loss。它会自动提高 decoder 敏感方向的惩罚权重，比普通 code MSE 更对症。

从当前结果看，最佳 adapter 的：

```text
mean code L2 = 0.3414
mean fc L2   = 0.5245
```

说明 fc 特征空间残差并不小。`L_fc` 是值得尝试的。

### 6.2 decoder reconstruction consistency

可以引入 teacher reconstruction consistency：

```text
teacher_recon = decoder42(teacher_code)
adapt_recon   = decoder42(adapter_code)
L_recon_teacher = || adapt_recon - teacher_recon ||^2
```

它与当前 `|| adapt_recon - gt ||^2` 不完全一样。当前训练中的 reconstruction loss 直接逼近 GT，而 teacher decoder 自身只能达到 `-29.84 dB` train NMSE。对 cross-seed adapter 来说，目标不是让 frozen seed42 decoder 超过自身能力，而是先复现 seed42 decoder 在 teacher code 上的行为。因此：

```text
L_recon_teacher
```

比直接 GT reconstruction 更适合作为 adapter 的稳定对齐目标。

### 6.3 intermediate decoder feature consistency

如果修改 decoder forward 以返回中间特征，可以加：

```text
L_mid = sum_l || h_l(adapter_code) - h_l(teacher_code) ||^2
```

候选层：

- `fc_decoder(code).view(B, seq_len, d_model)`。
- 第一层 TransformerDecoder 输出。
- 第二层 TransformerDecoder 输出或最终 norm 前输出。

优先级建议：

1. 先做 `fc_decoder` feature loss，最简单且理论清楚。
2. 再做 final reconstruction-to-teacher loss。
3. 最后再考虑中间 Transformer feature loss，因为需要改 decoder forward 或 hook，复杂度更高。

## 7. adapter 结构优化建议

当前 MLPAdapter 是：

```text
z_out = z_in + MLP(LayerNorm(z_in))
```

它的优点是 identity 初始化稳定；缺点是跨 seed 映射不一定是小残差。原始 seed2026 到 seed42 的 code L2 约 `10.61`，而最佳 adapter 的 `delta_from_source` 也约 `10.6`，说明 adapter 实际在做一个大变换。

### 7.1 显式 affine calibration

建议在 adapter 前或 adapter 内加入：

```text
z_calib = gamma * z + beta
```

其中 `gamma, beta` 可以：

- 直接根据 source/teacher train code 的 mean/std 初始化。
- 或作为可学习参数，初始化为统计匹配解。

这样先解决 mean/scale，再让 MLP 学非线性残差。当前不加 code loss 的 adapter 会把 norm 放大到 `103`，说明缺少显式统计约束时训练很容易跑到异常尺度。

### 7.2 线性 Procrustes / Ridge 初始化

由于 source 和 teacher code 已经高度相关，可以先拟合一个线性映射：

```text
z_teacher ≈ A z_source + b
```

候选：

- diagonal affine：参数少，防过拟合。
- orthogonal Procrustes：保范数，适合坐标旋转。
- ridge regression：表达力强，但要加正则。
- low-rank residual：`A = I + U V^T`，折中。

训练 adapter 时用这个线性映射初始化，再接 MLP residual：

```text
z_out = A z_in + b + residual_mlp(z_in)
```

这比现在从 identity residual 开始更贴近跨 seed 映射。

### 7.3 decoder-aware adapter

当前 adapter loss 主要是 code MSE 和 GT reconstruction。建议加入 decoder-aware 项：

```text
L = lambda_code * ||z_out - z_teacher||^2
  + lambda_fc   * ||W z_out - W z_teacher||^2
  + lambda_recT * ||D(z_out) - D(z_teacher)||^2
  + lambda_gt   * ||D(z_out) - gt||^2
```

其中 `D` 是 frozen seed42 decoder，`W` 是 `fc_decoder`。如果目标是接近 baseline，`lambda_recT` 和 `lambda_fc` 可能比继续增大 `lambda_code` 更有效。

### 7.4 误差谱约束

最佳 adapter 的误差 top50 主方向仍占 `53.58%`。可以考虑：

- 对 teacher-source 残差的主方向加权惩罚。
- 对低维 PCA/code 轴加更高权重，尤其当前 top error dims 集中在 `0..6` 等低维。
- 使用 per-dim teacher variance 归一化的 code loss：

```text
L = mean_i ((z_out_i - z_teacher_i)^2 / var_teacher_i)
```

或者相反，对高能主成分加权，取决于 decoder 敏感性分析结果。更稳妥的是用 `fc_decoder` 的 `W^T W` 做权重。

## 8. 还能做哪些辅助分析

### 8.1 decoder Jacobian 敏感性

对若干 batch 计算：

```text
|| J_decoder(z_teacher) * e_code || / || e_code ||
```

并比较：

- adapter 残差方向。
- 随机同范数残差方向。
- source-to-teacher 残差方向。

如果 adapter 残差方向比随机方向更敏感，说明 code MSE 没有避开 decoder 高敏感方向。

### 8.2 分层 feature 差异

记录：

```text
fc_decoder output diff
Transformer layer1 output diff
Transformer layer2 output diff
final output diff
```

看误差在哪一层被放大。如果误差主要在 `fc_decoder` 后已经成型，优先做 `L_fc`；如果 Transformer 层放大明显，再做 intermediate feature loss。

### 8.3 train/test gap

当前直接解码分析是 train 全量：

```text
best adapter train NMSE = -27.13 dB
best adapter log test NMSE = -24.753 dB
```

这说明 adapter 可能对 train code 对齐更好，对 test 泛化不足。可以重新导出 test code 或在 evaluate 时保存 test code，比较 train/test 的 code residual 分布。如果 test residual 明显更大，问题是 adapter 泛化；如果 test residual 相近，问题是 decoder 对测试样本更敏感。

### 8.4 checkpoint epoch 对齐

当前导出的 adapter 后 code 来自 `best_nmse.pth`。如果某些实验的 code loss 最优 epoch 与 NMSE 最优 epoch 不一致，则 best NMSE checkpoint 不一定是 code 最接近 teacher 的 checkpoint。可以额外保存 best code-loss checkpoint 或在训练中记录：

```text
test NMSE
code loss
fc feature loss
teacher reconstruction consistency
```

这样能判断 adapter 是被重建目标还是 code 目标拉偏。

## 9. 最终判断

1. seed42 teacher code 与 seed42 decoder checkpoint 是一致的；数据顺序也是 index-aligned，没有发现 teacher/decoder 不匹配问题。
2. adapter 加 code loss 后确实学到了从 seed2026 code 到 seed42 code 的大变换，code raw cosine 达到 `0.9998`。
3. 但 `0.34` 左右的 code L2 残差经过 seed42 decoder 后产生的输出扰动约 `-30 dB`，与 teacher 自身重建误差同量级，因此总 NMSE 只能到约 `-27 dB`。
4. 想接近 baseline，不能只继续提高普通 code MSE；需要 decoder-aware 的 loss 和结构，例如 `fc_decoder` feature loss、teacher reconstruction consistency、affine calibration、线性/低秩初始化。
5. 当前最优先的改进路径是：

```text
Affine calibration / linear initialized adapter
+ code MSE
+ fc_decoder feature MSE
+ decoder teacher reconstruction consistency
```

这个方向比单纯加深 MLP 或增大 `lambda_code` 更有针对性。
