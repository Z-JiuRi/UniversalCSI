# COST2100/in encoder canonical 实验分析

本文分析 `exps/COST2100/in/encoder_canonical` 下已有实验，重点关注两个问题：

1. encoder canonical 约束是否提升自编码重建性能。
2. 不同 seed 训练得到的码字是否更接近同一个坐标系，以及这种接近是否足以支持 cross-seed encoder/decoder + adapter。

分析对象包括：

- 无规范化 baseline。
- PCA/DCT auxiliary target。
- 固定随机正交投影 Q。
- 固定 Q + 低秩残差。
- 固定 codebook。
- PCA auxiliary target + code mean/var/cov 正则。
- aux PCA encoder/decoder 上的 adapter 实验。

实验数据来自 `args.json`、`run.log` 和 `codewords/train_code.pt`。码字一致性统计使用同索引 train codeword 的采样比较；因此它反映的是“同一个 CSI 样本在不同 seed encoder 下的 code 是否处于相近坐标系”。

## 1. 总体结论

当前最有效的方案仍然是：

```text
aux_pca_1e-3
```

它的配置本质是：

```text
canonical_head = none
anchor_target  = pca
lambda_anchor  = 1e-3
anchor_loss    = mse
```

也就是不改变 encoder head 结构，只在训练时给 encoder 输出的 code 加 PCA auxiliary target。这个方案同时做到：

- 重建性能最好：5 seed 平均 best NMSE 为 `-29.5082 dB`。
- 不同 seed 的 centered code 高度一致：centered cosine 约 `0.9937`。
- 相比 baseline，大幅减轻坐标旋转问题。

但是它还没有达到“不同 seed 的 encoder/decoder 可以直接互换”的程度。adapter 实验中，即使使用 aux PCA 训练出的 encoder/decoder，seed2026 encoder 接 seed42 decoder 的 best NMSE 最高也只有约 `-24.753 dB`，明显低于同 seed 自编码器的 `-29 dB` 级别。

新跑的 `aux_pca + code_reg` 系列说明：当前 mean/var/cov 正则会让码字 raw 分布更一致，但代价是重建性能明显退化。也就是说，它确实更“规范”，但过强地限制了 encoder/decoder 的表达自由度。

## 2. 重建性能汇总

### 2.1 所有主实验

| scheme                    | seed 数 | codeword | mean best NMSE |    std |     best |    worst | 主要配置                      |
| ------------------------- | ------: | -------: | -------------: | -----: | -------: | -------: | ----------------------------- |
| `aux_pca_1e-3`            |       5 |        5 |   **-29.5082** | 0.2446 | -29.8890 | -29.1810 | PCA aux, `lambda_anchor=1e-3` |
| `baseline`                |       5 |        5 |       -27.3056 | 0.7555 | -28.1800 | -26.2110 | 无 canonical                  |
| `codebook1024`            |       5 |        5 |       -24.8744 | 0.5279 | -25.3080 | -23.8380 | fixed codebook                |
| `aux_dct_1e-3`            |       5 |        5 |       -24.6118 | 0.2567 | -25.0720 | -24.3060 | DCT aux, `lambda_anchor=1e-3` |
| `aux_pca_5e-3_code_reg`   |       3 |        3 |       -25.4730 | 0.1256 | -25.6170 | -25.3110 | PCA aux + mean/var/cov        |
| `aux_pca_1e-3_code_reg`   |       3 |        3 |       -24.6980 | 0.1336 | -24.8330 | -24.5160 | PCA aux + mean/var/cov        |
| `aux_pca_1e-2_code_reg`   |       3 |        3 |       -23.7727 | 2.1607 | -25.3080 | -20.7170 | PCA aux + mean/var/cov        |
| `fixed_q_rank16_code_reg` |       5 |        5 |       -22.3052 | 0.2279 | -22.7410 | -22.1320 | fixed Q + low-rank + code reg |
| `fixed_q_rank16`          |       5 |        5 |       -22.2076 | 0.3725 | -22.8460 | -21.8630 | fixed Q + low-rank            |
| `fixed_q`                 |       5 |        5 |       -20.8876 | 0.5835 | -21.6710 | -20.0250 | fixed Q                       |
| `codebook1024_dct_reg`    |       5 |        5 |       -14.9098 | 0.0964 | -15.0730 | -14.7990 | codebook + DCT/code reg       |
| `fixed_q_rank16_pca`      |       5 |        5 |       -13.0734 | 0.0664 | -13.1230 | -12.9440 | fixed Q + low-rank + PCA      |

### 2.2 只比较共同 seed 42/2026/3407

由于新跑的 `aux_pca_*_code_reg` 只有 3 个 seed，下面只比较共同 seed：

| scheme                  | seed 42 | seed 2026 | seed 3407 |         mean |
| ----------------------- | ------: | --------: | --------: | -----------: |
| `baseline`              | -28.126 |   -28.180 |   -26.868 |     -27.7247 |
| `aux_pca_1e-3`          | -29.528 |   -29.620 |   -29.889 | **-29.6790** |
| `aux_pca_1e-3_code_reg` | -24.516 |   -24.745 |   -24.833 |     -24.6980 |
| `aux_pca_5e-3_code_reg` | -25.491 |   -25.617 |   -25.311 |     -25.4730 |
| `aux_pca_1e-2_code_reg` | -25.308 |   -25.293 |   -20.717 |     -23.7727 |

这个公平比较更直接：`aux_pca_1e-3` 明显最好；加 mean/var/cov 正则后，重建性能从约 `-29.68 dB` 掉到 `-24~-25 dB`。`lambda_anchor=5e-3` 比 `1e-3` 和 `1e-2` 的 code_reg 版本更好，但仍远差于纯 aux PCA。

## 3. 码字坐标一致性

码字比较指标：

- `raw_l2`：同一样本不同 seed code 的原始 L2 距离，越小越接近。
- `raw_cos`：原始 code cosine，越大越接近。
- `centered_l2`：去掉各自 batch 均值后的 L2。
- `centered_cos`：去均值后的 cosine，主要反映样本间变化方向是否一致。
- `centroid_l2`：不同 seed code 均值向量的距离。
- `procrustes`：允许最佳正交旋转/反射后仍剩余的相对误差，越小表示子空间结构越接近。

| scheme                    | seeds |  raw L2 | raw cos | centered L2 | centered cos | centroid L2 | Procrustes |
| ------------------------- | ----- | ------: | ------: | ----------: | -----------: | ----------: | ---------: |
| `baseline`                | 5     | 23.6203 | -0.0064 |     21.3784 |      -0.0035 |      9.8605 |     0.3394 |
| `aux_pca_1e-3`            | 5     |  6.8040 |  0.7802 |      1.5444 |       0.9937 |      6.4900 |     0.0929 |
| `aux_dct_1e-3`            | 5     |  4.6521 |  0.9878 |      3.0204 |       0.9985 |      3.3066 |     0.0908 |
| `aux_pca_1e-3_code_reg`   | 3     |  1.6975 |  0.9965 |      1.6904 |       0.9966 |      0.1706 |     0.0915 |
| `aux_pca_5e-3_code_reg`   | 3     |  1.6905 |  0.9964 |      1.6887 |       0.9964 |      0.1173 |     0.0972 |
| `aux_pca_1e-2_code_reg`   | 3     |  1.5057 |  0.9968 |      1.5054 |       0.9968 |      0.0603 |     0.0880 |
| `fixed_q`                 | 5     | 24.9033 |  0.0704 |     14.5397 |       0.2447 |     20.0310 |     0.5432 |
| `fixed_q_rank16`          | 5     | 13.5903 |  0.2619 |     12.1034 |       0.3095 |      6.0226 |     0.4407 |
| `fixed_q_rank16_code_reg` | 5     | 27.1249 |  0.2620 |     27.1177 |       0.2621 |      0.6040 |     0.6298 |
| `fixed_q_rank16_pca`      | 5     |  3.8433 |  0.8837 |      3.8391 |       0.8842 |      0.1801 |     1.0784 |
| `codebook1024`            | 5     |  0.1208 |  0.9019 |      0.0940 |      -0.0013 |      0.0748 |     0.4098 |
| `codebook1024_dct_reg`    | 5     |  0.0150 |  0.9921 |      0.0146 |       0.9919 |      0.0032 |     0.1048 |

从这个表看：

- `baseline` 的 raw/centered cosine 都接近 0，说明不同 seed code 基本处于互不对齐的坐标系。
- `aux_pca_1e-3` 把 centered cosine 提到 `0.9937`，说明同一样本的 code 变化方向已高度一致。
- `aux_pca + code_reg` 进一步把 raw code 的均值和尺度拉近，raw cosine 到 `0.996+`，centroid L2 降到 `0.06~0.17`。
- 但 `aux_pca + code_reg` 的重建性能明显下降，说明当前统计正则过强，或者 target variance/cov 的形式不适合这个重建任务。
- `codebook` 类方案 raw L2 极小不能直接解读为好，因为它可能是 code 坍缩导致的。

## 4. 码字分布规模与坍缩风险

| scheme                    | mean norm | norm std | mean dim std | mean abs | centroid norm |
| ------------------------- | --------: | -------: | -----------: | -------: | ------------: |
| `baseline`                |   16.6441 |   2.6533 |       0.6730 |   0.5853 |        6.9094 |
| `aux_pca_1e-3`            |   12.9459 |   2.1991 |       0.3870 |   0.3505 |        5.3003 |
| `aux_dct_1e-3`            |   29.5635 |   5.9991 |       1.3226 |   0.9920 |        2.6770 |
| `aux_pca_1e-3_code_reg`   |   19.8849 |   9.0700 |       0.9322 |   0.6930 |        0.4639 |
| `aux_pca_5e-3_code_reg`   |   19.2693 |   9.7288 |       0.9254 |   0.6851 |        0.3997 |
| `aux_pca_1e-2_code_reg`   |   19.1652 |   9.7488 |       0.9234 |   0.6859 |        0.3797 |
| `fixed_q`                 |   18.2958 |   0.7202 |       0.5274 |   0.6481 |       13.8400 |
| `fixed_q_rank16`          |   11.1958 |   1.6865 |       0.4609 |   0.3927 |        4.3413 |
| `fixed_q_rank16_code_reg` |   22.3626 |   1.4017 |       0.9510 |   0.7593 |        0.4772 |
| `fixed_q_rank16_pca`      |    5.0190 |   1.1071 |       0.0924 |   0.0651 |        0.1497 |
| `codebook1024`            |    0.2744 |   0.0041 |       0.0030 |   0.0077 |        0.2658 |
| `codebook1024_dct_reg`    |    0.1201 |   0.0091 |       0.0052 |   0.0041 |        0.0282 |

这里可以解释几个现象：

1. `codebook1024` 和 `codebook1024_dct_reg` 的 code norm 极小、维度 std 极小，基本可以视为码字动态范围坍缩。因此 raw L2 很小不是坐标系成功，而是表达力不足。
2. `fixed_q_rank16_pca` 的 mean dim std 只有 `0.0924`，也明显压低了 code 动态范围，和 `-13 dB` 的差重建性能一致。
3. `aux_pca + code_reg` 的 centroid norm 很小，说明 mean 正则确实生效；但 mean norm 和 norm std 都比纯 aux PCA 更大，表明当前 var/cov target 可能把 code 分布推到了 decoder 不容易使用的尺度。

## 5. 为什么 aux PCA 最有效

普通自编码器存在典型的 encoder/decoder 共同适配自由度：

```text
z' = A z
W' = W A^{-1}
```

只要 encoder 和 decoder 一起训练，很多 latent 坐标变换都能被 decoder 第一层吸收。因此不同 seed 会学到不同 code 坐标系。

`aux_pca_1e-3` 的优势是它没有硬改模型结构，只是给 encoder 输出提供公共 PCA 方向作为软 anchor。这样它能削弱 seed 间坐标旋转，同时保留 encoder/decoder 足够的表达能力。

相比之下：

- `fixed_q` 直接固定 feature 到 code 的主投影，表达力太硬。
- `fixed_q_rank16` 加低秩残差后略好，但仍不足。
- `codebook` 通过固定 codebook 限制 code 空间，容易出现动态范围过小。
- `DCT` 是公共物理 basis，但和 TransNet latent 空间不如 PCA 匹配。
- `aux_pca + code_reg` 更接近 raw code 分布对齐，但当前正则强度和 target 形式会显著损害重建。

## 6. 为什么 aux PCA 后 adapter 仍然不够好

adapter 实验都在：

```text
encoder: aux_pca_1e-3 seed2026
decoder: aux_pca_1e-3 seed42
adapter: mlp
```

日志结果：

| adapter 实验               | lambda_recon | lambda_code |   lr | before train NMSE |   best NMSE | best epoch | last test |
| -------------------------- | -----------: | ----------: | ---: | ----------------: | ----------: | ---------: | --------: |
| `recon1.0_code0.0_lr2e-4`  |          1.0 |         0.0 | 2e-4 |            3.9396 |     -12.571 |        400 |   -12.571 |
| `recon1.0_code1e-3_lr2e-4` |          1.0 |        1e-3 | 2e-4 |            3.9396 | **-24.753** |        330 |   -24.742 |
| `recon1.0_code1e-2_lr1e-3` |          1.0 |        1e-2 | 1e-3 |            3.9396 |     -24.084 |         60 |   -23.735 |
| `recon1.0_code1e-1_lr1e-3` |          1.0 |        1e-1 | 1e-3 |            3.9396 |     -24.342 |        120 |   -24.052 |
| `recon0.0_code1.0_lr1e-3`  |          0.0 |         1.0 | 1e-3 |            3.9396 |     -24.507 |        100 |   -24.194 |

重新导出 adapter 后 codeword 后，可以直接比较 adapter 输出 code 与 decoder seed42 的 teacher code：

| adapter 实验               |   best NMSE | raw L2 to teacher | raw cos to teacher | centered L2 to teacher | centered cos to teacher | delta from source |
| -------------------------- | ----------: | ----------------: | -----------------: | ---------------------: | ----------------------: | ----------------: |
| `recon1.0_code0.0_lr2e-4`  |     -12.571 |          101.1790 |             0.2085 |                21.2586 |                  0.8101 |          100.9554 |
| `recon1.0_code1e-3_lr2e-4` | **-24.753** |        **0.3385** |         **0.9998** |                 0.3267 |              **0.9998** |           10.5985 |
| `recon1.0_code1e-2_lr1e-3` |     -24.084 |            0.3973 |             0.9996 |                 0.3890 |                  0.9996 |           10.6236 |
| `recon1.0_code1e-1_lr1e-3` |     -24.342 |            0.3410 |             0.9997 |             **0.3133** |              **0.9998** |           10.6736 |
| `recon0.0_code1.0_lr1e-3`  |     -24.507 |            0.3409 |             0.9997 |                 0.3213 |                  0.9997 |           10.6557 |

这里的 `delta from source` 是 adapter 后 code 与原始 seed2026 encoder code 的平均 L2 距离。它约为 `10.6`，说明 adapter 确实做了大幅 code 变换，不再只是 identity。

更新后的结论：

- 不加 code loss 的 adapter 很差，只到 `-12.571 dB`。
- 不加 code loss 时，adapter 后 code 明显偏离 teacher code：raw L2 达到 `101.1790`，raw cosine 只有 `0.2085`，而且 code norm 被放大到约 `103`。这说明纯重建 loss 下 adapter 学到了 decoder 不容易稳定使用的异常 code 分布。
- 加 teacher code 后，adapter 输出已经几乎贴到 decoder seed42 的 teacher code：raw cosine 约 `0.9996~0.9998`，raw L2 只有 `0.34~0.40`。
- 但即使 adapter 后 code 已经非常接近 teacher code，NMSE 仍只有约 `-24.5 dB`，没有恢复到 seed42 自编码器自身的 `-29.5 dB` 级别。这说明 adapter 失败不能再简单归因于“一阶 raw code 没对齐”。
- 但继续提高 `lambda_code` 到 `1e-2/1e-1/1.0` 没有超过 `1e-3`，可能是学习率、loss scale 或 adapter 表达形式的瓶颈。
- 更可能的瓶颈包括：code MSE 看起来很小但 decoder 对残差极敏感；adapter best checkpoint 的重建最优点和 code 最优点不完全一致；teacher code 与 decoder checkpoint、数据顺序或导出 checkpoint 之间仍需更严格核对；以及 decoder 对高阶结构或逐样本细节误差非常敏感。

还有一个实现细节必须注意：早期导出逻辑保存的是：

```python
encoder_output = self.model.encoder(sparse_gt)
```

而不是：

```python
encoder_output = self.model.encode(sparse_gt)
```

因此旧 adapter 实验目录下导出的 `codewords/train_code.pt` 是 adapter 前的 encoder code，不是 adapter 后 decoder 实际接收的 code。实测这些 adapter codeword 与源 `aux_pca_1e-3/seed2026` 的 codeword 逐字节相同。

当前代码已将导出函数改为 `Trainer.save_codewords()`，并改为保存 `self.model.encode(sparse_gt)`。重新导出后，adapter 实验的 `codewords/train_code.pt` 才能用于判断 adapter 是否学到了 code 对齐。

## 7. 对新 code_reg 实验的判断

新实验的 code 正则配置为：

```text
lambda_code_mean = 1e-3
lambda_code_var  = 1e-3
lambda_code_cov  = 1e-4
lambda_code_l1   = 0
```

它们确实让 raw code 坐标显著更一致：

```text
aux_pca_1e-3:
  raw cos       = 0.7802
  centroid L2   = 6.4900
  centered cos  = 0.9937
  mean NMSE     = -29.5082

aux_pca_1e-3_code_reg:
  raw cos       = 0.9965
  centroid L2   = 0.1706
  centered cos  = 0.9966
  mean NMSE     = -24.6980
```

这说明 code_reg 不是没有效果；它的问题是把“坐标一致性”推进得太硬，牺牲了重建信息。

`aux_pca_5e-3_code_reg` 的 NMSE 最好，约 `-25.4730 dB`，但仍远低于纯 `aux_pca_1e-3`。`aux_pca_1e-2_code_reg` 在 seed3407 上退化到 `-20.717 dB`，说明更大的 PCA anchor 加当前 code_reg 并不稳定。

## 8. 建议的下一步实验

### 8.1 不建议继续投入的方向

不建议优先继续：

- `fixed_q`
- `fixed_q_rank16`
- `fixed_q_rank16_pca`
- `codebook1024`
- `codebook1024_dct_reg`

这些方案在当前实现下要么表达力不足，要么 code 坍缩，要么重建性能严重退化。

### 8.2 建议保留的主线

建议主线仍然围绕 `aux_pca`，但不要直接使用当前强度的 `mean/var/cov` 正则。

更合理的尝试是减弱统计正则：

```bash
scheme=aux_pca \
lambda_anchor=1e-3 \
lambda_code_mean=1e-4 \
lambda_code_var=1e-4 \
lambda_code_cov=1e-5 \
exp_name=COST2100/in/encoder_canonical/aux_pca_1e-3_code_reg_weak/seed2026_transnet_transnet \
seed=2026 gpu=0 \
bash scripts/train_encoder_canonical.sh
```

或者先只开 mean：

```bash
scheme=aux_pca \
lambda_anchor=1e-3 \
lambda_code_mean=1e-4 \
lambda_code_var=0 \
lambda_code_cov=0 \
exp_name=COST2100/in/encoder_canonical/aux_pca_1e-3_mean_reg/seed2026_transnet_transnet \
seed=2026 gpu=0 \
bash scripts/train_encoder_canonical.sh
```

理由：当前 aux PCA 最大的问题是 raw centroid 仍不一致；先只约束 mean，风险比同时约束 var/cov 小。

### 8.3 adapter 方向

adapter 目前最佳为：

```text
lambda_recon=1.0
lambda_code=1e-3
lr=2e-4
best NMSE=-24.753 dB
```

继续提高 `lambda_code` 没有改善。因此下一步不是简单加大 code loss，而是：

1. 核对 seed42 teacher code 与 seed42 decoder checkpoint 是否严格来自同一个 best checkpoint、同一份 train 数据顺序和同一套导出逻辑。
2. 用 seed42 decoder 分别输入 teacher code 和 adapter 后 code，直接比较两者经过 decoder 后的重建差异，判断 `0.34` 左右的 code L2 残差是否足以造成 `-24 dB` 的性能上限。
3. 分析 adapter 后 code 与 teacher code 的逐维 mean/var/cov 和误差谱，而不只看全局 L2/cosine。
4. 尝试在 adapter loss 中加入 decoder feature 或 reconstruction consistency，而不是继续单纯加大 code MSE 权重。

当前 adapter 后 codeword 已经重新导出，可以继续用这些文件分析 adapter code 对齐。旧结论里“adapter codeword 是 adapter 前 code”的限制只适用于重新导出之前的文件。

## 9. 最终判断

当前实验已经比较清楚：

```text
aux_pca_1e-3 是目前唯一同时提升重建性能和显著减轻 seed 间 code 坐标旋转的方案。
```

但是：

```text
码字 centered 方向一致 != decoder 可互换。
```

decoder 可互换要求 raw code 的完整分布都足够接近。`aux_pca + code_reg` 证明 raw 分布可以被拉近，但当前正则太伤重建；重新导出的 adapter 后 code 进一步说明，加 teacher code 后 adapter 已经能把 code 拉到非常接近 seed42 teacher code 的位置，但这仍不足以恢复同 seed decoder 的重建性能。

因此后续方向应从“更硬的固定坐标结构”转向“保留 aux PCA 表达力，同时温和约束 raw code 的低阶统计”，并基于已重新导出的 adapter 后 code 继续做残差敏感性和 teacher/decoder 一致性诊断。

