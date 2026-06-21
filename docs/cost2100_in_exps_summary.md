# COST2100 indoor 新实验结果分析

本文重新分析当前 `exps/COST2100/in` 目录下实际存在的新实验结果。统计脚本只读取当前目录中的 `args.json` 与 `run.log`，不沿用旧文档中的 `frozen_decoder`、`lora` 等已不存在实验。

主分析先统计统一自编码训练结果，即 `exps/COST2100/in/seed*/<encoder>_<decoder>/`；随后单独分析 `adapter/shared_adapter/` 下的共享 adapter 实验。两类实验训练目标不同，因此不混在同一张性能主表中比较。

## 1. 实验范围

当前纳入主分析的实验共有 102 个，全部有最终测试 NMSE，日志也都跑满目标 epoch。

| 实验组 | 数量 | 说明 |
| --- | ---: | --- |
| `seed42` 架构网格 | 42 | 14 个 encoder x 3 个 decoder 的横向组合。 |
| `transnet_hybrid` 多 seed | 58 | 固定 `encoder=transnet`、`decoder=hybrid`，更换随机种子。 |
| `transnet_transnet` 多 seed | 2 | 非 seed42 的 `transnet_transnet` 重复实验。 |
| 共享 adapter | 1 | 固定 `seed42/transnet_hybrid` decoder，用一个 adapter 对齐多个 transnet encoder seed。 |

公共配置基本一致：COST2100 indoor，输入 `(2, 32, 32)`，`cr=4`，码字维度 `512`，多数实验训练 `400` epoch，优化器配置来自各实验 `args.json`。

本文数值来自日志最终测试结果。若用于论文表格，建议再用当前代码统一加载 `best_nmse.pth` 重新 evaluate，确保指标实现和 checkpoint 选择完全一致。

## 2. 总体结论

- 当前最强单次结果是 `seed42/clnet_transnet`，final NMSE 为 `-31.542 dB`。
- `seed42/convnext_transnet` 几乎持平，final NMSE 为 `-31.483 dB`。
- 在 `seed42` 架构网格中，`transnet` decoder 的平均表现和上限都最好；`hybrid` 次之；`cnn_residual` 明显落后。
- `transnet_hybrid` 多 seed 波动较大，非 seed42 的 58 个结果均值为 `-25.960 dB`，标准差为 `0.809 dB`，最好 `-27.562 dB`，最差 `-24.067 dB`。
- `seed42/transnet_hybrid=-28.407 dB` 明显强于非 seed42 多 seed 均值，不宜只用它代表该结构的平均水平。

## 3. `seed42` 架构网格

`seed42` 包含 42 个实验，覆盖 14 个 encoder 与 3 个 decoder。

| decoder | n | final NMSE mean | std | best final | worst final |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cnn_residual` | 14 | -19.019 | 1.763 | -20.340 | -13.164 |
| `hybrid` | 14 | -25.934 | 2.057 | -29.050 | -22.387 |
| `transnet` | 14 | -27.840 | 2.935 | -31.542 | -21.884 |

### 3.1 Top 15 组合

| rank | seed | encoder | decoder | final NMSE | best NMSE | best epoch |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1 | 42 | `clnet` | `transnet` | -31.542 | -31.542 | 400 |
| 2 | 42 | `convnext` | `transnet` | -31.483 | -31.483 | 400 |
| 3 | 42 | `mlp_ae` | `transnet` | -31.077 | -31.113 | 390 |
| 4 | 42 | `mlp_mixer` | `transnet` | -30.740 | -30.740 | 400 |
| 5 | 42 | `swin` | `transnet` | -29.995 | -30.377 | 390 |
| 6 | 42 | `csinet` | `transnet` | -29.067 | -29.417 | 370 |
| 7 | 42 | `convnext` | `hybrid` | -29.050 | -29.050 | 400 |
| 8 | 42 | `attention_cnn` | `transnet` | -28.489 | -28.489 | 400 |
| 9 | 42 | `transnet` | `hybrid` | -28.407 | -28.407 | 400 |
| 10 | 42 | `mlp_mixer` | `hybrid` | -28.133 | -28.853 | 330 |
| 11 | 42 | `transnet` | `transnet` | -28.126 | -28.126 | 400 |
| 12 | 42 | `swin` | `hybrid` | -27.890 | -28.688 | 390 |
| 13 | 42 | `clnet` | `hybrid` | -27.880 | -28.412 | 370 |
| 14 | 42 | `cnn` | `transnet` | -27.205 | -27.392 | 390 |
| 15 | 42 | `attention_cnn` | `hybrid` | -26.658 | -26.675 | 390 |

### 3.2 `transnet` decoder

| encoder | final NMSE | best NMSE | best epoch |
| --- | ---: | ---: | ---: |
| `clnet` | -31.542 | -31.542 | 400 |
| `convnext` | -31.483 | -31.483 | 400 |
| `mlp_ae` | -31.077 | -31.113 | 390 |
| `mlp_mixer` | -30.740 | -30.740 | 400 |
| `swin` | -29.995 | -30.377 | 390 |
| `csinet` | -29.067 | -29.417 | 370 |
| `attention_cnn` | -28.489 | -28.489 | 400 |
| `transnet` | -28.126 | -28.126 | 400 |
| `cnn` | -27.205 | -27.392 | 390 |
| `resnet` | -25.631 | -25.631 | 400 |
| `cbam_cnn` | -25.137 | -25.137 | 400 |
| `crnet` | -25.001 | -25.023 | 390 |
| `sparse_resnet` | -24.379 | -24.379 | 400 |
| `dscnn` | -21.884 | -21.884 | 400 |

### 3.3 `hybrid` decoder

| encoder | final NMSE | best NMSE | best epoch |
| --- | ---: | ---: | ---: |
| `convnext` | -29.050 | -29.050 | 400 |
| `transnet` | -28.407 | -28.407 | 400 |
| `mlp_mixer` | -28.133 | -28.853 | 330 |
| `swin` | -27.890 | -28.688 | 390 |
| `clnet` | -27.880 | -28.412 | 370 |
| `attention_cnn` | -26.658 | -26.675 | 390 |
| `sparse_resnet` | -25.659 | -25.659 | 400 |
| `mlp_ae` | -25.524 | -25.833 | 350 |
| `cnn` | -25.275 | -25.616 | 390 |
| `resnet` | -24.755 | -24.880 | 390 |
| `crnet` | -24.691 | -24.822 | 370 |
| `cbam_cnn` | -24.178 | -24.685 | 380 |
| `dscnn` | -22.593 | -22.593 | 400 |
| `csinet` | -22.387 | -22.447 | 390 |

### 3.4 `cnn_residual` decoder

| encoder | final NMSE | best NMSE | best epoch |
| --- | ---: | ---: | ---: |
| `swin` | -20.340 | -20.599 | 390 |
| `convnext` | -20.313 | -20.313 | 400 |
| `transnet` | -20.254 | -20.596 | 390 |
| `cnn` | -20.019 | -20.057 | 380 |
| `mlp_mixer` | -19.895 | -21.023 | 350 |
| `mlp_ae` | -19.766 | -19.864 | 390 |
| `sparse_resnet` | -19.654 | -19.654 | 400 |
| `attention_cnn` | -19.493 | -19.591 | 370 |
| `resnet` | -19.176 | -19.329 | 360 |
| `crnet` | -19.090 | -19.746 | 360 |
| `clnet` | -18.514 | -18.839 | 350 |
| `cbam_cnn` | -18.404 | -19.134 | 390 |
| `dscnn` | -18.185 | -18.185 | 400 |
| `csinet` | -13.164 | -13.164 | 400 |

从横向结果看，`transnet` decoder 与强 encoder 的组合最有优势，前 6 名中全部使用 `transnet` decoder。`hybrid` decoder 的最好结果是 `convnext_hybrid=-29.050 dB`，其次是 `transnet_hybrid=-28.407 dB`。`cnn_residual` decoder 的最好结果也只有约 `-20.340 dB`，说明其解码容量不足以支撑当前 `cr=4` 的高质量重建。

## 4. `transnet_hybrid` 多 seed

非 seed42 的 `transnet_hybrid` 共有 58 个重复实验。

| 指标 | 数值 |
| --- | ---: |
| n | 58 |
| mean | -25.960 |
| std | 0.809 |
| min / best | -27.562 |
| 25% | -26.527 |
| median | -26.018 |
| 75% | -25.394 |
| max / worst | -24.067 |

### 4.1 最好 seed

| seed | final NMSE | best NMSE | best epoch |
| ---: | ---: | ---: | ---: |
| 3407 | -27.562 | -27.562 | 400 |
| 796 | -27.478 | -27.478 | 400 |
| 223 | -27.447 | -27.447 | 400 |
| 1234 | -27.195 | -27.195 | 400 |
| 12867 | -26.931 | -26.931 | 400 |
| 1115 | -26.820 | -26.852 | 370 |
| 17669 | -26.815 | -27.034 | 380 |
| 9856 | -26.790 | -27.020 | 380 |
| 1474 | -26.699 | -26.699 | 400 |
| 1014 | -26.665 | -26.665 | 400 |
| 28544 | -26.657 | -26.657 | 400 |
| 23685 | -26.620 | -26.620 | 400 |
| 31039 | -26.567 | -26.567 | 400 |
| 517 | -26.532 | -26.532 | 400 |
| 1480 | -26.529 | -26.529 | 400 |

### 4.2 最差 seed

| seed | final NMSE | best NMSE | best epoch |
| ---: | ---: | ---: | ---: |
| 4442 | -24.067 | -26.487 | 360 |
| 404 | -24.142 | -24.676 | 390 |
| 30243 | -24.333 | -24.333 | 400 |
| 2048 | -24.564 | -24.564 | 400 |
| 287 | -24.613 | -25.707 | 390 |
| 14115 | -24.745 | -26.163 | 390 |
| 424 | -24.854 | -26.047 | 350 |
| 520 | -24.860 | -25.757 | 380 |
| 13498 | -24.912 | -25.351 | 390 |
| 31306 | -25.031 | -25.692 | 390 |
| 9436 | -25.238 | -26.553 | 380 |
| 17287 | -25.322 | -26.442 | 390 |

多 seed 结果说明 `transnet_hybrid` 对随机初始化比较敏感。最好 seed `3407` 达到 `-27.562 dB`，但仍不如 `seed42/transnet_hybrid` 的 `-28.407 dB`。部分 seed 的 best checkpoint 明显好于 final checkpoint，例如 `seed4442` best 为 `-26.487 dB`，final 只有 `-24.067 dB`，说明后期训练存在回退，正式汇报时应优先比较 `best_nmse.pth`。

## 5. `transnet_transnet` 重复实验

非 seed42 的 `transnet_transnet` 只有 2 个重复结果：

| seed | final NMSE | best NMSE | best epoch |
| ---: | ---: | ---: | ---: |
| 3407 | -28.520 | -28.520 | 400 |
| 2026 | -28.180 | -28.180 | 400 |

加上 `seed42/transnet_transnet=-28.126 dB`，该结构目前稳定在约 `-28 dB`，比多数 `transnet_hybrid` seed 更强，但样本数仍少于 `transnet_hybrid`，后续需要更多 seed 才能判断稳定性。

## 6. 共享 adapter 实验

目录 `exps/COST2100/in/adapter/shared_adapter/decoder_seed42/transnet_hybrid` 是一个独立的 adapter 对齐实验。它不属于普通端到端联合训练，而是加载 `seed42/transnet_hybrid/checkpoints/best_nmse.pth` 中的 hybrid decoder，并让多个 transnet encoder seed 通过一个共享 adapter 适配这个 decoder。

关键配置如下：

| 项目 | 数值 |
| --- | --- |
| decoder | `hybrid` |
| pretrained decoder | `exps/COST2100/in/seed42/transnet_hybrid/checkpoints/best_nmse.pth` |
| encoder | `transnet` |
| encoder seeds | 18 个有效 seed：0, 223, 314, 404, 424, 520, 644, 796, 1014, 1024, 1115, 1234, 1337, 2026, 2048, 2718, 3407, 31415 |
| adapter hidden | 16 |
| adapter blocks | 2 |
| batch size | 1024 |
| epochs | 400 |

整体结果：

| 指标 | 数值 |
| --- | ---: |
| final loss | 8.9907e-06 |
| final NMSE | -17.014 |
| best NMSE | -17.017 |
| best epoch | 399 |

最终各 encoder seed 的分项结果如下：

| encoder seed | loss | NMSE |
| ---: | ---: | ---: |
| 314 | 8.9204e-06 | -17.047 |
| 644 | 8.9527e-06 | -17.032 |
| 404 | 8.9613e-06 | -17.028 |
| 520 | 8.9603e-06 | -17.028 |
| 2048 | 8.9646e-06 | -17.026 |
| 1115 | 8.9812e-06 | -17.018 |
| 796 | 8.9844e-06 | -17.017 |
| 31415 | 8.9837e-06 | -17.017 |
| 223 | 8.9859e-06 | -17.016 |
| 1014 | 8.9871e-06 | -17.015 |
| 2718 | 8.9876e-06 | -17.015 |
| 1337 | 8.9949e-06 | -17.011 |
| 1234 | 8.9982e-06 | -17.010 |
| 0 | 9.0042e-06 | -17.007 |
| 1024 | 9.0066e-06 | -17.006 |
| 2026 | 9.0265e-06 | -16.996 |
| 424 | 9.0389e-06 | -16.990 |
| 3407 | 9.0937e-06 | -16.964 |

分项 NMSE 的均值为 `-17.014 dB`，标准差只有 `0.018 dB`。这说明共享 adapter 把不同 encoder seed 的输出拉到了非常接近的水平，但这个水平明显低于端到端联合训练的 `transnet_hybrid`。作为对比，非 seed42 的 `transnet_hybrid` 多 seed 均值是 `-25.960 dB`，而 `seed42/transnet_hybrid` 是 `-28.407 dB`。

adapter metrics 也显示 adapter 修正幅度很大：最终 `adapter_residual_ratio` 在各 seed 上约为 `1.39` 到 `1.76`，均值约 `1.58`。这意味着 adapter 输出的残差量级已经超过原始 code 的量级，模型不是在做轻微校准，而是在强行重写码字表示。结合 `-17 dB` 的结果，可以判断当前共享 adapter 很难把多个独立 encoder seed 的码字空间可靠对齐到同一个固定 decoder。

这个实验的意义更偏向“码字空间可对齐性验证”：它证明多个 encoder seed 共享一个小 adapter 时会得到稳定但偏低的重建质量。它不应作为主模型性能结果汇报，也不能直接替代端到端训练。

## 7. 建议

1. 若目标是追求当前最佳 NMSE，优先复现实验 `clnet_transnet` 和 `convnext_transnet`，并至少补 3 到 5 个 seed。
2. 若目标是报告稳健基线，`transnet_hybrid` 应报告多 seed 均值、标准差和 best checkpoint 结果，不能只报 `seed42`。
3. `cnn_residual` decoder 当前没有继续作为主线 decoder 的必要，除非后续专门改结构增强全局重建能力。
4. 对 adapter 路线，如果还要继续做，应优先比较“单 seed 单 adapter”和“共享 adapter”，并记录 adapter 前的直接拼接 NMSE，否则无法判断 adapter 真实增益。
5. 建议生成一份统一 evaluate 表：对所有 `checkpoints/best_nmse.pth` 用当前 `utils/statics.py::evaluator()` 重跑测试集，避免日志时代差异影响结论。
