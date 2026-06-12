# COST2100 in 场景实验总分析

本文分析范围是 `exps/COST2100/in` 下当前可见的全部实验目录。依据来自每个实验的 `args.json` 与 `run.log`，并结合当前代码中 `UniversalCSI`、decoder、训练/测试流程与 LoRA 逻辑理解实验含义。

需要先说明一个指标口径问题：本文中的数值来自各实验 `run.log`，因此反映的是实验当时写入日志的 NMSE。当前代码中的测试 NMSE 已经改成全测试集 total NMSE，即 `10 * log10(sum(error^2) / sum(gt^2))`；如果部分旧日志是在修改前生成的，它们可能是按样本 NMSE 再平均。由于 COST2100 每个样本能量差异不会无限大，趋势判断仍有参考价值，但严谨论文表格或最终结论应对所有 checkpoint 用当前代码重新 evaluate 一遍。

## 1. 实验覆盖与分类

本次解析到 93 个实验，全部都有最终测试结果：

| 实验族 | 数量 | 含义 |
| --- | ---: | --- |
| `joint_base` | 46 | encoder 与 decoder 从头联合训练，主要包括 seed42 的 14 个 encoder x 3 个 decoder 网格，以及 seed2026/seed3407 的 `transnet_transnet`、`transnet_hybrid` 重复实验。 |
| `frozen_decoder` | 42 | 从 checkpoint 加载 decoder 并冻结，只训练 encoder。主要用于验证固定 decoder 时不同 seed/encoder 能否适配。 |
| `lora` | 5 | 冻结 encoder 与 decoder 后，仅对 hybrid decoder 的 `token_projection` 加 LoRA 训练。 |

所有实验均为 COST2100 `in` 场景，典型维度为 `(2, 32, 32)`，压缩率参数 `cr=4`，码字长度为 `2048 / 4 = 512`。训练 loss 是原始稀疏 CSI 张量上的 MSE。NMSE 计算中输入和预测都会减去 `0.5`，误差平方和除以真实信号能量平方和后取 dB。

## 2. 代码对应的实验语义

### 2.1 UniversalCSI 的参数含义

`encoder` 与 `decoder` 是两个相对独立的模块。`UniversalCSIModel.forward(x)` 执行：

```text
x -> encoder -> code_adapter(optional) -> decoder -> sparse_pred
```

`encoder` 输出统一是 `(B, code_dim)` 的压缩码字；`decoder` 统一接收 `(B, code_dim)` 并输出 `(B, 2, 32, 32)`。因此同一个 decoder 可以理论上接不同 encoder，但这只是在形状上兼容，不代表码字语义分布兼容。

`code_adapter` 默认关闭。开启时是在码字中间插入 `LayerNorm + Linear`。当前这些实验基本没有使用它，因此所有 encoder 到 decoder 的适配压力都直接压在 encoder 末端投影和 decoder 入口层上。

### 2.2 三类 decoder 的本质差异

`transnet` decoder：

- 入口是 `fc_decoder: Linear(code_dim, input_dim)`。
- 将码字投影成 token 序列后，用 `TransformerDecoder`。
- 当前实现将同一个 token 张量同时作为 `tgt` 和 `memory`，因此它并不是典型 seq2seq 跨注意力，而是一个更重的 token mixing 重建模块。

`hybrid` decoder：

- 入口先做 `semantic_projector = LayerNorm(code_dim) + Linear(code_dim, code_dim)`。
- 再由 `token_projection: Linear(code_dim, input_dim)` 扩展到全 CSI token。
- `token_mixer` 是 `TransformerEncoder`，负责全局 token 混合。
- 最后 `CNNRefinementHead` 做局部残差精修：`coarse + residual_scale * refine(coarse)`。

`cnn_residual` decoder：

- 本质偏局部卷积精修，对全局压缩码字到全 CSI 的长程依赖建模能力弱。
- 在本批实验中几乎系统性落后。

### 2.3 frozen decoder 实验实际在测什么

`frozen_decoder` 的机制是：从 `pretrained_decoder` 加载 `decoder.*` 权重并冻结，encoder 随当前 seed 初始化后训练。它测的不是“现成 encoder 与现成 decoder 缝合是否兼容”，而是“在固定 decoder 的约束下，encoder 能否学出 decoder 可理解的码字”。

因此 frozen decoder 实验更接近“平台方固定 decoder，设备侧 encoder 重新训练适配平台 decoder”的场景。它和后续 “现成 encoder + 现成 decoder + LoRA” 是两个难度不同的设定。

### 2.4 LoRA 实验实际在测什么

当前 LoRA 只支持 `decoder=hybrid` 的 `decoder.token_projection`。也就是说 LoRA 只修改从 512 维码字扩展到 2048 维 token 的入口投影，不修改：

- encoder；
- `semantic_projector`；
- `token_mixer`；
- CNN refinement；
- decoder 输出局部精修路径。

这意味着 LoRA 的能力主要是“把输入码字重新线性映射到 decoder 原本熟悉的 token 空间”。如果 encoder 码字语义已经接近 decoder 的训练分布，LoRA 只需小修即可；如果码字语义完全错位，单独改入口 projection 很可能不够。

## 3. 总体结果概览

| group | decoder | n | mean final | best | worst | std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen_decoder | hybrid | 34 | -26.377 | -29.214 | -20.498 | 2.722 |
| frozen_decoder | transnet | 8 | -27.927 | -28.025 | -27.555 | 0.144 |
| joint_base | cnn_residual | 14 | -19.019 | -20.340 | -13.164 | 1.763 |
| joint_base | hybrid | 16 | -26.032 | -29.050 | -22.387 | 1.964 |
| joint_base | transnet | 16 | -27.904 | -31.542 | -21.884 | 2.751 |
| lora | hybrid | 5 | -9.151 | -28.380 | -1.732 | 9.855 |

不能直接只看均值，因为不同实验族里的 encoder 构成不一样。更有意义的现象是：

1. 在 seed42 的联合训练大网格中，`transnet` decoder 对很多强 encoder 的上限最高，最佳达到 `clnet_transnet = -31.542 dB`、`convnext_transnet = -31.483 dB`。
2. `hybrid` decoder 的均值低于 `transnet`，但在 `transnet_hybrid` 和 `convnext_hybrid` 上表现稳定，尤其 frozen decoder 后能稳定回到 `-28.6 dB` 以上。
3. `cnn_residual` decoder 系统性落后，通常只有 `-18` 到 `-20 dB`，说明单纯 CNN residual 解码能力不足以承担 `cr=4` 下从 512 维码字恢复全局 CSI 的任务。
4. LoRA 结果呈现两种完全不同形态：从已经适配好的 frozen model 做 LoRA 能微增；从两个独立 joint 模型直接拼 encoder/decoder 再 LoRA 会从灾难性初值开始，rank 再大也难恢复。

## 4. seed42 联合训练大网格

### 4.1 `transnet` decoder

| encoder | final NMSE | best NMSE | best epoch |
| --- | ---: | ---: | ---: |
| clnet | -31.542 | -31.542 | 400 |
| convnext | -31.483 | -31.483 | 400 |
| mlp_ae | -31.077 | -31.113 | 390 |
| mlp_mixer | -30.740 | -30.740 | 400 |
| swin | -29.995 | -30.377 | 390 |
| csinet | -29.067 | -29.417 | 370 |
| attention_cnn | -28.489 | -28.489 | 400 |
| transnet | -28.126 | -28.126 | 400 |
| cnn | -27.205 | -27.392 | 390 |
| resnet | -25.631 | -25.631 | 400 |
| cbam_cnn | -25.137 | -25.137 | 400 |
| crnet | -25.001 | -25.023 | 390 |
| sparse_resnet | -24.379 | -24.379 | 400 |
| dscnn | -21.884 | -21.884 | 400 |

这里最重要的结论不是“TransNet decoder 一定最好”，而是“当前训练设定下，Transformer decoder 的解码容量与强 encoder 的码字分布形成了最强协同”。特别是 CLNet、ConvNeXt、MLP-AE、MLP-Mixer 这些 encoder，和 `transnet` decoder 联合训练时明显超过 `transnet_hybrid` 基线。

这说明当前瓶颈不只在 encoder，也不只在 decoder，而在联合训练时形成的“码字坐标系”。强 encoder 可能学出更结构化、更适合 Transformer decoder 解析的码字；而 decoder 一旦和 encoder 共同演化，它可以充分利用这种码字结构。

### 4.2 `hybrid` decoder

| encoder | final NMSE | best NMSE | best epoch |
| --- | ---: | ---: | ---: |
| convnext | -29.050 | -29.050 | 400 |
| transnet | -28.407 | -28.407 | 400 |
| mlp_mixer | -28.133 | -28.853 | 330 |
| swin | -27.890 | -28.688 | 390 |
| clnet | -27.880 | -28.412 | 370 |
| attention_cnn | -26.658 | -26.675 | 390 |
| sparse_resnet | -25.659 | -25.659 | 400 |
| mlp_ae | -25.524 | -25.833 | 350 |
| cnn | -25.275 | -25.616 | 390 |
| resnet | -24.755 | -24.880 | 390 |
| crnet | -24.691 | -24.822 | 370 |
| cbam_cnn | -24.178 | -24.685 | 380 |
| dscnn | -22.593 | -22.593 | 400 |
| csinet | -22.387 | -22.447 | 390 |

`hybrid` 的优势不是最高上限，而是它提供了一个更“平台化”的 decoder 入口：`semantic_projector + token_projection + token_mixer + CNN refine`。在 joint 训练时，它不如 `transnet` decoder 的强组合，但在 frozen decoder 场景下更容易让新 encoder 适配。

从表中看，`convnext_hybrid` 最强，`transnet_hybrid` 次之；`mlp_mixer/swin/clnet` 的 best 都在 `-28.4` 到 `-28.9 dB`，但 final 回落，说明有一定后期退化或过拟合。这个回落现象在 hybrid 上比 transnet 更明显，可能来自 CNN refinement 和 residual scale 后期对测试集泛化的轻微损伤，也可能来自 scheduler 后期仍继续优化导致局部解码路径过拟合。

### 4.3 `cnn_residual` decoder

| encoder | final NMSE | best NMSE | best epoch |
| --- | ---: | ---: | ---: |
| swin | -20.340 | -20.599 | 390 |
| convnext | -20.313 | -20.313 | 400 |
| transnet | -20.254 | -20.596 | 390 |
| cnn | -20.019 | -20.057 | 380 |
| mlp_mixer | -19.895 | -21.023 | 350 |
| mlp_ae | -19.766 | -19.864 | 390 |
| sparse_resnet | -19.654 | -19.654 | 400 |
| attention_cnn | -19.493 | -19.591 | 370 |
| resnet | -19.176 | -19.329 | 360 |
| crnet | -19.090 | -19.746 | 360 |
| clnet | -18.514 | -18.839 | 350 |
| cbam_cnn | -18.404 | -19.134 | 390 |
| dscnn | -18.185 | -18.185 | 400 |
| csinet | -13.164 | -13.164 | 400 |

`cnn_residual` 基本可以排除为主线 decoder。它的问题不是某个 encoder 不适配，而是所有 encoder 下都弱。这说明从压缩码字重建 CSI 的第一阶段必须有足够强的全局映射能力；局部卷积精修只能修细节，不能替代从 512 维码字到 2048 维 CSI 的全局语义展开。

## 5. TransNet 多 seed 联合训练：为什么 hybrid 有时不如 transnet

| decoder | seed | final | best | best_epoch |
| --- | ---: | ---: | ---: | ---: |
| cnn_residual | 42 | -20.254 | -20.596 | 390 |
| hybrid | 42 | -28.407 | -28.407 | 400 |
| hybrid | 2026 | -25.870 | -28.207 | 380 |
| hybrid | 3407 | -27.562 | -27.562 | 400 |
| transnet | 42 | -28.126 | -28.126 | 400 |
| transnet | 2026 | -28.180 | -28.180 | 400 |
| transnet | 3407 | -28.520 | -28.520 | 400 |

`transnet_transnet` 在 42/2026/3407 三个 seed 下非常稳定：`-28.126`、`-28.180`、`-28.520 dB`。而 `transnet_hybrid` 波动明显：seed42 是 `-28.407`，seed3407 是 `-27.562`，seed2026 final 掉到 `-25.870`，但 best 曾到 `-28.207`。

这个现象的根因很可能是 hybrid decoder 的优化路径更复杂：

1. `hybrid` 比 `transnet` 多了 semantic projector、CNN refinement、residual scale。它的解码链路更长，局部路径可能在训练后期改变 coarse 输出分布。
2. `transnet_hybrid` 的 seed2026 best 在 epoch 380 达到 `-28.207`，最终 epoch 400 掉到 `-25.870`。这不是容量不足，而是训练后期发生了明显退化。报告中应优先看 best checkpoint，而不是 final。
3. `transnet_transnet` 的 decoder 更“单一路径”：fc 展开 + TransformerDecoder，训练动力学更稳，因此 seed 方差小。

所以“hybrid 不如 transnet”不能简单解释为 hybrid 架构更差。更准确的说法是：在 joint 训练里，hybrid 的潜在能力存在，但训练稳定性弱于 transnet；在固定 decoder 场景里，hybrid 反而体现出更好的可适配性。

## 6. Frozen decoder：固定 decoder 后性能能不能涨回来

### 6.1 transnet encoder 多 seed 适配固定 decoder

| decoder | seed | epochs | final | best | best_epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | 0 | 400 | -28.620 | -28.620 | 400 |
| hybrid | 1 | 400 | -28.620 | -28.620 | 400 |
| hybrid | 42 | 200 | -28.309 | -28.309 | 200 |
| hybrid | 42 | 400 | -28.656 | -28.656 | 400 |
| hybrid | 666 | 400 | -28.628 | -28.628 | 400 |
| hybrid | 999 | 400 | -28.618 | -28.619 | 390 |
| hybrid | 2026 | 400 | -28.615 | -28.616 | 390 |
| hybrid | 3407 | 400 | -28.613 | -28.613 | 400 |
| transnet | 0 | 400 | -27.907 | -27.907 | 400 |
| transnet | 1 | 400 | -27.988 | -27.988 | 400 |
| transnet | 42 | 200 | -27.555 | -27.555 | 200 |
| transnet | 42 | 400 | -28.007 | -28.007 | 400 |
| transnet | 666 | 400 | -27.976 | -27.976 | 400 |
| transnet | 999 | 400 | -28.025 | -28.025 | 400 |
| transnet | 2026 | 400 | -27.985 | -27.985 | 400 |
| transnet | 3407 | 400 | -27.972 | -27.972 | 400 |

这组实验结论非常强：

1. 固定 seed42 的 hybrid decoder 后，不同 seed 初始化的 transnet encoder 都能稳定收敛到 `-28.61` 到 `-28.66 dB`。
2. 固定 seed42 的 transnet decoder 后，不同 seed encoder 稳定在 `-27.91` 到 `-28.03 dB`。
3. 400 epoch 明显优于 200 epoch：hybrid 从 `-28.309` 到 `-28.656`，transnet 从 `-27.555` 到 `-28.007`。训练确实更快，但 200 epoch 还没完全追上 400 epoch。

这说明固定 decoder 的方案是可行的，而且 frozen hybrid decoder 比 frozen transnet decoder 高约 `0.6 dB`。这和 joint 训练里 transnet decoder 常常更强并不矛盾：joint 训练测的是共同学习上限；frozen decoder 测的是 decoder 作为固定协议时的可适配性。hybrid 的 `semantic_projector + token_projection` 入口更像一个可学习码字解释器，一旦 decoder 固定，encoder 能围绕它学出兼容码字。

### 6.2 固定 hybrid decoder + 不同 encoder

seed3407：

| encoder | final | best | best_epoch |
| --- | ---: | ---: | ---: |
| swin | -29.214 | -29.291 | 360 |
| convnext | -29.179 | -29.179 | 390 |
| mlp_mixer | -29.020 | -29.020 | 400 |
| clnet | -28.814 | -28.960 | 360 |
| transnet | -28.613 | -28.613 | 400 |
| attention_cnn | -26.857 | -26.857 | 400 |
| mlp_ae | -26.298 | -26.312 | 380 |
| sparse_resnet | -25.953 | -26.019 | 390 |
| cnn | -25.442 | -25.726 | 390 |
| cbam_cnn | -25.288 | -25.312 | 340 |
| resnet | -25.272 | -25.272 | 400 |
| dscnn | -22.771 | -22.808 | 390 |
| csinet | -20.897 | -20.897 | 400 |
| crnet | -20.498 | -20.498 | 400 |

seed2026：

| encoder | final | best | best_epoch | epochs |
| --- | ---: | ---: | ---: | ---: |
| convnext | -29.037 | -29.037 | 200 |
| swin | -28.946 | -28.946 | 200 |
| mlp_mixer | -28.884 | -28.884 | 200 |
| clnet | -28.669 | -28.785 | 170 |
| transnet | -28.615 | -28.616 | 390 |
| attention_cnn | -26.385 | -26.385 | 200 |
| mlp_ae | -26.192 | -26.192 | 200 |
| cnn | -25.270 | -25.270 | 200 |
| sparse_resnet | -25.226 | -25.338 | 190 |
| cbam_cnn | -24.568 | -24.568 | 200 |
| resnet | -24.465 | -24.592 | 180 |
| csinet | -22.574 | -22.574 | 200 |
| dscnn | -21.258 | -21.258 | 200 |
| crnet | -21.158 | -21.158 | 200 |

这组实验说明，固定 hybrid decoder 并不要求 encoder 一定是 transnet。强 encoder 适配固定 decoder 后可以超过 transnet encoder：

- seed3407 下 `swin/convnext/mlp_mixer/clnet` 都超过 `-28.8 dB`，最强 `swin` best 为 `-29.291 dB`。
- seed2026 下即使很多实验只有 200 epoch，`convnext/swin/mlp_mixer/clnet` 也已经到 `-28.7` 到 `-29.0 dB`。

这对最终目标很关键：如果平台固定 decoder，要让不同厂商 encoder 接入，不一定只能要求所有厂商用 transnet encoder。更合理的协议是固定 decoder 和码字维度，让各厂商训练自己的 encoder 适配固定 decoder；强 encoder 架构可以带来额外收益。

但这里也有一个风险：不同 encoder 的表现差距很大。`csinet/crnet/dscnn` 在 frozen hybrid 下远低于强 encoder。也就是说固定 decoder 不是万能协议；encoder 必须有足够表达能力，并且训练目标必须直接对齐固定 decoder。

## 7. LoRA 实验：为什么一组正常、一组灾难

| seed | rank/alpha | before NMSE | final | best | best_epoch | pretrained_encoder | pretrained_decoder | pretrained |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2026 | 8/16 | 28.634 | -1.732 | -1.733 | 399 | `seed42/transnet_hybrid/base` | `seed2026/transnet_hybrid` |  |
| 2026 | 16/32 | 28.634 | -2.720 | -2.720 | 400 | `seed42/transnet_hybrid/base` | `seed2026/transnet_hybrid` |  |
| 2026 | 32/64 | 28.634 | -4.879 | -4.883 | 398 | `seed42/transnet_hybrid/base` | `seed2026/transnet_hybrid` |  |
| 2026 | 64/128 | 28.634 | -8.042 | -8.043 | 399 | `seed42/transnet_hybrid/base` | `seed2026/transnet_hybrid` |  |
| 3407 | 8/16 | -28.240 | -28.380 | -28.380 | 387 |  |  | `frozen_decoder/seed3407/transnet_hybrid` |

这里的差异非常有解释力。

seed3407 LoRA 是从一个已经适配固定 decoder 的完整 frozen model 加载：encoder 与 decoder 的码字协议已经一致，Before LoRA 就是 `-28.240 dB`。LoRA rank8/alpha16 只是在 decoder 入口投影上做小幅校正，最终到 `-28.380 dB`。提升不大但合理，因为 baseline 已经很强，LoRA 只是微调。

seed2026 LoRA 则是直接把 seed42 的 joint encoder 和 seed2026 的 joint decoder 拼在一起。Before LoRA 是 `+28.634 dB`，loss 是 `3.3000e-01`，这是灾难性错位，不是轻微域偏移。即使 rank 从 8 提到 64，final 也只从 `-1.732` 改到 `-8.042 dB`，仍远离可用水平。

根因是码字没有统一语义。两个 joint model 即便架构相同、数据相同、seed 不同，它们中间 512 维码字也可以经过任意可逆或近似可逆的坐标变换，只要各自 decoder 能解码即可。自编码器训练只约束输入输出，不约束 latent 坐标系。因此“拿 A 的 encoder 输出给 B 的 decoder”通常没有理由能工作。

这也解释了为什么 frozen decoder 实验能成功：训练 encoder 时 decoder 已经固定，encoder 被迫学习该 decoder 能理解的 latent 坐标系。LoRA 实验中如果 encoder 没有经过固定 decoder 适配，仅靠 decoder 入口的低秩更新去修复整个 latent 坐标系，难度极大。

## 8. 对“不同 seed 表示不同厂商”的理解

不同 seed 可以作为“厂商差异”的弱模拟，但要区分三种真实场景：

1. 同一平台协议场景：平台发布固定 decoder，各厂商用自己的 seed/架构训练 encoder 适配它。对应 frozen decoder 实验。这个场景最现实，也最容易成功。
2. 黑盒厂商 encoder 场景：厂商已经训练好 encoder，平台拿不到训练过程，只能通过 LoRA 或 adapter 让 decoder 适配。对应现成 encoder + 现成 decoder + LoRA。这个场景更难，因为 latent 坐标系未对齐。
3. 生成模型数据对场景：你最终想得到 `<压缩码字, decoder 微调权重>`。如果这些码字来自已经适配同一固定 decoder 的 encoder，那么 decoder 微调权重会较小，数据分布更规整；如果码字来自任意 joint encoder，LoRA 权重会承担对齐 latent 坐标系的巨大负担，学习问题会变成“从任意 latent 坐标系恢复 decoder 协议”，泛化难度显著提高。

因此，如果最终目标是一个可落地系统，主线应是：固定一个 decoder，要求所有 encoder 在这个固定 decoder 下训练。然后再收集不同 encoder/seed/数据条件下的 `<codeword, LoRA>`。这样生成模型学习的是小幅适配规律，而不是从灾难性错位里重建整个协议。

如果研究目标是“面对未按协议训练的未知 encoder，生成模型是否能救回来”，那应该另开一条更难的 benchmark，不应和主线混在一起评价。

## 9. 当前最可信结论

1. `cnn_residual` decoder 不适合作为主线固定 decoder。它在 joint 训练下系统性落后，说明全局解码能力不足。
2. `transnet` decoder 在 joint 训练中上限最高，尤其配合 CLNet、ConvNeXt、MLP-AE、MLP-Mixer 等 encoder，能到 `-30` 到 `-31.5 dB`。
3. `hybrid` decoder 在 joint 训练中不一定最高，但作为固定 decoder 更适合平台化：transnet encoder 多 seed 适配后稳定约 `-28.62 dB`，强 encoder 可到 `-29.0` 到 `-29.3 dB`。
4. frozen decoder 实验已经证明“固定 decoder + 不同 seed 训练 encoder”能把性能涨回来，并且可以达到甚至略超过原始 `transnet_hybrid` joint baseline。
5. LoRA 对已经对齐的 frozen model 是有效微调，但提升很小；对未对齐的 joint encoder/decoder 拼接，单独 `token_projection` LoRA 远远不够。
6. 目前 seed2026 的 LoRA rank 扫描更像是在证明“灾难性 latent 错位无法靠小 LoRA 修复”，不应作为 LoRA 方法无效的证据。

## 10. 建议补充实验

### 10.1 统一用当前 total NMSE 重新评测

必须补：

- 对所有已有 `best_nmse.pth` 重新跑 evaluate，输出一个统一 metric 版本的 CSV。
- 报告中同时保存 `old_log_nmse` 和 `re_eval_total_nmse`。

原因：当前实验横跨指标修改前后，直接用 run.log 做最终论文结论存在口径风险。

### 10.2 固定 decoder 的完整 seed x encoder 矩阵

建议以 seed42 hybrid decoder 为固定 decoder，补齐：

- encoder seed：0、1、42、666、999、2026、3407；
- encoder 架构：至少 transnet、convnext、swin、mlp_mixer、clnet；
- epoch：200 和 400 都保留。

目的：

- 判断强 encoder 的收益是否稳定，不只是 seed2026/3407 两个点。
- 判断 200 epoch 是否足够用于筛选，400 epoch 是否必要用于最终报告。

### 10.3 固定 decoder 来源 seed 的影响

现在 frozen decoder 基本固定的是 seed42 decoder。需要补：

- fixed decoder = seed42/2026/3407 的 `transnet_hybrid` joint 或 frozen checkpoint；
- encoder seed = 0/1/42/666/999/2026/3407；
- encoder 架构先用 transnet，再扩展 convnext/swin。

目的：

- 判断 seed42 decoder 是否偶然更好；
- 判断平台 decoder 选择应看 joint best、frozen adaptability，还是 LoRA adaptability。

### 10.4 LoRA 需要分成两条线

主线 LoRA：

- 加载完整 frozen model：`--pretrained exps/.../frozen_decoder/.../best_nmse.pth`；
- 冻结 encoder/decoder；
- LoRA rank：4、8、16、32；
- target：先 `token_projection`，再考虑 `semantic_projector.linear`、`token_mixer` 的 attention/FFN Linear。

困难线 LoRA：

- `--pretrained_encoder` 与 `--pretrained_decoder` 分别来自两个独立 joint model；
- 记录 Before LoRA；
- 如果 Before LoRA 高于 0 dB，不应期待 token_projection LoRA 恢复到 -28 dB；
- 需要尝试更强 adapter：full-rank code adapter、decoder input MLP、semantic_projector + token_projection 联合 LoRA。

### 10.5 加入 latent 对齐诊断

建议对不同实验保存 codeword 后计算：

- 每维均值/方差；
- codeword L2 norm 分布；
- seed42 encoder code 与 seed2026 encoder code 的 CKA/CCA 相似度；
- 训练一个线性 map `W` 从 encoder A code 到 encoder B code，再看 frozen decoder B 的 NMSE。

如果线性 map 能把 `+28 dB` 拉回接近 `-28 dB`，说明错位主要是线性坐标变换；如果不能，说明 encoder latent 语义差异是非线性的，LoRA target 需要更深。

## 11. 推荐主线

当前证据支持以下路线：

1. 固定 `hybrid` decoder 作为平台 decoder，而不是 `cnn_residual`。
2. 先用 `transnet` encoder 做稳定性基线，因为它在 frozen hybrid 下跨 seed 极稳。
3. 再用 `convnext/swin/mlp_mixer/clnet` 作为强 encoder 候选，因为它们在 fixed hybrid decoder 下能超过 transnet encoder。
4. 生成 `<压缩码字, decoder LoRA 权重>` 数据时，优先从“固定 decoder 训练出的 encoder”生成，这样 LoRA 是小幅校正，生成模型任务更清晰。
5. 把“未对齐 joint encoder/decoder 直接拼接”作为单独的 stress test，不要和主线混为一谈。

如果只选一个最稳的当前配置，建议从：

```text
fixed decoder: seed42 transnet_hybrid/base/checkpoints/best_nmse.pth
encoder: transnet / convnext / swin / mlp_mixer / clnet
training: frozen_decoder, 400 epochs
LoRA: 从 frozen_decoder 完整 checkpoint 出发，token_projection rank 8/16 起步
```

这个配置最符合平台固定 decoder 的真实目标，也最能避免 latent 坐标系错位导致的伪失败。
