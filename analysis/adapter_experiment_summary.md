# Adapter / Mapper / Flow Matching 实验总结

> 范围：本报告按“广义适配”汇总已有完成实验，但按要求**去掉 `encoder_canonical` 和 LoRA 相关实验**。保留内容包括主项目普通 code-space adapter、teacher-code adapter、`mapper/`、`flow_matching/`，以及相关的 `decoder_generalization_fm/`。未完成、没有 `metrics.json` 且日志没有最终指标的实验不作为结论依据。  
> 主要任务：把源 encoder/code space 适配到 seed42 的目标 decoder。核心指标优先看 decoder NMSE；只有 code NMSE 的实验会单独标注。

## 1. 总体结论

去掉 `encoder_canonical` 和 LoRA 后，当前有效路线主要剩三类：

| 档位 | 方法 | 代表结果 | 判断 |
| --- | --- | ---: | --- |
| 最强重建 | 大模型 `flow_matching/code_only` | seed3407 transnet -> seed42 decoder `-29.09 dB`；seed2026 clnet/crnet -> seed42 decoder 约 `-29.01/-29.00 dB` | 目前最接近目标 teacher decoder 的路线，但参数和训练成本最大 |
| 最强 code 拟合 | `mapper` Hybrid + alignaffine + residual condition | seed2026 transnet -> seed42 code NMSE `-30.33 dB`；seed2026 clnet/crnet -> seed42 code NMSE `-28.50/-28.44 dB` | code-space 对齐很强，但很多实验没有同步 decoder eval |
| 主项目基础线 | 普通 MLP/MLPDirect/Transformer adapter | seed2026 transnet -> seed42 hybrid 约 `-24.67/-24.91 dB`；seed3407 长训最好 `-23.32 dB` | 能证明插入式 adapter 可行，但跨 seed/跨 encoder 稳定性不足 |

核心判断：**普通 adapter 的主要问题不是参数不够，而是缺少显式 code-space 对齐结构**。效果好的 mapper/flow matching 都显式建模了源 code 到目标 code 的坐标变换；普通 MLP adapter 虽然有约 `5.24M` 参数，但跨 seed/encoder 时容易停在 `-20~-25 dB`。

## 2. 方法族和参数量

| 方法族 | 位置 | 典型参数量 | 特点 |
| --- | --- | ---: | --- |
| MLPAdapter / MLPDirectAdapter | `models/adapters/` | `~5.24M` | 512 维 code 上两层 MLP；同分布有效，跨域不稳定 |
| TransformerAdapter | `models/adapters/` | `~1.87M` | 序列化 code 后做 transformer，已有结果不如 MLP |
| DiagonalAffineAdapter | `models/adapters/` | `~1K` | 只做逐维 scale+bias，容量太低 |
| LowRankAffineAdapter rank32 | `models/adapters/` | `~33K` | 低秩线性修正，单独使用不足 |
| GatedLowRankAffineLinear rank32 | `models/adapters/` | `~295K` | 低秩 + gate linear，已有非 canonical 结果有限 |
| GatedLowRankAffineMLP rank32 hidden2048 | `models/adapters/` | `~5.28M` | teacher-code adapter 使用，但效果一般 |
| MLP mapper | `mapper/` | `~8.4M` | code 到 code 的 ResMLP 映射 |
| Hybrid mapper | `mapper/` | `~23.1M` | flow + MLP，code 拟合明显强于纯 MLP |
| Flow mapper | `mapper/` | `~14.7M` | 可逆 flow，现有结果不如 Hybrid |
| Flow Matching 大模型 | `flow_matching/` | 约 `142.9M` | ODE/velocity 模型，decoder NMSE 最强但成本最高 |
| Flow Matching 小模型 h128 | `flow_matching/` | 明显小于大模型 | 省参数但效果大幅下降 |

## 3. 主项目普通 Code Adapter

普通 adapter 使用 `pretrained_encoder` + frozen `pretrained_decoder`，在 encoder code 和 decoder 之间插入 adapter。这里统计的是主项目 `exps/COST2100/in/adapter/` 下已经完成的实验。

### 3.1 TransNet + Hybrid decoder 的 seed 迁移

| 源 seed -> 目标 decoder | Adapter | test NMSE | 备注 |
| --- | --- | ---: | --- |
| seed42 -> seed42 | MLPAdapter | `-28.02 dB` | 同分布，接近原生 transnet+hybrid |
| seed42 -> seed42 | MLPDirectAdapter | `-27.73 dB` | 比残差 MLP 稍弱 |
| seed2026 -> seed42 | MLPAdapter | `-24.67 dB` | 跨 seed 掉约 3 dB+ |
| seed2026 -> seed42 | MLPDirectAdapter | `-24.91 dB` | 略好于 MLPAdapter，但仍远低于同 seed |
| seed1024 -> seed42 | MLPAdapter | `-23.63 dB` | 中等 |
| seed1024 -> seed42 | MLPDirectAdapter | `-23.72 dB` | 接近 MLPAdapter |
| seed520 -> seed42 | MLPAdapter | `-23.60 dB` | 中等 |
| seed796 -> seed42 | MLPAdapter | `-23.51 dB` | 中等 |
| seed3407 -> seed42 | MLPAdapter | `-21.17 dB` | 普通配置很差 |
| seed3407 -> seed42 | MLPAdapter, recon=1 code=1e-3, 1000ep | `-23.32 dB` | 加 code loss 和更长训练改善约 2.1 dB |
| seed3407 -> seed42 | MLPDirectAdapter | `-21.46 dB` | 与 MLPAdapter 接近 |
| seed3407 -> seed42 | TransformerAdapter | `-20.91 dB` | 没有收益 |

结论：普通 MLP adapter 对同分布很有效，但跨 seed 后明显掉点。MLPDirect 和残差 MLP 差距不大，说明主要瓶颈不是残差形式，而是源 code 与目标 decoder 的分布错位。

### 3.2 跨 encoder 到 hybrid decoder

| 源 encoder | Adapter | loss 权重 | test NMSE |
| --- | --- | --- | ---: |
| clnet_hybrid seed42 | MLPAdapter | recon=1, code=0 | `-22.16 dB` |
| clnet_hybrid seed42 | MLPAdapter | recon=1, code=1e-3 | `-21.65 dB` |
| cbam_cnn_hybrid seed42 | MLPAdapter | recon=1, code=1e-3 | `-22.07 dB` |
| crnet_hybrid seed42 | MLPAdapter | recon=1, code=1e-3 | `-21.04 dB` |

结论：跨 encoder 时普通 MLP adapter 基本停在 `-21~-22 dB`。这说明不同 encoder 的 code space 不只是一个容易由 MLP 直接吸收的小扰动，decoder 对 code 的绝对坐标、尺度和协方差都很敏感。

## 4. Teacher Code Adapter

Teacher-code adapter 使用 teacher code 监督 gated low-rank affine MLP，实验目录在 `exps/COST2100/in/teacher_code_adapter/`。

| 源 -> 目标 | 配置 | test NMSE |
| --- | --- | ---: |
| seed2026 -> seed42 | code=1e-3, fc=0 | `-21.38 dB` |
| seed2026 -> seed42 | code=1e-3, fc=1e-2 | `-20.81 dB` |
| seed2026 -> seed42 | code=1e-2, fc=1e-2 | `-20.78 dB` |
| seed2026 -> seed42 | recT=1.0 | `-21.01 dB` |
| seed2026 -> seed42 | tPCA/tWhiten variants | `-20.82~-20.84 dB` |
| seed3407 -> seed42 | code=1e-3, fc=1e-2 | `-20.28 dB` |

结论：teacher-code adapter 结果弱于普通 MLP adapter 的部分 seed 迁移，也明显弱于 mapper/flow matching。它虽然直接监督目标 code，但没有把源 code 推到目标 decoder 的高质量区域；`fc`、`recT`、PCA/whiten 变体都没有带来决定性改善。

## 5. Mapper 实验

`mapper/` 是显式 code-space translator，不再是主项目里的插入式 adapter，但从目标上属于同一类适配问题。这里保留 `mapper/exps`、`mapper/exps_decoder_aware`、`mapper/exps_combined_losses` 的完成结果。

### 5.1 基础 mapper：code-level 结果

很多 mapper 实验的 `metrics.json` 主要记录 code NMSE，而没有同步写 decoder NMSE。下面先按 code 拟合质量比较。

| 源 -> seed42 transnet code | Mapper / 配置 | code NMSE |
| --- | --- | ---: |
| seed2026 transnet -> seed42 | Hybrid + alignaffine + residual condition start | `-30.33 dB` |
| seed3407 transnet -> seed42 | Hybrid + alignaffine + residual condition start | `-29.76 dB` |
| seed2026 clnet -> seed42 | Hybrid + alignaffine + residual condition start | `-28.50 dB` |
| seed2026 crnet -> seed42 | Hybrid + alignaffine + residual condition start | `-28.44 dB` |
| seed3407 clnet -> seed42 | Hybrid + alignaffine + residual condition start | `-28.33 dB` |
| seed3407 csinet -> seed42 | Hybrid + alignaffine + residual condition start | `-27.54 dB` |
| seed3407 crnet -> seed42 | Hybrid + alignaffine + residual condition start | `-27.02 dB` |
| seed2026 csinet -> seed42 | Hybrid + alignaffine + residual condition start | `-26.19 dB` |

这些结果说明：显式 affine 对齐加 residual condition 后，Hybrid mapper 对 code space 的拟合非常强，明显超过普通 MLP adapter。尤其 transnet/clnet/crnet 的 code NMSE 已进入 `-28~-30 dB` 区间。

### 5.2 Hybrid 与 MLP mapper 对比

| 源 -> 目标 | Hybrid 代表 code NMSE | MLP 代表 code NMSE | 结论 |
| --- | ---: | ---: | --- |
| seed2026 transnet -> seed42 | `-30.33 dB` | `-25.33 dB` | Hybrid 明显更强 |
| seed3407 transnet -> seed42 | `-29.76 dB` | `-24.80 dB` | Hybrid 明显更强 |
| seed2026 clnet -> seed42 | `-28.50 dB` | `-23.38 dB` | Hybrid 明显更强 |
| seed2026 crnet -> seed42 | `-28.44 dB` | `-23.41 dB` | Hybrid 明显更强 |
| seed2026 csinet -> seed42 | `-26.19 dB` | `-20.97 dB` | Hybrid 明显更强 |

结论：纯 MLP mapper 的容量虽然有 `~8.4M`，但缺少 flow/显式结构时，code-space 对齐大约差 5 dB。Hybrid 的额外参数和结构确实有效。

### 5.3 Flow mapper

| 源 -> 目标 | code NMSE | decoder NMSE |
| --- | ---: | ---: |
| seed2026 transnet -> seed42 | `-18.92 dB` | `-22.34 dB` |
| seed2026 clnet -> seed42 | `-18.79 dB` | `-21.66 dB` |
| seed2026 crnet -> seed42 | `-18.76 dB` | `-21.59 dB` |
| seed2026 csinet -> seed42 | `-17.84 dB` | `-20.98 dB` |

结论：当前 flow mapper 不如 Hybrid。虽然 flow 可逆、形式优雅，但在这些 code 分布上没有学到足够好的目标 code 对齐。

### 5.4 Decoder-aware / combined-loss mapper

`mapper/exps_decoder_aware` 和 `mapper/exps_combined_losses` 加入 `recT/rec/fc/tail/white` 等损失，当前日志多记录 code NMSE。

| 方向 | 最好/代表配置 | code NMSE |
| --- | --- | ---: |
| seed2026 transnet -> seed42 | hybrid recT | `-25.38 dB` |
| seed3407 transnet -> seed42 | hybrid recT_rec_fc_tail | `-25.24 dB` |
| seed2026 clnet -> seed42 | hybrid recT_fc | `-23.41 dB` |
| seed2026 crnet -> seed42 | hybrid smooth_tail_white_recT_rec_fc_decTail | `-23.23 dB` |
| seed2026 csinet -> seed42 | hybrid recT_rec_fc_tail | `-23.09 dB` |
| MLP decoder-aware transnet | best | `-22.66 dB` |

结论：decoder-aware 损失对重建目标更直接，但在当前记录的 code NMSE 上反而不如基础 alignaffine + residual condition 的 Hybrid mapper。它可能牺牲了 code MSE 来换 decoder 友好性，因此后续比较必须统一跑 decoder eval，不能只看 code NMSE。

## 6. Flow Matching

### 6.1 大模型 code_only

| 源 -> seed42 transnet | code NMSE | best decoder NMSE | 备注 |
| --- | ---: | ---: | --- |
| seed3407 transnet -> seed42 | `-46.82 dB` | `-29.09 dB` | 已有 decoder eval |
| seed2026 clnet -> seed42 | `-38.19 dB` | `-29.01 dB` | 跨 encoder 很强 |
| seed2026 crnet -> seed42 | `-37.67 dB` | `-29.00 dB` | 跨 encoder 很强 |
| seed2026 csinet -> seed42 | `-19.40 dB` | `-22.66 dB` | csinet 仍困难 |
| seed2026 transnet -> seed42 warmcos | `-48.15 dB` | 未在该日志写出 | code 拟合最强，但需补 decoder eval |

结论：大模型 flow matching 是目前最接近“通用 code translator”的方法。它不仅 code NMSE 极低，而且在 clnet/crnet 跨 encoder 上 decoder NMSE 接近 `-29 dB`。代价是模型参数和训练成本远高于其他 adapter。

### 6.2 小模型 h128_b4_t64

| 源 -> seed42 transnet | endpoint | code NMSE | best decoder NMSE |
| --- | --- | ---: | ---: |
| seed2026 transnet -> seed42 | end=1.0 | `-14.63 dB` | `-18.20 dB` |
| seed2026 transnet -> seed42 | end=0.0 | `-14.23 dB` | `-17.77 dB` |
| seed3407 transnet -> seed42 | end=1.0 | `-13.45 dB` | `-16.87 dB` |
| seed2026 clnet -> seed42 | end=1.0 | `-12.81 dB` | `-14.45 dB` |
| seed2026 crnet -> seed42 | end=1.0 | `-12.86 dB` | `-14.43 dB` |
| seed2026 csinet -> seed42 | end=1.0 | `-10.14 dB` | `-11.41 dB` |

结论：小模型 flow matching 暂时不值得作为主线。它比大模型便宜很多，但 decoder NMSE 只有 `-11~-18 dB`，甚至弱于普通 MLP adapter。

## 7. Decoder Generalization FM

`decoder_generalization_fm` 更像“根据条件生成/泛化 decoder”的方向，不是直接 code adapter。当前完成日志里 `set_transformer_film_h1024_b4_lr5e-4_ep400` 的 best test NMSE 为正值 `29.33 dB`，说明该路线当前没有形成有效 decoder 适配，暂不作为主线。

## 8. 关键分析

### 8.1 为什么普通 MLP adapter 不够

普通 MLP adapter 的容量并不小，约 `5.24M`，但跨 seed/encoder 通常只有 `-20~-25 dB`。问题不在容量，而在优化目标和结构先验：目标 decoder 对 code 的绝对位置、尺度、协方差和尾部维度很敏感；没有显式对齐时，MLP 容易学到训练集局部映射，却不能稳定进入目标 decoder 的高质量流形。

### 8.2 affine 对齐是关键结构

表现好的非 canonical、非 LoRA 方法也几乎都用了 affine 或显式坐标对齐：

- Hybrid mapper + alignaffine 的 code NMSE 可到 `-30.33 dB`。
- flow_matching 在 alignaffine 初始条件下可以把部分跨 encoder decoder NMSE 推到 `-29 dB`。
- 普通 MLP adapter 没有显式对齐，跨 seed/encoder 大多停在 `-20~-25 dB`。

这说明 code-space 差异首先是全局坐标系差异，其次才是非线性残差。后续如果继续做轻量 adapter，也应先解决 affine/whiten/mean-cov 对齐，再训练小残差。

### 8.3 code NMSE 和 decoder NMSE 不能混用

部分 mapper 的 code NMSE 非常好，但没有同步 decoder eval；部分 decoder-aware 训练 code NMSE 不如纯 MSE，却可能更符合 decoder。后续所有比较应固定三列：`code NMSE`、`train decoder NMSE`、`val/test decoder NMSE`。只看其中一个会误判方法。

### 8.4 跨 encoder 难度排序

从已有非 canonical、非 LoRA 实验看，跨 encoder 难度大致为：

```text
transnet seed 迁移 < crnet/clnet -> transnet < csinet -> transnet
```

csinet 在多条路线里都更难：flow_matching 大模型 seed2026 csinet 只有 `-22.66 dB`，Hybrid mapper 虽能把 code NMSE 做到 `-26.19 dB`，但仍明显弱于 transnet/clnet/crnet。

## 9. 推荐后续方向

1. 主线保留两条：`Hybrid mapper + alignaffine residual condition`、`flow_matching 大模型`。
2. 普通 MLPAdapter、MLPDirectAdapter、TransformerAdapter 暂时只作为 baseline，不建议作为主线。
3. Teacher-code adapter 当前收益不足，除非加入更明确的 affine/mean-cov 初始化，否则不建议继续大规模扫参。
4. 所有 mapper 实验统一导出 mapped code 后跑同一个 decoder eval 脚本，避免 code NMSE 和 decoder NMSE 分裂。
5. 对 csinet 单独做分析，不要用 transnet/crnet 的经验直接套；它可能需要更强的目标 code distribution regularization。
6. 若目标是做轻量非 LoRA adapter，优先尝试 `closed-form affine/whiten + 小 residual MLP`，而不是直接上 5M MLP adapter。

