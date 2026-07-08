# 全部实验结果汇总与分析报告（含参数量分析）

> **实验背景**：CSI 反馈自编码重建任务，在 COST2100 室内场景（in）和 WAIRD 场景上使用不同的 encoder-decoder 组合。压缩率 CR=4（input_dim=2048, code_dim=512, d_model=64, FFN=256）。  
主要探索三大类方法：(1) **原生 encoder-decoder** 联合训练，(2) **码字间映射器（Mapper）** 将源编码器的码字映射到目标解码器的码字空间，(3) **LoRA 微调** 等 decoder 适配方法。

***

## 1. 基线：原生 Encoder-Decoder 联合训练

### 1.1 模型参数量

各 encoder 和 decoder 的参数量（code_dim=512, d_model=64）：

| Encoder       | 参数量     | Decoder      | 参数量                                  |
| ------------- | ---------- | ------------ | --------------------------------------- |
| transnet      | **1,148K** | transnet     | **1,183K** (含 fc_decoder 1,051K)       |
| csinet        | **1,049K** | hybrid       | **1,423K** (含 token_projection 1,051K) |
| crnet         | **1,049K** | cnn_residual | **1,062K** (含 fc_decoder 1,051K)       |
| clnet         | **1,050K** |              |                                         |
| cnn           | **2,158K** |              |                                         |
| cbam_cnn      | **2,159K** |              |                                         |
| resnet        | **2,195K** |              |                                         |
| convnext      | **2,158K** |              |                                         |
| mlp_mixer     | **2,200K** |              |                                         |
| attention_cnn | **2,122K** |              |                                         |
| swin          | **2,169K** |              |                                         |
| dscnn         | **2,107K** |              |                                         |
| sparse_resnet | **2,195K** |              |                                         |
| mlp_ae        | **5,250K** |              |                                         |

> **注意**：`fc_decoder`（Linear 512→2048）占 decoder 参数的绝大部分（1,050,624/1,183K ≈ 89%），是方法间共用的最大单模块。

**原生完整模型参数量** = encoder + decoder：

- 最小模型（csinet+transnet）：**2,232K**
- 最大模型（mlp_ae+hybrid）：**6,673K**
- 主流配置（transnet+transnet）：**2,331K**
- 最佳性能（clnet+transnet）：**2,233K**

### 1.2 COST2100 室内场景 (seed=42)

固定 seed=42，全部 encoder 与三种 decoder（transnet / hybrid / cnn_residual）组合的最终测试 loss 和 NMSE：

| Encoder                    | + transnet                 | + hybrid                | + cnn_residual      |
| -------------------------- | -------------------------- | ----------------------- | ------------------- |
| **transnet** (1,148K)      | 7.95e-7 / **-28.13 dB**    | 7.18e-7 / **-28.41 dB** | 4.59e-6 / -20.25 dB |
| **clnet** (1,050K)         | 3.47e-7 / **-31.54 dB** ⭐ | 7.59e-7 / -27.88 dB     | 6.78e-6 / -18.51 dB |
| **csinet** (1,049K)        | 6.13e-7 / -29.07 dB        | 2.90e-6 / -22.39 dB     | 2.34e-5 / -13.16 dB |
| **crnet** (1,049K)         | 1.57e-6 / -25.00 dB        | 1.66e-6 / -24.69 dB     | 5.81e-6 / -19.09 dB |
| **cbam_cnn** (2,159K)      | 1.58e-6 / -25.14 dB        | 1.80e-6 / -24.18 dB     | 6.85e-6 / -18.40 dB |
| **cnn** (2,158K)           | 9.50e-7 / -27.21 dB        | 1.40e-6 / -25.28 dB     | 4.85e-6 / -20.02 dB |
| **resnet** (2,195K)        | 1.39e-6 / -25.63 dB        | 1.60e-6 / -24.76 dB     | 5.66e-6 / -19.18 dB |
| **dscnn** (2,107K)         | 3.30e-6 / -21.88 dB        | 2.69e-6 / -22.59 dB     | 7.35e-6 / -18.19 dB |
| **convnext** (2,158K)      | 3.56e-7 / **-31.48 dB** ⭐ | 6.07e-7 / -29.05 dB     | 4.69e-6 / -20.31 dB |
| **mlp_mixer** (2,200K)     | 4.20e-7 / **-30.74 dB**    | 7.15e-7 / -28.13 dB     | 4.76e-6 / -19.90 dB |
| **attention_cnn** (2,122K) | 6.95e-7 / -28.49 dB        | 1.04e-6 / -26.66 dB     | 5.50e-6 / -19.49 dB |
| **swin** (2,169K)          | 4.88e-7 / -30.00 dB        | 7.46e-7 / -27.89 dB     | 4.56e-6 / -20.34 dB |
| **mlp_ae** (5,250K)        | 3.90e-7 / **-31.08 dB**    | 1.30e-6 / -25.52 dB     | 5.25e-6 / -19.77 dB |
| **sparse_resnet** (2,195K) | 1.90e-6 / -24.38 dB        | 1.31e-6 / -25.66 dB     | 5.25e-6 / -19.65 dB |

**结论**：

- **transnet decoder 表现最优**，尤其搭配 clnet (NMSE=-31.54 dB) 和 convnext (NMSE=-31.48 dB) encoder
- **hybrid decoder 仅次于 transnet**，最佳组合 transnet+hybrid (-28.41 dB)
- **cnn_residual decoder 最差**，所有组合 NMSE 均低于 -20 dB
- 完全相同的 encoder-decoder 类型（transnet+transnet）达到 -28.13 dB
- 参数效率上，clnet/transnet 用最少参数（~2.2M）达到最佳性能，而 mlp_ae 虽参数最多（~6.4M）但性能并非最优

### 1.3 COST2100 多 seed 统计 (transnet+hybrid)

| seed 范围      | 均值 test MSE | NMSE 均值 | NMSE 范围        |
| -------------- | ------------- | --------- | ---------------- |
| 59 个 seed     | ~1.12e-6      | ~-26.2 dB | -24.1 ~ -28.4 dB |
| seed=42 (最佳) | 7.18e-7       | -28.41 dB | -                |
| seed=3407      | 8.56e-7       | -27.56 dB | -                |
| seed=2026      | 1.15e-6       | -25.87 dB | -                |

### 1.4 WAIRD 场景

| seed | 模型                        | NMSE             |
| ---- | --------------------------- | ---------------- |
| 42   | transnet+transnet (~2,331K) | -23.96 dB        |
| 42   | transnet+hybrid (~2,571K)   | **-31.80 dB** ⭐ |
| 2026 | transnet+transnet           | -23.75 dB        |
| 2026 | transnet+hybrid             | **-31.10 dB**    |
| 0    | transnet+hybrid             | **-31.57 dB**    |

**结论**：WAIRD 场景下 **hybrid decoder 远优于 transnet decoder**（~8 dB 差距），与 COST2100 不同。

***

## 2. 码字空间映射器（Code-space Mapper）

### 2.1 映射器类型与参数量

| 映射器类型                   | 架构                                         | 可训练参数量 | 说明                              |
| ---------------------------- | -------------------------------------------- | ------------ | --------------------------------- |
| **IdentityMapper**           | 恒等映射                                     | **0**        | 无参数，仅传递码字                |
| **MLP Mapper**               | ResMLP (dim=512, h=2048, 4 blocks)           | **8,398K**   | 4× ResidualMLPBlock(512→2048→512) |
| **Delta MLP Mapper**         | Linear(512→2048) + 4× HiddenResBlock(h=2048) | **35,673K**  | 先升维再残差，参数量最大          |
| **Flow Mapper**              | 8× AffineCoupling(half=256, h=1024)          | **14,705K**  | 可逆流模型                        |
| **HybridFlowMLP**            | Flow + ResMLP 组合                           | **~23,103K** | Flow (~14.7M) + ResMLP (~8.4M)    |
| **AffineResMLP (h=512)**     | AffineAlign + 4× ResBlock(512→512→512)       | **2,102K**   | staged_mlp_lora 中使用            |
| **AffineResMLP (h=1024)**    | AffineAlign + 4× ResBlock(512→1024→512)      | **4,201K**   | 更好的 mapper                     |
| **Flow Matching Translator** | Alignment + VelocityMLP(h=2048, 4 blocks)    | **~142.9M**  | ODE 驱动生成式映射                |

**映射器参数量级别对比**：原始完整模型 ~2.3M / MLP 映射器 ~8.4M / Hybrid 映射器 ~23.1M / Flow Matching 映射器 ~142.9M

### 2.2 源→目标映射结果（seed2026→seed42, transnet+transnet）

| Mapper                              | 可训练参数量 | Code MSE | Decoder MSE | Decoder NMSE     |
| ----------------------------------- | ------------ | -------- | ----------- | ---------------- |
| 原教师码字 seed42                   | 0            | -        | 5.55e-7     | **-29.10 dB**    |
| **Flow Matching (warmcos, h=2048)** | **142.9M**   | 2.47e-5  | 5.59e-7     | **-29.07 dB** ⭐ |
| **Hybrid (smooth_tail_white)**      | **~23.1M**   | -        | 9.66e-7     | **-26.70 dB**    |
| **Hybrid (tail_whiten 1e-4)**       | **~23.1M**   | -        | 9.55e-7     | **-26.75 dB**    |
| Hybrid (基础)                       | ~23.1M       | 0.00312  | 1.31e-6     | -25.39 dB        |
| Flow Matching (basic ep=80)         | 142.9M       | 0.00197  | 8.69e-7     | -27.16 dB        |
| MLP (纯 MLP, h=2048)                | 8,398K       | 0.00405  | 1.51e-6     | -24.76 dB        |
| Flow (早期, 8 coupling)             | 14,705K      | 0.00686  | 2.17e-6     | -23.19 dB        |

### 2.3 跨 encoder 映射（seed2026 不同 encoder → seed42 transnet decoder）

| 源 Encoder | Mapper                     | 参数量 | Decoder NMSE  | 与原始差距 |
| ---------- | -------------------------- | ------ | ------------- | ---------- |
| transnet   | Hybrid (smooth_tail_white) | ~23.1M | -26.70 dB     | -1.13 dB   |
| **clnet**  | Hybrid (smooth_tail_white) | ~23.1M | **-25.68 dB** | -5.86 dB   |
| **crnet**  | Hybrid (smooth_tail_white) | ~23.1M | **-25.52 dB** | -0.52 dB   |
| **csinet** | Hybrid (smooth_tail_white) | ~23.1M | **-25.09 dB** | -3.98 dB   |
| clnet      | MLP                        | 8,398K | -24.08 dB     | -7.46 dB   |
| crnet      | MLP                        | 8,398K | -24.14 dB     | -0.86 dB   |
| csinet     | MLP                        | 8,398K | -22.66 dB     | -6.41 dB   |

**结论**：

- **跨 encoder 映射有显著性能损失**，尤其是 csinet→transnet 和 clnet→transnet
- **crnet→transnet 映射损失最小**（仅 0.5-0.9 dB）—— 说明 crnet 码字空间最接近 transnet
- **Hybrid mapper 整体优于纯 MLP mapper**，但参数量是后者的 2.75 倍

### 2.4 Decoder-Aware 映射（感知解码器的 Mapper）

| 源                | Mapper 类型                          | 参数量 | Decoder NMSE  |
| ----------------- | ------------------------------------ | ------ | ------------- |
| seed2026 transnet | Hybrid recT_rec_fc                   | ~23.1M | **-26.64 dB** |
| seed3407 transnet | Hybrid smooth_tail_white recT_rec_fc | ~23.1M | **-26.55 dB** |
| seed2026 clnet    | Hybrid recT_fc                       | ~23.1M | -25.60 dB     |
| seed2026 clnet    | Hybrid recT_rec                      | ~23.1M | -25.56 dB     |
| seed2026 crnet    | Hybrid smooth_tail_white recT_rec_fc | ~23.1M | -25.43 dB     |
| seed2026 csinet   | Hybrid smooth_tail_white recT_rec_fc | ~23.1M | -25.09 dB     |

**结论**：Decoder-aware 训练对跨 encoder 映射有一定改善，但参数量维持与 Hybrid mapper 相同。

***

## 3. LoRA 微调实验

本节只汇总 `decoder_lora/exps` 中已经完成并写出最终指标的实验；未跑完、没有 `metrics.json` 或日志中没有最终 `*_best_*` 指标的实验不纳入结论。需要特别区分两类指标：

- `train_all/all`：在训练 CSI 上解码评估，适合比较训练动态和参数量趋势，但不是泛化测试结果。
- `val_external/test_external`：在外部 val/test CSI 上评估，适合作为泛化结果。

### 3.1 LoRA 参数量修正

当前 `decoder_lora/models.py::LoRALinear` 对每个 Linear 新增：

```text
rank * in_features + rank * out_features
```

目标 transnet decoder 的 LoRA 注入点为：

```text
fc_decoder: 512 -> 2048
decoder FFN: 2 层，每层 linear1(64 -> 2048) + linear2(2048 -> 64)
```

因此 only-LoRA 参数量为：

```text
LoRA params = 2560 * fc_rank + 8448 * ffn_rank
```

| 配置          | LoRA 参数量 | 占 transnet decoder (~1.183M) |
| ------------- | ----------: | ----------------------------: |
| fc32 + ffn4   |     115,712 |                          9.8% |
| fc64 + ffn8   |     231,424 |                         19.6% |
| fc128 + ffn16 |     462,848 |                         39.1% |
| fc256 + ffn16 |     790,528 |                         66.8% |
| fc256 + ffn32 |     925,696 |                         78.2% |

### 3.2 Only-LoRA：有外部 val/test 的完成实验

下表使用 `best_loss` checkpoint，即外部 val loss 最优点；这是当前最适合横向比较泛化性能的选择。

| 源→目标                   | LoRA 配置                    | LoRA 参数量 |   epoch |  train NMSE | Raw train NMSE |    val NMSE | Raw val NMSE |   test NMSE | Raw test NMSE |
| ------------------------- | ---------------------------- | ----------: | ------: | ----------: | -------------- | ----------: | ------------ | ----------: | ------------- |
| seed2026 transnet→seed42  | fc256 a2048 + ffn32 a256     | **925,696** | **399** | **-28.045** | **-28.884**    | **-25.582** | **-27.692**  | **-25.547** | **-27.629**   |
| seed112 transnet→seed42   | fc256 a2048 + ffn32 a256     | **925,696** | **399** | **-26.489** | **-27.304**    | **-23.719** | **-26.651**  | **-23.741** | **-26.603**   |
| **seed2026 clnet→seed42** | **fc256 a2048 + ffn16 a128** | **790,528** | **397** | **-24.708** | **-30.938**    | **-21.781** | **-30.651**  | **-21.737** | **-30.602**   |

关键观察：

- seed2026 transnet 的效果最好，test 达到 **-25.55 dB**。
- seed112 transnet 的 test 约 **-23.74 dB**，明显低于 seed2026，说明不同源 seed 的 code space 可迁移性差异很大。
- 跨 encoder 的 seed2026 clnet→seed42 只有 **-21.74 dB**，only-LoRA 难以弥补架构差异导致的 code 分布差异。

### 3.3 Only-LoRA rank/alpha sweep（完成实验，train/all 指标）

以下 sweep 均为 seed2026 transnet→seed42，`align_mode=affine`，`code_adapter=none`，`lr=5e-4, eta_min=2e-4, epochs=400`。这些实验日志只有 `all_best_*` 指标，因此表中 NMSE 是训练集 `all` 指标，不应直接当作 test NMSE。

| 配置                     | alpha 比例 | LoRA 参数量 | best all NMSE | epoch |
| ------------------------ | ---------- | ----------: | ------------: | ----: |
| fc32 a256 + ffn4 a32     | alpha/r=8  |     115,712 |       -24.565 |   400 |
| fc64 a256 + ffn8 a32     | alpha/r=4  |     231,424 |       -25.530 |   400 |
| fc64 a512 + ffn8 a64     | alpha/r=8  |     231,424 |       -25.702 |   400 |
| fc64 a1024 + ffn8 a128   | alpha/r=16 |     231,424 |       -25.836 |   400 |
| fc128 a256 + ffn16 a32   | alpha/r=2  |     462,848 |       -26.364 |   400 |
| fc128 a512 + ffn16 a64   | alpha/r=4  |     462,848 |       -26.648 |   400 |
| fc128 a1024 + ffn16 a128 | alpha/r=8  |     462,848 |       -26.883 |   400 |
| fc128 a2048 + ffn16 a256 | alpha/r=16 |     462,848 |       -25.503 |    75 |
| fc256 a2048 + ffn16 a128 | alpha/r=8  |     790,528 |       -27.545 |   400 |
| fc256 a2048 + ffn32 a256 | alpha/r=8  |     925,696 |   **-28.035** |   400 |

结论：

- 在 train/all 指标上，参数量越大整体越好；`fc256+ffn32` 最强。
- `fc128+ffn16` 下，alpha/r 从 2 到 8 持续提升，但 alpha/r=16 明显不稳定，最优停在 epoch 75，最终没有继续改善。
- `fc64+ffn8` 的 alpha/r=16 略优于 8 和 4，但收益较小。

### 3.4 Code Adapter / AffineResMLP + LoRA 联合训练实验

当前完成的 `affine_res_mlp` 实验为：

| 源→目标                  | align    | code adapter         | LoRA 配置            | 总可训练参数 | 其中 LoRA | best all NMSE |
| ------------------------ | -------- | -------------------- | -------------------- | -----------: | --------: | ------------: |
| seed2026 transnet→seed42 | identity | AffineResMLP h512 b2 | fc32 a256 + ffn8 a64 |    1,203,200 |   149,504 |       -24.780 |

该配置没有外部 val/test 最终指标，且 train/all 也明显弱于 only-LoRA 的大 rank 配置。因此从现有完成结果看，`AffineResMLP h512 b2 + 小 LoRA` 不是当前最优路线。

### 3.5 LoRA 小结

- 当前最好的外部 test 结果来自 seed2026 transnet→seed42 的 only-LoRA：`fc256 a2048 + ffn32 a256`，test NMSE **-25.55 dB**，LoRA 参数 **925,696**。
- 这个结果仍弱于前文 Hybrid mapper 的约 **-26.70 dB**，但参数量远低于 Hybrid mapper（0.93M vs ~23.1M）。
- only-LoRA 对同架构不同 seed 有效；对跨 encoder，当前 clnet→transnet 只有 **-21.74 dB**，性能明显不足。

***

## 4. 码字空间 Adapter 实验

### 4.1 Adapter 模型参数量

| Adapter 类型                        | 文件                                     | 参数量（code_dim=512） | 架构                                                              |
| ----------------------------------- | ---------------------------------------- | ---------------------- | ----------------------------------------------------------------- |
| **MLPAdapter**                      | `mlp_adapter.py`                         | **5,243K**             | LN + Linear(512→2048→512)                                         |
| **MLPDirectAdapter**                | `mlp_direct_adapter.py`                  | **5,243K**             | 同上，无残差连接                                                  |
| **DiagonalAffineAdapter**           | `diagonal_affine_adapter.py`             | **1K**                 | gamma(512) + bias(512)                                            |
| **LowRankAffineAdapter**            | `lowrank_affine_adapter.py`              | **33K** (rank=32)      | Linear(512→32→512)                                                |
| **LowRankAffineMLPAdapter**         | `lowrank_affine_mlp_adapter.py`          | **5,276K** (rank=32)   | LowRank(~33K) + ResMLP(~5,243K)                                   |
| **GatedLowRankAffineLinearAdapter** | `gated_lowrank_affine_linear_adapter.py` | **295K** (rank=32)     | LowRank(~33K) + gate-Linear(~262K)                                |
| **GatedLowRankAffineMLPAdapter**    | `gated_lowrank_affine_mlp_adapter.py`    | **5,276K** (rank=32)   | LowRank(~33K) + gate-MLP(~5,243K)                                 |
| **TransformerAdapter**              | `transformer_adapter.py`                 | **1,873K**             | LN + Linear(512→512) + TransformerEnc(1层,d=64) + Linear(512→512) |

> 注意：adapter 是**额外**参数，插在 encoder 和 decoder 之间。原始模型（transnet+transnet, ~2,331K）加上 adapter 后总参数量 = ~7,607K（+MLPAdapter）。

### 4.2 MLP Adapter

| seed              | 实验配置              | Adapter 参数量 | 总模型参数量 | NMSE             |
| ----------------- | --------------------- | -------------- | ------------ | ---------------- |
| 42                | MLP adapter           | 5,243K         | **7,574K**   | **-28.02 dB**    |
| 3407              | MLP adapter (default) | 5,243K         | 7,574K       | -21.17 dB        |
| 2026              | MLP adapter           | 5,243K         | 7,574K       | -24.67 dB        |
| 1024              | MLP adapter           | 5,243K         | 7,574K       | -23.63 dB        |
| 796               | MLP adapter           | 5,243K         | 7,574K       | -23.51 dB        |
| cbam_cnn → hybrid | MLP adapter           | 5,243K         | 8,825K       | -22.07 dB        |
| clnet → hybrid    | MLP adapter           | 5,243K         | 8,716K       | -21.65~-22.16 dB |
| crnet → hybrid    | MLP adapter           | 5,243K         | 8,716K       | -21.04 dB        |

### 4.3 MLP Direct Adapter

| seed | Adapter 参数量 | NMSE          |
| ---- | -------------- | ------------- |
| 42   | 5,243K         | **-27.73 dB** |
| 2026 | 5,243K         | -24.91 dB     |
| 1024 | 5,243K         | -23.72 dB     |
| 3407 | 5,243K         | -21.46 dB     |

### 4.4 Transformer Adapter

| seed | Adapter 参数量 | NMSE      |
| ---- | -------------- | --------- |
| 3407 | **1,873K**     | -20.91 dB |

**结论**：Code adapter 在 seed=42 时表现接近原生 decoder（-28.02 dB），但增加 225% 参数量（2.3M→7.6M），且其他 seed 下降明显。Transformer Adapter 用最少适配参数量（1.9M）达到中等性能。

### 4.5 Teacher Code Adapter (gate_lowrank_affine_mlp)

| 源→目标         | 损失权重          | 适配器参数量 | 总模型参数量 | NMSE      |
| --------------- | ----------------- | ------------ | ------------ | --------- |
| seed2026→seed42 | code=1e-2 fc=1e-2 | 5,276K       | **7,607K**   | -20.78 dB |
| seed2026→seed42 | code=1e-3 fc=1e-2 | 5,276K       | 7,607K       | -20.81 dB |
| seed3407→seed42 | code=1e-3 fc=1e-2 | 5,276K       | 7,607K       | -20.28 dB |

**结论**：Teacher code adapter（5,276K 适配参数）在跨 seed 映射中表现中等（~-20.5 dB），较 Hybrid mapper（23.1M）差约 5 dB，但参数量仅为其 1/4。

### 4.6 Encoder Canonical Adapter（gated_lowrank_affine_mlp）

| 源                | 适配器参数量 | NMSE          |
| ----------------- | ------------ | ------------- |
| seed2026 transnet | 5,276K       | -24.74 dB     |
| seed42 clnet      | 5,276K       | -26.42 dB     |
| seed2026 clnet    | 5,276K       | **-26.24 dB** |
| seed3407 clnet    | 5,276K       | -26.43 dB     |
| seed2026 crnet    | 5,276K       | **-27.33 dB** |
| seed3407 crnet    | 5,276K       | -24.47 dB     |
| seed2026 csinet   | 5,276K       | -24.84 dB     |
| seed42 csinet     | 5,276K       | -25.89 dB     |

**结论**：Encoder canonical adapter（5.3M 额外参数）在跨 seed 同架构映射中效果最好（-26~-27 dB），参数量仅为 Hybrid mapper 的 1/4，但 NMSE 相近。

### 4.7 WAIRD Adapter

| seed | 配置               | 适配器参数量 | NMSE      |
| ---- | ------------------ | ------------ | --------- |
| 3407 | recon=0.0 code=1.0 | 5,276K       | -20.53 dB |
| 3407 | recon=1.0 code=0.0 | 5,276K       | -21.51 dB |

***

## 5. Flow Matching 码字映射

| 源                       | ODE steps        | 参数量     | Code MSE | Decoder NMSE     |
| ------------------------ | ---------------- | ---------- | -------- | ---------------- |
| seed2026 transnet→seed42 | 16 (warmcos)     | **142.9M** | 2.47e-5  | **-29.07 dB** ⭐ |
| seed2026 transnet→seed42 | 16 (best ep=80)  | 142.9M     | 0.00197  | -27.16 dB        |
| seed3407 transnet→seed42 | 16 (best ep=83)  | 142.9M     | 0.00266  | -26.37 dB        |
| seed2026 crnet→seed42    | 16 (best ep=132) | 142.9M     | 0.00297  | -25.56 dB        |
| seed2026 clnet→seed42    | 16 (best ep=75)  | 142.9M     | 0.00651  | -23.67 dB        |
| seed2026 csinet→seed42   | 16 (best ep=78)  | 142.9M     | 0.01214  | -20.14 dB        |

**结论**：Flow matching 在同架构映射中性能最优（-29.07 dB，接近 oracle -29.10 dB），但模型规模巨大（142.9M 参数，是原始模型的 61 倍），且需要 16 步 ODE 采样，推理开销大。

***

## 6. 总体对比总结

### 6.1 方法分类性能排名（同架构 seed→seed 场景，按 NMSE 排序）

| 排名 | 方法                                     | NMSE          | 方法参数量 | 原始模型参数     | 总参数量     | 相对原始 |
| ---- | ---------------------------------------- | ------------- | ---------- | ---------------- | ------------ | -------- |
| 🥇   | **Oracle 原始码字**                      | -29.10 dB     | 0          | 2,331K           | **2,331K**   | 1.0×     |
| 🥇   | **Flow Matching (warmcos)**              | **-29.07 dB** | 142,900K   | 0 (冻结)         | **142,900K** | 61.3×    |
| 🥈   | **Native clnet+transnet**                | **-31.54 dB** | 0 (完整)   | 2,233K           | **2,233K**   | 1.0×     |
| 🥉   | Native transnet+hybrid                   | -28.41 dB     | 0 (完整)   | 2,571K           | **2,571K**   | 1.1×     |
| 4    | MLP Adapter (seed=42)                    | -28.02 dB     | 5,243K     | 2,331K           | **7,574K**   | 3.2×     |
| 5    | Staged MLP(h=1024)+LoRA                  | -26.66 dB     | 4,345K     | 0 (冻结)         | **4,345K**   | 1.9×     |
| 6    | Hybrid Mapper (smooth_tail)              | -26.70 dB     | 23,103K    | 0 (冻结)         | **23,103K**  | 9.9×     |
| 7    | Flow Matching (basic ep=80)              | -27.16 dB     | 142,900K   | 0 (冻结)         | **142,900K** | 61.3×    |
| 8    | Decoder-Aware Hybrid                     | -26.64 dB     | 23,103K    | 0 (冻结)         | **23,103K**  | 9.9×     |
| 9    | Encoder Canonical Adapter                | -27.33 dB     | 5,276K     | 2,331K           | **7,607K**   | 3.3×     |
| 10   | Decoder LoRA fc256+ffn32 (external test) | -25.55 dB     | 926K       | 0 (+2,331K 冻结) | **926K**     | 0.40×    |
| 11   | MLP Mapper (h=2048)                      | -24.76 dB     | 8,398K     | 0 (冻结)         | **8,398K**   | 3.6×     |
| 12   | Teacher Code Adapter                     | -20.81 dB     | 5,276K     | 2,331K           | **7,607K**   | 3.3×     |

### 6.2 参数效率分析（每百万参数带来的 NMSE 提升 vs 原始模型）

从 seed=42 transnet+transnet (-28.13 dB) 出发：

| 方法                     | 额外参数           | NMSE 变化       | 参数效率（dB/M） |
| ------------------------ | ------------------ | --------------- | ---------------- |
| Decoder LoRA fc256+ffn32 | +0.93M             | +2.58 dB (变差) | 负效率           |
| Staged MLP(h=1024)+LoRA  | +4.35M (冻结原始)  | +1.47 dB (变差) | 负效率           |
| MLP Adapter (seed=42)    | +5.24M             | +0.11 dB (稍好) | +0.02 dB/M       |
| Encoder Canonical        | +5.28M             | +0.80 dB (稍好) | +0.15 dB/M       |
| Hybrid Mapper            | +23.10M (冻结原始) | +1.43 dB (变差) | 负效率           |
| Flow Matching            | +142.9M (冻结原始) | -0.94 dB (略好) | -0.007 dB/M      |

> **参数效率结论**：在额外增加参数后，无论是映射器还是适配器，都难以达到原始完整训练的性能。**最经济的方案是直接训练原生模型**——如果目标场景有数据可用的话。

### 6.3 跨 encoder 场景性能对比（带参数量）

| 源 Encoder→目标   | Hybrid Mapper      | MLP Mapper    | Flow Matching   | Decoder LoRA   | Staged MLP+LoRA   |
| ----------------- | ------------------ | ------------- | --------------- | -------------- | ----------------- |
| transnet→transnet | **-26.70** (23.1M) | -24.76 (8.4M) | -27.16 (142.9M) | -25.55 (0.93M) | **-26.66** (4.3M) |
| clnet→transnet    | **-25.68** (23.1M) | -24.08 (8.4M) | -23.67 (142.9M) | -21.74 (0.79M) | -25.09 (4.3M)     |
| crnet→transnet    | **-25.52** (23.1M) | -24.14 (8.4M) | -25.56 (142.9M) | 无当前完成记录 | -25.07 (4.3M)     |
| csinet→transnet   | **-25.09** (23.1M) | -22.66 (8.4M) | -20.14 (142.9M) | 无当前完成记录 | -23.02 (4.3M)     |

### 6.4 核心发现

1. **参数效率最优**：**原生 clnet+transnet** 用最少参数（2.2M）获得最佳性能（-31.54 dB）
2. **最佳零样本映射**：**Flow Matching（warmcos）** 在同架构映射中达到 -29.07 dB（几乎无损），但模型巨大（143M）
3. **最佳参数量性价比**：**Staged MLP(h=1024)+LoRA** 用 4.3M 参数达到 -26.66 dB，参数量仅为 Hybrid Mapper 的 1/5
4. **LoRA 低参数方案**：已完成 sweep 中最小配置 `fc32+ffn4` 需要 116K 参数，train/all NMSE 为 -24.57 dB；当前有外部 test 的最强 LoRA 是 `fc256+ffn32`，test NMSE 为 -25.55 dB
5. **跨 encoder 始终有损**：所有映射/适配方法在跨 encoder 时都有显著性能下降，crnet 编码器的码字最接近 transnet 空间，映射损失最小
6. **Adapter 方案（5.3M）** 在 seed=42 时表现接近原生（-28.02 dB vs -28.13 dB），但在其他 seed 下大幅下降，泛化能力不足

### 6.5 方法选择建议

| 使用场景            | 推荐方法                                 | 参数量     | 预期 NMSE          |
| ------------------- | ---------------------------------------- | ---------- | ------------------ |
| 有充足训练数据      | 原生 clnet+transnet                      | **2.2M**   | **-31.5 dB**       |
| 仅需少量额外参数    | Decoder LoRA fc32+ffn4（train/all 指标） | **0.12M**  | -24.6 dB           |
| LoRA 泛化结果最好   | Decoder LoRA fc256+ffn32                 | **0.93M**  | -25.5 dB           |
| 参数适中+性能平衡   | Staged MLP(h=1024)+LoRA                  | **4.3M**   | **-26.7 dB**       |
| 跨 encoder 通用方案 | Hybrid Mapper                            | **23.1M**  | **-25.1~-26.7 dB** |
| 追求极限映射性能    | Flow Matching                            | **142.9M** | **-29.1 dB**       |
| 无法接受推理开销    | MLP Adapter                              | **5.2M**   | -28.0 dB (seed=42) |

***

*报告生成时间：2025-07-06*  
*数据集：COST2100 室内场景（in），CR=4，input_dim=2048，code_dim=512*