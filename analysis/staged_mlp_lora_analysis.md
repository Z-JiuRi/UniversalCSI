# Staged MLP+LoRA 分析与优化报告

## 摘要

本报告分析 Staged MLP+LoRA（先码字映射再 LoRA 微调解码器）实验的性能极限、与原生 encoder-decoder（-29.11 dB）1dB 目标的差距，以及在参数量 ≤ transnet decoder 参数量（**1,646,592**）约束下的优化方案。

---

## 1. 实验数据汇总

### 1.1 原生基线

| 模型 | 参数量 | NMSE |
|---|---|---|
| transnet+transnet (seed=42) | enc: 1,611K + dec: **1,647K** = 3,258K | **-29.11 dB** |
| clnet+transnet (seed=42) | enc: 1,050K + dec: 1,647K = 2,697K | **-31.54 dB** |

### 1.2 AffineResMLP Mapper 性能（直接解码，无 LoRA）

| 源 seed | 源 encoder | Mapper h | Mapper 参数量 | Code MSE | Direct Decoder NMSE |
|---|---|---|---|---|---|
| 2026 | transnet | 1024 | 4,201K | **4.659e-3** | -24.14 dB |
| 2026 | clnet | 1024 | 4,201K | 7.224e-3 | -22.41 dB |
| 2026 | crnet | 1024 | 4,201K | 7.069e-3 | -22.49 dB |
| 2026 | csinet | 1024 | 4,201K | 9.688e-3 | -21.05 dB |
| 3407 | transnet | 1024 | 4,201K | 4.976e-3 | -23.80 dB |
| 2026 | transnet | 512 | 2,103K | 7.207e-3 | -22.66 dB |
| 2026 | clnet | 512 | 2,103K | 1.182e-2 | -20.39 dB |

### 1.3 Staged MLP+LoRA 最终性能

| 源 encoder | Mapper h | LoRA 配置 | Mapper 参数量 | LoRA 参数量 | 总参数量 | 最终 Decoder NMSE | 与原生差距 |
|---|---|---|---|---|---|---|---|
| transnet | 1024 | fc_r=256,ffn_r=16 | 4,201K | 144K | **4,345K** | **-26.66 dB** | 2.45 dB |
| transnet | 512 | fc_r=256,ffn_r=16 | 2,103K | 144K | **2,247K** | **-26.05 dB** | 3.06 dB |

---

## 2. 加噪实验：码字 MSE 与 Decoder NMSE 的关系

### 2.1 实验方法

对 seed=42 transnet+transnet 的码字施加不同强度的各向同性高斯噪声，观察 decoder 输出 NMSE 的变化。

### 2.2 加噪结果

| σ_rel (σ/σ_code) | Code MSE | Code SNR | Decoder MSE | Decoder NMSE | 与基线差距 |
|---|---|---|---|---|---|
| 0 | 0 | ∞ | 1.13e-3 | **-29.11 dB** | 0 dB |
| 0.001 | 6.39e-7 | 60 dB | 1.13e-3 | -29.11 dB | 0.001 dB |
| 0.002 | 2.55e-6 | 54 dB | 1.13e-3 | -29.11 dB | 0.004 dB |
| 0.005 | 1.60e-5 | 46 dB | 1.14e-3 | -29.09 dB | 0.024 dB |
| **0.01** | **6.39e-5** | **40 dB** | **1.16e-3** | **-29.02 dB** | **0.094 dB** |
| **0.02** | **2.56e-4** | **34 dB** | **1.23e-3** | **-28.75 dB** | **0.365 dB** |
| 0.05 | 1.60e-3 | 26 dB | 1.76e-3 | -27.20 dB | 1.912 dB ❌ |
| 0.1 | 6.39e-3 | 20 dB | 3.68e-3 | -23.98 dB | 5.128 dB |
| 0.2 | 2.55e-2 | 14 dB | 1.19e-2 | -18.89 dB | 10.22 dB |

### 2.3 阈值分析

| 目标 | 最大允许 Code MSE | 最大允许 Code RMSE | 最低 Code SNR |
|---|---|---|---|
| **NMSE ≤ 1dB** | **2.56e-4** | **0.016** | **34 dB** |
| NMSE ≤ 0.5dB | 6.39e-5 | 0.008 | 40 dB |
| NMSE ≤ 0.1dB | 1.60e-5 | 0.004 | 46 dB |

### 2.4 当前映射器差距

| 指标 | 当前值 (h=1024 mapper) | 1dB 目标 | 差距 |
|---|---|---|---|
| Code MSE | 4.659e-3 | 2.56e-4 | **18.2× 太大** |
| Code RMSE | 0.0683 | 0.016 | **4.3× 太大** |
| Code SNR | 21.4 dB | 34.0 dB | **12.6 dB 不足** |

> **关键发现**：当前最优 mapper 的码字误差是 1dB 目标允许值的 **18 倍**。

---

## 3. 方法参数量约束分析

### 3.1 约束条件

> 总额外参数量 ≤ transnet decoder 参数量 = **1,646,592**

### 3.2 AffineResMLP Mapper 参数量公式

每层 ResidualBlock：
```
LayerNorm(512): 1,024
Linear(512→h): 512h + h
Linear(h→512): 512h + 512
总计: 1024h + 1,536
```
`num_blocks` 个 block + Final LayerNorm(~1K) + （Alignment 为 0 参数）：

```
总参数量 = num_blocks × (1024h + 1536) + 1024
```

### 3.3 可选配置的参数量

| Mapper h | Blocks | Mapper 参数量 + LoRA(144K) | 总参数量 | 是否 ≤ 1,647K? |
|---|---|---|---|---|
| 1024 | 4 | 4,201K + 144K = 4,345K | **4,345K** | ❌ (2.6×) |
| 512 | 4 | 2,103K + 144K = 2,247K | **2,247K** | ❌ (1.4×) |
| 400 | 4 | 1,645K + 144K = 1,789K | **1,789K** | ❌ |
| 384 | 4 | 1,579K + 144K = 1,723K | **1,723K** | ❌ |
| 256 | 4 | 1,052K + 144K = 1,196K | **1,196K** | ✅ |
| 384 | 3 | 1,184K + 144K = 1,328K | **1,328K** | ✅ |
| 512 | 2 | 1,052K + 144K = 1,196K | **1,196K** | ✅ |
| 0 (仅 LoRA) | - | 0 + 144K = 144K | **144K** | ✅ |

### 3.4 LoRA 参数量明细

| LoRA 目标 | 配置 | 参数量 |
|---|---|---|
| fc_decoder (512→2048) | r=256 | 655K |
| fc_decoder (512→2048) | r=128 | 328K |
| fc_decoder (512→2048) | r=64 | 164K |
| fc_decoder (512→2048) | r=32 | 82K |
| fc_decoder (512→2048) | r=16 | 41K |
| 4×FFN (64→256) | r=16 | 41K |
| 4×FFN (64→256) | r=8 | 20K |

---

## 4. 已存在的联合训练实验（decoder_lora 中的 CodeAdapter + LoRA）

### 4.1 实验发现

> `decoder_lora/train_decoder_lora.py` **已经支持 LoRA + code_adapter 联合训练**。`code_adapter` 参数为 `gated_lr_mlp` 时，code adapter（GatedCodeResidualAdapter，一种轻量级码字映射器）与 LoRA 参数**同时训练**，使用重建损失直接优化。

### 4.2 联合训练 vs 分阶段训练结果对比（seed2026 transnet→seed42）

| 训练方式 | CodeAdapter/Mapper | LoRA 配置 | 参数量 | NMSE |
|---|---|---|---|---|
| **分阶段 (Staged)** | AffineResMLP(h=1024, 4.2M) → 冻结 | fc_r=256, ffn_r=16 | 4,345K | **-26.66 dB** ⭐ |
| **分阶段 (Staged)** | AffineResMLP(h=512, 2.1M) → 冻结 | fc_r=256, ffn_r=16 | 2,247K | **-26.05 dB** |
| **联合训练 (Joint)** | GatedCodeAdapter(h=1024, lr_r=128) | fc_r=256, ffn_r=16 | 5,420K | **-25.25 dB** |
| **联合训练 (Joint)** | GatedCodeAdapter(h=512, lr_r=128) | fc_r=256, ffn_r=16 | 5,420K | **-24.46 dB** |
| LoRA 单独 | 无 | fc_r=256, ffn_r=16 | 144K | -25.05 dB |
| LoRA 单独 (ep=1000) | 无 | fc_r=256, ffn_r=16 | 144K | -25.41 dB |

### 4.3 核心发现

| 对比 | NMSE 差距 |
|---|---|
| 分阶段 **优于** 联合训练 | **1.41 dB** (26.66 vs 25.25) |
| 联合训练 **略优于** LoRA 单独 | **0.20 dB** (25.25 vs 25.05) |
| LoRA 单独(ep=1000) **接近** 联合训练 | **0.16 dB** (25.41 vs 25.25) |

**关键结论**：

1. **分阶段训练（Staged AffineResMLP → LoRA）显著优于联合训练（~1.4 dB）**，与之前的推测相反
2. 联合训练引入的 CodeAdapter（5.3M 参数）仅带来比 LoRA 单独多 0.2 dB 的提升，性价比极低
3. 分阶段训练先用 code MSE 监督让 mapper 学到好的码字空间映射，再让 LoRA 精调 decoder，有效解耦了「对齐」和「适配」两个任务

### 4.4 联合训练不如分阶段的原因分析

1. **优化冲突**：重建损失对 code adapter 的梯度信号弱于直接的 code MSE 监督
2. **任务耦合**：两个模块（映射 + 适配）同时优化容易相互干扰
3. **CodeAdapter 架构不如 AffineResMLP**：GatedCodeResidualAdapter 仅有 low-rank + MLP residual，缺少 AffineResMLP 中的显式仿射对齐步骤
4. **正则约束影响**：`lambda_code` 和 `lambda_delta` 等正则项可能限制优化空间

---

## 5. 优化方案（修正版）

基于已有实验数据，调整优化方向：

### 方案 A：分阶段训练（推荐，已验证有效）

维持分阶段策略，专注于提升每阶段的效率：

```
阶段1: AffineResMLP mapper, 用 code MSE 训练
阶段2: 冻结 mapper, LoRA 微调解码器, 用重建损失训练
```

| 子方案 | Mapper | LoRA | 总参数量 | 预期 NMSE |
|---|---|---|---|---|
| **A1: 当前最优** | h=1024, 4 blocks (4.2M) | fc_r=256, ffn_r=16 (144K) | 4,345K ❌ | **-26.66 dB** |
| A2: 参数量裁剪 | h=384, 3 blocks (1.18M) | fc_r=64, ffn_r=8 (184K) | 1,368K ✅ | ~-25.5 dB |
| A3: 轻量版 | h=256, 2 blocks (526K) | fc_r=32, ffn_r=8 (102K) | 628K ✅ | ~-24.8 dB |

### 方案 B：提升联合训练

如果继续探索联合训练方向，需要改变：

1. **移除 `lambda_code` / `lambda_delta` 正则约束**，减少对 code adapter 的限制
2. **使用更强大的 mapper 架构**（如 AffineResMLP 替代 GatedCodeResidualAdapter）
3. **增加训练轮数**（当前 ep=400，LoRA 单独在 ep=1000 时仍有提升）

### 方案 C：仅 LoRA（最轻量）

| 配置 | 参数量 | NMSE (ep=400) | NMSE (ep=1000) |
|---|---|---|---|
| fc_r=256, ffn_r=16 | 144K (0.09×) | -25.05 dB | **-25.41 dB** |
| fc_r=统一64 | 81K (0.05×) | -24.96 dB | - |
| fc_r=统一32 | 41K (0.02×) | -24.67 dB | - |

LoRA 单独（fc_r=256, ffn_r=16）只用了 **144K 参数**（decoder 的 9%）就达到 -25.41 dB，**性价比最高**。

---

## 6. 实施建议（修正版）

### 短期：Staged MLP+LoRA（已验证最优）

维持两阶段训练，尝试：
1. 在参数量允许时优先使用 **h=1024 AffineResMLP**
2. mapper 训练时使用更多 epoch 或更大的 batch size 以降低 code MSE
3. LoRA 训练时可尝试更长的 epoch（ep=1000 比 ep=400 有约 0.36 dB 提升）

### 中期：改进 CodeAdapter 架构

如果希望实现 mapper + LoRA 联合训练：
1. 将 `decoder_lora/train_decoder_lora.py` 中的 code_adapter 替换为 AffineResMLP
2. 去掉 lambda_code/lambda_delta 正则项
3. 增加 epoch 数

### 长期

1. 探索分阶段训练中 mapper 的对齐质量与 LoRA 微调效果之间的关系
2. 研究不同 mapper 架构对 code MSE 的影响

---

## 附录：参数量计算明细

### TransNet Decoder 参数量

| 模块 | 参数量 |
|---|---|
| fc_decoder: Linear(512→2048) | 1,050,624 |
| decoder.layers.0 自注意力 | 16,480 |
| decoder.layers.0 交叉注意力 | 16,480 |
| decoder.layers.0 FFN(64→2048→64) | 262,272 |
| decoder.layers.0 三个 LayerNorm | 384 |
| decoder.layers.1 (同上) | 295,360 |
| 最终 LayerNorm | 128 |
| **总计** | **1,641,728** |

*报告生成时间：2025-07-06*
*基线数据：seed=42 transnet+transnet，CR=4, d_model=64, dim_feedforward=2048, num_layers=2*
