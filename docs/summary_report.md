# UniversalCSI 码字空间对齐实验：总结分析

> 生成日期：2025-06-24
> 数据集：COST2100 indoor (100k train, 20k val, 20k test)
> 默认配置：encoder=transnet, decoder=hybrid, cr=4 (code_dim=512), input=(2,32,32)

---

## 1. 项目概况

本项目研究 CSI 反馈自编码器中的**码字空间对齐**问题：不同随机种子（或不同架构）训练出的 encoder 将同一 CSI 信号映射到互不兼容的码字空间，导致 encoder_A 的输出无法被 decoder_B 正确解码。

**核心 pipeline**：

```
CSI (B, 2, 32, 32) → [Frozen Encoder_seedX] → code (512D)
    → [Trainable Adapter] → adapted_code (512D)
    → [Frozen Decoder_seed42] → reconstructed CSI
    → MSE Loss
```

---

## 2. 实验全景

### 2.1 基线：seed42 联合训练 NMSE 矩阵

同一 seed (42) 下 14 种 encoder × 3 种 decoder 的联合训练结果（单位：dB，越低越好）：

| Encoder | transnet | cnn_residual | hybrid |
|----------|:---:|:---:|:---:|
| attention_cnn | -28.49 | -19.49 | -26.66 |
| cbam_cnn | -25.14 | -18.40 | -24.18 |
| clnet | **-31.54** | -18.51 | -27.88 |
| cnn | -27.20 | -20.02 | -25.27 |
| convnext | -31.48 | -20.31 | **-29.05** |
| crnet | -25.00 | -19.09 | -24.69 |
| csinet | -29.07 | -13.16 | -22.39 |
| dscnn | -21.88 | -18.18 | -22.59 |
| mlp_ae | -31.08 | -19.77 | -25.52 |
| mlp_mixer | -30.74 | -19.89 | -28.13 |
| resnet | -25.63 | -19.18 | -24.75 |
| sparse_resnet | -24.38 | -19.65 | -25.66 |
| swin | -30.00 | -20.34 | -27.89 |
| transnet | -28.13 | -20.25 | -28.41 |

**最佳组合**：
- transnet decoder 最佳：clnet (-31.54), convnext (-31.48), mlp_ae (-31.08)
- hybrid decoder 最佳：convnext (**-29.05**), mlp_mixer (-28.13), transnet (-28.41)
- cnn_residual decoder 整体较弱：最佳 swin (-20.34)

> Adapter 实验采用 **transnet encoder + hybrid decoder**（-28.41 dB 基线）。后续可探索用 clnet/convnext 等其他 encoder 做 adapter 源。

### 2.2 Adapter 实验结果

| Adapter | Seed | 训练前 NMSE (dB) | 最佳 NMSE (dB) | Δ | 联合训练基线 |
|---------|------|:---:|:---:|:---:|:---:|
| mlp (残差) | seed0 | +29.2 | -20.5 | 49.7 | -26.6 |
| mlp (残差) | seed42 | -27.9 | **-28.4** | 0.5 | -28.4 |
| mlp (残差) | seed520 | +26.6 | -23.6 | 50.2 | — |
| mlp (残差) | seed796 | +26.6 | -23.5 | 50.1 | — |
| mlp (残差) | seed1024 | +27.9 | -23.6 | 51.5 | — |
| mlp (残差) | seed2026 | +30.3 | **-24.7** | 55.0 | — |
| mlp (残差) | seed3407 | +23.9 | -21.2 | 45.1 | -27.1 |
| mlp (残差) + code distill | seed3407 | +23.9 | -20.5 | 44.4 | -27.1 |
| mlp_direct | seed0 | +26.8 | -20.7 | 47.5 | — |
| transformer | seed3407 | +23.9 | -20.9 | 44.8 | -27.1 |

> **注**：seed3407 + code distill 实验使用 `lambda_recon=0.0, lambda_code=1.0`（纯 code-space MSE）。

### 2.3 训练曲线特征

```
Epoch:  0   10   20   30   50   100   200   400
seed0:  +29  +0.5 -2.6 -10.8 -18.0 -19.3 -19.9 -20.5 dB
```

- **前 10 epoch** 完成 ~40% 的工作（+29 → +0.5 dB）
- **前 50 epoch** 完成 ~88%（→ -18.0 dB）
- 后续 350 epoch 仅再改善 2.5 dB，呈缓慢收敛态势

---

## 3. 码字空间分析

### 3.1 跨 Seed 码字统计

基于 7 个 seed（42, 0, 2026, 3407, 520, 796, 1024）的 `train_code.pt`（各 100k × 512）：

| Seed | norm_mean | norm_std |
|------|-----------|----------|
| 42 | 27.28 | 1.48 |
| 0 | 30.96 | 1.01 |
| 2026 | 23.91 | 1.86 |
| 3407 | 25.34 | 1.95 |
| 520 | 31.51 | 0.77 |
| 796 | 28.52 | 0.96 |
| 1024 | 28.88 | 1.34 |

**发现**：不同 seed 的码字范数统计差异显著（范围 23.9–31.5），说明编码器学到了不同尺度的表示。

### 3.2 跨 Seed 余弦相似度

| Seed pair | Cos Sim (mean) |
|-----------|:---:|
| 42 vs 0 | -0.011 |
| 42 vs 2026 | -0.026 |
| 42 vs 3407 | +0.059 |
| 42 vs 520 | +0.005 |
| 42 vs 796 | -0.006 |
| 42 vs 1024 | -0.028 |
| 0 vs 2026 | +0.033 |
| 3407 vs 520 | +0.060 |

**发现**：所有跨 seed 对之间的平均余弦相似度接近 **0**（范围 -0.03 ~ +0.06）。对于 512 维随机单位向量，期望余弦相似度约为 0 ± 0.04，说明跨 seed 码字**彼此正交**。

### 3.3 逐维度相关性

| Seed pair | Per-dim Pearson r (mean) | |r|>0.1 占比 |
|-----------|:---:|:---:|
| 42 vs 0 | -0.003 | 27.0% |
| 42 vs 2026 | +0.000 | 32.4% |
| 42 vs 3407 | +0.009 | 30.9% |
| 42 vs 520 | -0.001 | 33.8% |

**发现**：对应维度之间几乎无相关性（mean r ≈ 0）。`seed42` 的第 i 维与 `seed0` 的第 i 维没有对应关系——编码器使用了**完全不同的基底**。

### 3.4 线性可恢复性（核心发现）

| Seed pair | Linear R² (ridge λ=0.1) |
|-----------|:---:|
| 42 vs 0 | 0.918 |
| 42 vs 2026 | 0.961 |
| 42 vs 3407 | 0.939 |
| 42 vs 520 | 0.948 |

**关键发现**：尽管逐维度关系混乱（r≈0），但通过线性回归可以将另一个 seed 的码字**近乎完美地映射**到 seed42 的码字空间（R² 高达 0.91–0.96）。

**这意味着**：
- 不同 seed 的码字**张成的是近似的低维子空间**（可能只有远小于 512 的有效维度）
- 但它们使用了**不同的坐标系**（基底旋转/置换）
- Adapter 的核心任务本质上是**学习一个基底变换矩阵**，而不是学习全新的信息表示

### 3.5 方差重叠

```
R² (variance overlap) seed3407 vs seed42: 0.000078
Code MSE (raw seed3407 vs seed42): 2.56
Code NMSE: +2.45 dB (即 code 的 MSE 超过了信号功率)
```

两个码字空间的**方差几乎完全正交**——共享方差仅 0.008%。

### 3.6 综合分析

```
不同 Seed 的码字空间关系：

  R^512 空间中：
  
  seed42 码字子空间  ──── 旋转/置换 ────  seed0 码字子空间
       (~250 var)                                (~250 var)
  
  共享信息 ≈ 0（正交基底）
  线性可恢复性 R² ≈ 0.95（子空间几乎重合）
  
  矛盾？不矛盾！
  - "共享信息 ≈ 0" 指在对应维度坐标下，维度 i 和维度 i 不相关
  - "线性可恢复性" 指存在一个 512×512 的线性变换（即基底变换）
    可以将 seedA 的坐标无缝映射到 seedB 的坐标系
  
  类比：
    - 两本书用不同的字母表写就（基底不同）
    - 但只要学一个字母映射表（线性变换），就能读懂对方的内容
    - 字母映射表 ≈ Adapter 的核心工作
```

---

## 4. 核心发现总结

### 4.1 Adapter 的有效性

- ✅ **码字空间对齐是可行的**：2 层 MLP adater (512→2048→512, 2.1M 参数) 可将 NMSE 从 +29 dB 拉回到 -20~-25 dB
- ✅ **同源无损伤**：seed42→seed42 adapter 保持 -28.4 dB，残差结构 + 零初始化保证恒等起步
- ⚠️ **残余 gap 3–8 dB**：与联合训练基线（-28~-32 dB）仍有实质性差距
- ⚠️ **seed 间差异大**：最佳 seed2026（-24.7 dB），最差 seed0（-20.5 dB），差 4.2 dB

### 4.2 Adapter 类型比较

| | mlp (残差) | mlp_direct (无残差) | transformer |
|---|---|---|---|
| 参数量 | 2.10M | 2.10M | 1.05M |
| 初始化 | 恒等（W2=0） | 随机（W2~N(0,0.01²)） | 恒等 |
| 同源 seed | -28.1 ✅ | +26.9 ❌ | -28.1 ✅ |
| 跨 seed 平均 | **-22.7** | -22.7 | -20.9 |
| 结论 | **推荐** | 应废弃 | 参数更少，效果接近 |

- **残差 MLP 最优**：同源无损 + 跨 seed 表现稳定
- **无残差 MLP 不可用**：初始阻断信号，同源也从 +26.9 dB 起步
- **Transformer 参数效率更高**（1.05M vs 2.10M），但最终性能略低

### 4.3 Teacher Code 蒸馏的首次实验结果

```
seed3407, λ_recon=0.0, λ_code=1.0 (纯 code loss):
  NMSE: -20.48 dB  vs  重建 loss 的 -21.17 dB
  纯 code loss 反而更差！
```

**分析**：
- 纯 code-space MSE 忽略了**码字维度对重建结果的不同重要性**
- 重建 loss 通过 decoder 的 Jacobian 隐式加权了各维度
- 建议尝试 λ_recon + λ_code 联合，或使用 λ_recon >> λ_code

### 4.4 根本瓶颈

1. **码字维度并非全部同等重要**：部分维度主导重建质量，部分维度承载冗余信息
2. **Decoder 内部表征过拟合**：decoder 的前几层（`token_projection`, `semantic_projector`）已经过拟合到 seed42 码字的特定分布
3. **单点操作的局限**：仅在 512 维入口处做变换，无法纠正 decoder 内部各层的表征分布偏移

---

## 5. 改进方向（优先级排序）

### 方向 1：线性+非线性分解 Adapter ⭐⭐⭐⭐⭐

码字空间对齐的核心是**基底变换**（线性），叠加小量架构差异修正（非线性）。当前 2 层 MLP 将两者混合。

```python
out = Linear(512→512)(x) + SmallMLP(x)  # 线性主导 + 非线性修正
```

预期收益：加速收敛（线性部分可快速学习基底变换），提升上限。

### 方向 2：λ_recon + λ_code 联合训练 ⭐⭐⭐⭐

```bash
# 建议从小的 λ_code 开始
lambda_recon=1.0 lambda_code=0.1 adapter=mlp bash scripts/train_adapter.sh
```

纯 code loss 忽略了维度重要性，但作为**辅助正则项**可以引导 adapter 朝正确方向收敛。

### 方向 3：Decoder 内部 Adapter ⭐⭐⭐⭐

在 decoder 的前几层（`token_projection`, `semantic_projector`）之后各插入轻量 adapter（1×1 卷积或小 MLP），保持原始权重不变。

预期收益：从根本上解决"decoder 内部表征过拟合"问题。

### 方向 4：更深的 Adapter ⭐⭐⭐

当前 MLP 只有 1 个隐藏层。增加深度（3-4 层 MLP 或 2-3 层 Transformer）可能提升拟合能力。

### 方向 5：多 Seed 共享 Adapter ⭐⭐⭐

训练一个 adapter 同时服务多个 seed 的 encoder，利用多 seed 数据增强泛化性。

---

## 6. 代码与实验结构

### 6.1 关键文件

| 文件 | 作用 |
|------|------|
| `main.py` | 入口，组装训练管线 |
| `utils/parser.py` | CLI 参数定义（含 `--teacher_code`, `--lambda_recon`, `--lambda_code`） |
| `utils/solver.py` | `Trainer` 和 `Tester`，训练循环 + teacher code 蒸馏逻辑 |
| `utils/init.py` | 模型初始化、预训练权重加载与冻结 |
| `models/UniversalCSI.py` | 模型工厂 `universal_csi()`，`encode()` → `forward()` 管线 |
| `models/adapters/mlp_adapter.py` | 残差 MLP Adapter（推荐） |
| `models/adapters/mlp_direct_adapter.py` | 无残差 MLP Adapter |
| `models/adapters/transformer_adapter.py` | Transformer Adapter |
| `scripts/train_adapter.sh` | Adapter 训练启动脚本 |

### 6.2 数据流（带 teacher code）

```
train_loader → (sparse_gt, indices)
  │
  ├─→ model.encoder(sparse_gt) → raw_code
  │       └─→ code_adapter(raw_code) → adapted_code
  │               ├─→ decoder(adapted_code) → sparse_pred
  │               │       └─→ recon_loss = MSE(sparse_pred, sparse_gt)
  │               │
  │               └─→ teacher_codes[indices]
  │                       └─→ code_loss = MSE(adapted_code, teacher_code)
  │
  total_loss = λ_recon * recon_loss + λ_code * code_loss
```

### 6.3 实验目录结构

```
exps/COST2100/in/
├── seed42/              # 42 组联合训练基线 (14 enc × 3 dec)
│   └── transnet_hybrid/      # -28.41 dB 基线
├── seed{0,520,796,1024,2026,3407,...}/  # ~60 个独立种子联合训练
│   └── transnet_hybrid/
├── adapter/
│   ├── mlp/seed{0,42,520,796,1024,2026,3407}/        # 残差 MLP
│   ├── mlp_direct/seed{0,42,520,796,1024,2026,3407}/ # 无残差 MLP
│   └── transformer/seed3407/                          # Transformer
└── codewords/           # 预计算的 encoder 输出 (train_code.pt)
```

---

## 7. 快速复现命令

```bash
# 标准 adapter 训练（残差 MLP）
adapter=mlp seed=2026 gpu=6 bash scripts/train_adapter.sh

# 启用 teacher code 蒸馏（联合 loss）
teacher_code=exps/COST2100/in/seed42/transnet_hybrid/codewords/train_code.pt \
lambda_recon=1.0 lambda_code=0.1 \
adapter=mlp seed=2026 gpu=6 bash scripts/train_adapter.sh

# Transformer adapter
adapter=transformer adapter_hidden_dim=256 seed=3407 gpu=2 bash scripts/train_adapter.sh
```
