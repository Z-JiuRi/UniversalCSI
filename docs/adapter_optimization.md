# Adapter 性能优化分析

## 当前状态回顾

**结构**: `encoder(x) → code(512d) → [MLPAdapter] → code'(512d) → frozen decoder → output`

**MLPAdapter (残差版)**: `y = x + W2 @ GELU(W1 @ LN(x))`，W1=kaiming, W2=zero，从恒等起步。

**训练配置**: encoder/decoder 均冻结，仅 adapter 可训练（~2M 参数）。

**已修复**: 初始化 bug（`_reset_parameters` 覆盖 + 双零梯度死锁）。

---

## 一、当前结构的根本局限

### 1.1 单点操作的粒度问题

adapter 只在 code 这一个点上工作。code 是 encoder 的最紧致表示——512 维向量，每个维度承载高度压缩的全局信息。在这个级别做变换需要极高的精度：一点点扰动就在 decoder 端被多层非线性放大。

类比：要在 JPEG 压缩后的 DCT 系数上修改图像内容，比在像素空间修改难得多。

### 1.2 容量可能不足

2 层 MLP (512→2048→512) 的理论最大表达力是有限的。跨 seed 的码字空间对齐可能不是一个"平滑函数"——不同 seed 训练出的 encoder 收敛到 loss landscape 中完全不同的局部极小值，两点之间的变换可能需要更复杂的结构。

### 1.3 损失函数过于间接

adapter 唯一的监督信号是最终 CSI 重建的 MSE。adapter 的输出经过 decoder 的 semantic_projector、token_projection、token_mixer、CNN refine 四层变换才到达输出。梯度经过这条长链反向传播，信号被稀释。adapter 收不到关于"你的输出离目标 code 有多远"的直接反馈。

---

## 二、架构优化方向

### 2.1 增加深度

```
当前: LN → Linear(512→2048) → GELU → Linear(2048→512) → +x
方案A: LN → Linear(512→1024) → GELU → Linear(1024→1024) → GELU → Linear(1024→512) → +x
方案B: LN → Linear(512→512) → GELU → Linear(512→512) → GELU → Linear(512→512) → +x
```

| 方案 | 参数 | 优势 |
|------|------|------|
| 当前 | 2M | 最简单 |
| A (3层，瓶颈1024) | ~1.6M | 更深的非线性 |
| B (3层，无瓶颈) | ~0.8M | 每层维度不变，无信息压缩 |

方案 B 值得先试——参数量反而更少（每层 512×512），但多一层非线性，且避免了 512→2048→512 的极端 expansion/compression。

### 2.2 改为低秩分解 (LoRA 风格)

```
当前: LN → Linear(512→2048) → GELU → Linear(2048→512) → +x
方案:  LN → Linear(512→r) → Linear(r→512) → +x   (r = 4/8/16)
```

不做非线性膨胀，用极低秩矩阵分解。参数量从 2M → r×512×2 ≈ 4K~16K。

**为什么可能更好**: 
- 低秩约束相当于在码字空间的低维子空间里做变换，天然正则化
- 参数量极少，优化快，不易过拟合
- 跨 seed 的码字空间差异可能确实分布在低秩子空间里

### 2.3 引入 Transformer block

```
方案: LN → Self-Attention(512, 8 heads) → FFN(512→1024→512) → +x
```

用一个 Transformer encoder layer 替代 MLP。Self-attention 可以捕获 code 各维度之间的交互关系——这比全连接更灵活，因为 attention 权重是数据相关的。

参数: self-attention (512×512×4 ≈ 1M) + FFN (512×1024×2 ≈ 1M) ≈ 2M，和当前持平。

### 2.4 改为 convolutional-style 1D 处理

将 code 视为 1D 序列（或 reshape 为 16×32），用 1D/2D 卷积处理：

```
code(512) → reshape(16, 32) → Conv1d → ... → flatten → Linear → code(512)
```

只在 code 内部做局部的 token 级别交互，而不是全局全连接。参数量可控（取决于 kernel size 和通道数）。

### 2.5 多位置插入 adapter

不仅在 code 级别，在 decoder 内部的关键位置也插入小型 adapter：

```
encoder → code → [adapter_1] → decoder.semantic_projector → [adapter_2] → ... → decoder.token_mixer → [adapter_3] → ...
```

多级 adapter 分担压力：code 级做粗对齐，decoder 内部做细调整。每个 adapter 可以更小。

这需要恢复之前删除的 decoder 内部 adapter 机制，但设计可以更简洁——比如只在 semantic_projector 后加一个 adapter。

---

## 三、训练策略优化

### 3.1 分阶段训练

```
阶段1 (warmup): 只训练 W2，W1 冻结
  - W2 从 0 开始学习将 W1 的随机投影映射到目标
  - 避免 W1 和 W2 同时变化导致的不稳定
阶段2 (full): W1 + W2 联合训练
```

### 3.2 学习率 warmup + 更大的 lr

当前 lr=2e-4，对只有 2M 参数的 adapter 可能偏小。adapter 参数是全新的（encoder/decoder 是预训练的），可以从更高 lr 起步：

```
lr=1e-3, warmup=10% epochs, cosine decay to 1e-5
```

### 3.3 辅助损失：code-level alignment

在 adapter 的输出和"目标 seed 的 code"之间加一个辅助 MSE：

```
L_total = L_recon + λ * ||adapted_code - target_code||²
```

但这需要知道"目标 seed 的 code"——需要另一个 encoder inference。代价较大但信号直接。

### 3.4 逐步解冻

```
epoch 1-50:  只训练 adapter
epoch 51-100: adapter + decoder 的 LayerNorm 参数解冻
epoch 101-150: adapter + decoder 的最后一层解冻
...
```

给 adapter 一个独享的优化窗口，然后逐步引入 decoder 的微调。

---

## 四、诊断建议

在修改架构之前，先通过以下实验定位瓶颈：

### 4.1 验证 adapter 是否真的在学习

观察 `delta_ratio` 的轨迹：
- 如果始终接近 0 → adapter 没有在改变 code，梯度可能有问题或 lr 太小
- 如果快速增长到 >1 然后震荡 → adapter 在过度修改 code，需要降低 lr 或加正则
- 如果缓慢增长到 0.1~0.3 并稳定 → adapter 在学习，问题在架构容量

### 4.2 Sanity check: 同 seed 训练

用同一个 seed 的 encoder 和 decoder（即 adapter 理论上应该学习恒等映射）：

```
pretrained_encoder = seed42/checkpoint
pretrained_decoder = seed42/checkpoint  (同一个 checkpoint)
```

如果这个配置下 adapter 的 NMSE 也差 → 训练本身有问题  
如果 NMSE 接近 joint training → adapter 概念可行，问题在跨 seed 变换的难度

### 4.3 逐步增加 adapter 容量

从最简单的开始试：

```
1. LoRA r=4   (8K params)
2. LoRA r=16  (32K params)  
3. 3层 MLP 无瓶颈 (0.8M params)
4. Transformer block (2M params)
```

如果 LoRA r=4 已经能显著提升 → 问题不是容量，而是需要低秩约束  
如果 Transformer 才能提升 → 需要 attention 的跨维度交互

---

## 五、推荐实验顺序

| 优先级 | 实验 | 理由 |
|:--:|------|------|
| 1 | **Sanity check: 同 seed** | 排除训练本身的问题 |
| 2 | **LoRA r=8/16** | 改动最小，验证低秩假设 |
| 3 | **3 层 MLP 无瓶颈 (512→512→512→512)** | 更深但参数更少 |
| 4 | **分阶段训练** | 不改变架构，优化训练过程 |
| 5 | **Transformer block** | 架构升级 |
| 6 | **多位置 adapter** | 最大改动 |

---

## 六、代码修改建议

以下架构变体都遵循当前 adapter 接口 (`forward(x) → y`)，可以作为 `--adapter` 的新选项添加：

| `--adapter` 值 | 结构 |
|----------------|------|
| `mlp` | 当前残差 MLP (2 层) |
| `mlp_direct` | 当前无残差 MLP (2 层) |
| `mlp_deep` | 残差 MLP (3 层，无瓶颈) |
| `lora` | LoRA 低秩分解 (`--adapter_hidden_dim` 控制秩 r) |
| `transformer` | 1 层 Transformer encoder block |

实现时创建一个 `models/adapters/` 下的新文件 + 在 `universal_csi()` 中注册即可，改动隔离，互不影响。
