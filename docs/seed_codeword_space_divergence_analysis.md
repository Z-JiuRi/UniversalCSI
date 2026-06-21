# 不同 Seed 码字空间差异的深层分析与对齐方法

> 本文基于 UniversalCSI 仓库源码和 `exps/COST2100/in/` 下 59 组跨 seed 实验，
> 从代码层面逐层分析为什么不同随机种子训练出的 encoder 码字空间完全不可通用，
> 以及如何在训练或推理阶段缩小这种差异。

---

## 目录

1. [现象描述：跨 seed 码字空间不兼容](#1-现象描述)
2. [表层原因：随机种子如何影响代码执行](#2-表层原因)
3. [中层原因：初始化差异如何在训练中被放大](#3-中层原因)
4. [深层原因：为什么自编码器本质上不约束码字空间](#4-深层原因)
5. [源码级根因链条总览](#5-源码级根因链条)
6. [实验数据佐证](#6-实验数据佐证)
7. [对齐方法：如何缩小不同 seed 的码字空间差异](#7-对齐方法)
8. [Adapter 路线可行性分析与改进](#8-adapter-路线)
9. [总结与建议](#9-总结)

---

## 1. 现象描述

在 `exps/COST2100/in/` 目录下，59 个不同 seed 独立训练了 `transnet_hybrid`
自编码器。每个实验各自的 NMSE 都在 `-24 ~ -28 dB` 的正常范围内：

| 指标 | 值 |
|---|---|
| 59 个 seed 的 NMSE 均值 | -25.96 dB |
| NMSE 标准差 | 0.81 dB |
| 最好 (seed42) | -28.41 dB |
| 最差 (seed4442) | -24.07 dB |
| seed 间极差 | ~4.3 dB |

**但当把 seed1 的 encoder 权重拼接 seed2 的 decoder 权重时，NMSE 劣化到
`+15 ~ +29 dB`（正数），完全无法起作用。** 具体实验记录：

| 组合 | 错配 NMSE |
|---|---|
| seed3407 encoder + seed42 hybrid decoder | **+23.854 dB** |
| seed3407 encoder + seed42 transnet decoder | **+15.461 dB** |
| seed42 encoder + seed2026 hybrid decoder | **+28.634 dB** |

正的 NMSE 意味着重建输出的误差比输入信号功率还大——decoder 对跨 seed 码字
完全"读不懂"。

**核心问题**：为什么同一个架构、同一个数据集、仅改变随机种子，训练出来的
encoder 码字空间就完全不可互换？

---

## 2. 表层原因：随机种子如何影响代码执行

### 2.1 Seed 在代码中的设置链路

训练入口 `main.py` 调用 `init_device(seed, ...)` → `seed_everything(seed)`：

```python
# utils/init.py, L12-L18
def seed_everything(seed):
    random.seed(seed)           # Python 标准库随机数
    torch.manual_seed(seed)     # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU RNG
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

这个函数在 **模型构建和数据加载之前** 被调用（`main.py` L27），因此 seed
直接控制了以下所有后续随机过程：

### 2.2 Seed 影响的四个关键环节

```
seed ─┬─→ ① 模型权重初始化
      │     (nn.init.xavier_uniform_, nn.init.kaiming_uniform_)
      │
      ├─→ ② 训练数据 batch 顺序
      │     (DataLoader shuffle=True 的排列)
      │
      ├─→ ③ AdamW 优化器的初始动量/方差估计
      │     (不同 batch 顺序 → 不同梯度 → 不同 Adam 状态)
      │
      └─→ ④ cuDNN 算法选择的确定性
            (deterministic=True, benchmark=False)
```

> **注意**：当前代码 **没有** 设置 `numpy.random.seed()`，也 **没有** 使用
> dropout（所有 Transformer 层的 `dropout=0.0`），也 **没有** 数据增强。
> 因此推理阶段没有额外随机性。

### 2.3 具体来说，seed 改变了什么

#### ① 权重初始化

`UniversalCSIModel.__init__` 在构建完 encoder 和 decoder 之后，立即调用
`_reset_parameters(strategy)` 重新初始化所有权重：

```python
# models/UniversalCSI.py, L79-L87
class UniversalCSIModel(nn.Module):
    def __init__(self, encoder, decoder, init_strategy="typed"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_strategy = init_strategy
        self._reset_parameters(init_strategy)
```

对于 `transnet_hybrid` 组合，`select_init_strategy()` 返回 `"typed"`，
进入分类型初始化：

```python
# models/UniversalCSI.py, L102-L126
def _reset_typed_parameters(self):
    for module in self.modules():
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.constant_(module.in_proj_bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)    # ← 依赖 torch RNG
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.Conv2d, ...)):
            nn.init.kaiming_uniform_(module.weight, a=0.3,
                                      nonlinearity="leaky_relu")  # ← 依赖 torch RNG
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, ...)):
            nn.init.constant_(module.weight, 1)   # ← 确定性
            nn.init.constant_(module.bias, 0)      # ← 确定性
```

**关键观察**：`xavier_uniform_` 和 `kaiming_uniform_` 的数值完全由当前
`torch` RNG 状态决定。不同 seed → 不同 RNG 状态 → 不同初始权重。

模型的总参数量取决于具体架构，以 `transnet_hybrid` 为例（`cr=4, d_model=64,
dim_feedforward=2048, hidden=16, num_blocks=2`）：

| 组件 | 参数量 |
|---|---|
| TransNet Encoder (2层 Transformer + fc) | ~1.6M |
| Hybrid Decoder (semantic_projector + Transformer + CNN refine) | ~3.0M |
| **合计** | **~4.6M** |

这约 460 万个参数的初始值都受 seed 控制。

#### ② 训练数据 batch 顺序

```python
# dataloader/dataloader.py, L90-L94
train_loader = DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          ...
                          shuffle=True)   # ← 每个 epoch 重新洗牌
```

`DataLoader` 内部使用 `torch.randperm()` 进行 shuffle，其随机序列由
`torch.manual_seed(seed)` 决定。不同 seed → 每个 epoch 的 batch 排列不同 →
梯度顺序不同。

#### ③ 优化器轨迹

AdamW 维护了每个参数的一阶动量 $m_t$ 和二阶动量 $v_t$：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

不同的初始权重和不同的 batch 顺序导致每一步的梯度 $g_t$ 都不同，
因此优化轨迹在参数空间中走出完全不同的路径。

#### ④ 学习率调度

当前使用的 `WarmUpCosineAnnealingLR` 是 **per-step** 调度（在 `_iteration`
的每个 batch 后执行 `scheduler.step()`）。调度器本身是确定性的，但由于模型
看到同一个 epoch 内的 batch 顺序不同，相同 step 数下模型所处的参数空间位置
已经不同。

---

## 3. 中层原因：初始化差异如何在训练中被放大

### 3.1 损失景观的多极值性质

CSI 自编码器的 MSE 损失：

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| D(E(x_i)) - x_i \|^2$$

这个目标函数关于 encoder 和 decoder 的联合参数空间 **高度非凸**。具体来说，
对于一个具有 460 万参数的模型，损失景观中存在无数个局部最小值和鞍点。

**不同的初始化点 + 不同的梯度路径 → 收敛到损失景观中不同的"盆地"（basin）。**
虽然这些盆地的最终 MSE 值可能相近（都在 -25~-28 dB），但它们对应的参数
空间位置完全不同。

### 3.2 对称性与等价解

自编码器损失只约束 $D(E(x)) \approx x$，但对 $E(x)$ 的具体取值没有任何
偏好。这意味着存在庞大的等价解空间：

**定理（自编码器等价变换）**：设 $E^*$, $D^*$ 是一组最优解，
则对于任意可逆变换 $T: \mathbb{R}^{d} \to \mathbb{R}^{d}$，
$E' = T \circ E^*$, $D' = D^* \circ T^{-1}$ 也是最优解，
因为 $D'(E'(x)) = D^*(T^{-1}(T(E^*(x)))) = D^*(E^*(x)) = x$。

在实际神经网络中，虽然网络的有限容量限制了精确的等价变换，但这种对称性仍然
以近似形式存在。最常见的等价变换包括：

| 变换类型 | 说明 | 对跨 seed 的影响 |
|---|---|---|
| **线性旋转/反射** | 正交矩阵 $T$ 作用于码字空间 | 改变码字坐标系但保持几何结构 |
| **缩放** | 对角矩阵 $T$ | 改变码字各维度的尺度 |
| **维度置换** | 置换矩阵 $T$ | 打乱码字维度的语义分配 |
| **非线性扭曲** | 非线性 $T$（有限网络容量下的近似） | 扭曲码字空间的局部几何 |
| **符号翻转** | 某些维度的 $\pm 1$ 缩放 | 码字的正负极性翻转 |

> 这些变换在不同 seed 的训练中随机发生。两个 seed 学到的解可能只是一个
> 旋转/缩放/置换的关系，但对于固定了内部权重的 decoder 来说，输入任何一种
> "错误坐标系"的码字都会导致灾难性失败。

### 3.3 梯度对称性破缺的链式放大

训练的第一步就打破了对称性。设两个 seed 的初始参数为 $\theta_0^{(A)}$ 和
$\theta_0^{(B)}$，第一步梯度：

$$g_1^{(A)} = \nabla_\theta \mathcal{L}(\theta_0^{(A)}; \text{batch}_1^{(A)})$$
$$g_1^{(B)} = \nabla_\theta \mathcal{L}(\theta_0^{(B)}; \text{batch}_1^{(B)})$$

- $\theta_0^{(A)} \neq \theta_0^{(B)}$（初始化不同）
- $\text{batch}_1^{(A)} \neq \text{batch}_1^{(B)}$（shuffle 顺序不同）

因此 $g_1^{(A)}$ 和 $g_1^{(B)}$ 指向完全不同的方向。经过 400 个 epoch 的
训练（默认设置），每步的微小差异通过非线性梯度传播链式累积，最终导致
encoder 和 decoder 收敛到不同的"约定"（convention）。

### 3.4 从 transnet 编码器结构看差异来源

TransNet encoder 的结构：

```
View(N, seq_len=32, d_model=64) → TransformerEncoder(2层) → Flatten → Linear(2048, 512)
```

其中 `TransformerEncoder` 的 `MultiheadAttention`（2 头）对输入 token 做
全局混合。不同初始化导致：

- **注意力权重矩阵** `in_proj_weight: (192, 64)` 的初始值不同
  → 关注不同的 token 子空间
  → 形成不同的 token 混合模式
  → 输出的 token 表示含义不同
  → 后续 `fc: (2048, 512)` 被迫学习不同的线性投影

- **最终全连接层 fc** 的初始值不同
  → 即使前面的 Transformer 输出相同的特征，fc 也会将其投影到
  码字空间的不同方向

这两层差异叠加，使得不同 seed 的 encoder 输出的 512 维码字在数值分布和
语义上完全不同。

### 3.5 从 hybrid 解码器结构看敏感性

Hybrid Decoder 的解码链路：

```
code(512) → semantic_projector(LN + Linear) → token_projection(512→2048)
→ View(32, 64) → TransformerEncoder(2层) + LN → View(2,32,32)
→ CNNRefinementHead(refine) + residual_scale
```

解码的第一步就是 `semantic_projector` 和 `token_projection`。这两个模块
的权重是与特定 seed 的 encoder 共同训练的，形成了对特定码字坐标系的"期望"。

当输入一个来自不同 seed 的码字时：

1. **`semantic_projector`**（`LayerNorm + Linear`）：LayerNorm 期望特定的
   均值/方差分布，Linear 期望特定方向的输入。跨 seed 码字可能在完全不同的
   数值范围，导致 LayerNorm 的归一化效果扭曲。

2. **`token_projection`** (`Linear(512, 2048)`)：这个 512→2048 的线性映射
   是整个解码器最脆弱的环节。它学到的是"seed_A 的码字第 i 维 → token 序列
   第 j 个位置的第 k 个特征"的映射。跨 seed 码字的维度语义不同，这个映射
   直接失效。

3. **后续 Transformer 层**：接收到错误的 token 表示后，自注意力进一步放大
   错误，因为注意力权重是针对正确 token 分布学习的。

这就解释了为什么错配的 NMSE 不是温和地劣化（例如 -20 dB），而是灾难性崩溃
到 +20~+29 dB。

---

## 4. 深层原因：为什么自编码器本质上不约束码字空间

### 4.1 损失函数的"盲区"

当前训练的唯一损失是：

```python
# main.py, L49
criterion = nn.MSELoss()

# 训练时:
sparse_pred = model(sparse_gt)   # model(x) = decoder(encoder(x))
loss = criterion(sparse_pred, sparse_gt)
```

这个 MSE 损失只关心 **端到端重建质量**，完全不关心中间的码字 `encoder(x)`
长什么样。在优化理论中，码字空间是损失函数的一个"自由变量"——只要
`decoder(code) ≈ x`，code 取什么值都可以。

### 4.2 数学分析：码字空间的不确定性

设训练集为 $\{x_i\}_{i=1}^N$，最优自编码器满足：

$$\forall i: D^*(E^*(x_i)) = x_i$$

码字集合 $\{c_i = E^*(x_i)\}$ 必须满足：
- $c_i \in \mathbb{R}^{512}$（code_dim=512）
- $c_i$ 两两不同（否则 decoder 无法区分不同输入）
- decoder 能从 $c_i$ 重建 $x_i$

但这些约束对 $\{c_i\}$ 的具体数值 **极度欠定**。512 维空间中的点云
$\{c_i\}$ 可以被旋转、缩放、拉伸、扭曲，而不影响重建质量——只要 decoder
做相应的逆变换即可。

### 4.3 信息论视角

从信息论看，自编码器学习的是 $x_i$ 到 $c_i$ 的一个 **编码表**
（codebook），且这个编码表只需要满足 **充分性**（sufficient statistics）——
$c_i$ 包含重建 $x_i$ 的全部信息。但编码表的 **格式**（format）完全不受约束。

这就像两种语言都能完整表达同一段话，但词汇、语法、语序可以完全不同。
不同 seed 训练出的 encoder/decoder 就像发明了不同的"语言"——各自内部
能完美沟通，但跨语言直接通信就完全失败。

### 4.4 对比监督学习

在分类任务中，输出层的每个神经元有明确语义（对应一个类别），交叉熵损失
**显式约束了输出空间的坐标系**。因此不同 seed 训练的分类模型虽然中间特征
不同，但输出层（分类概率）的语义是统一的。

自编码器没有这种外部锚点。码字空间完全是 encoder 和 decoder 之间的"私有
协议"，外部无法干预。

---

## 5. 源码级根因链条总览

将以上分析总结为从代码到现象的完整因果链：

```
随机种子（seed）
    │
    ├─→ torch.manual_seed(seed)
    │       │
    │       ├─→ nn.init.xavier_uniform_() / kaiming_uniform_()
    │       │       │
    │       │       └─→ Encoder 和 Decoder 的 ~4.6M 参数初始化不同
    │       │              │
    │       │              └─→ 第一步梯度方向不同
    │       │                     │
    │       │                     └─→ 优化轨迹完全发散
    │       │
    │       └─→ DataLoader shuffle 排列不同
    │               │
    │               └─→ 每步 batch 组成不同
    │                      │
    │                      └─→ 梯度估计不同（与初始化差异叠加）
    │
    ├─→ 优化 400 epoch (默认)
    │       │
    │       └─→ AdamW 动量/方差估计累积差异
    │              │
    │              └─→ 模型收敛到损失景观的不同极值
    │
    └─→ 最终训练好的模型
            │
            ├─→ encoder(x) = 不同坐标系的码字
            └─→ decoder 学会读该 encoder 的特定坐标系
                   │
                   └─→ 跨 seed encoder/decoder 拼接 → +15~+29 dB 崩溃
```

**根本原因总结**：MSE 重建损失只约束端到端输出，不约束中间码字的坐标系。
不同 seed 通过初始化和训练路径差异，收敛到码字空间中不同的"方言"。这不是
bug，而是自编码器的数学本质。

---

## 6. 实验数据佐证

### 6.1 同 seed 正常、跨 seed 灾难

| 配置 | NMSE |
|---|---|
| seed42 encoder + seed42 decoder | **-28.41 dB** ✓ |
| seed3407 encoder + seed3407 decoder | **-27.56 dB** ✓ |
| seed3407 encoder + seed42 decoder | **+23.85 dB** ✗ |
| seed42 encoder + seed2026 decoder | **+28.63 dB** ✗ |

差距超过 50 dB，从可用到完全崩溃。

### 6.2 后验 Adapter 的天花板

在 encoder 和 decoder 之间插入 `CodeAdapter`（`LayerNorm + Linear`，零初始化
残差连接），只训练 adapter：

| 方法 | 初始 NMSE | 最终 NMSE | 与 baseline 差距 |
|---|---|---|---|
| adapter (recon-only) | +23.85 | **-20.65** | 7.8 dB |
| adapter (code-only) | +23.85 | **-21.06** | 7.4 dB |
| adapter (learnable λ) | +23.85 | **-20.39** | 8.0 dB |

Adapter 能把模型从完全崩溃拉回来，但距离 baseline 的 -28 dB 有 ~7-8 dB 的差距。

### 6.3 LoRA 只改 token_projection 也不够

| Rank | Alpha | NMSE |
|---|---|---|
| 8 | 16 | -1.73 dB |
| 16 | 32 | -2.72 dB |
| 32 | 64 | -4.88 dB |
| 64 | 128 | -8.04 dB |

即使 rank=64，距 baseline 仍有 20 dB 差距。这说明码字空间差异不是
`token_projection` 单层线性变换能弥补的。

### 6.4 冻结目标 Decoder 训练新 Encoder 才有效

| Encoder Seed | 冻结 Seed42 Hybrid Decoder 后的 NMSE |
|---|---|
| 0 | -28.62 dB |
| 1 | -28.62 dB |
| 2026 | -28.62 dB |
| 3407 | -28.61 dB |
| 42 | -28.66 dB |
| 666 | -28.63 dB |
| 999 | -28.62 dB |

**标准差仅 0.02 dB**。当 encoder 在训练过程中直接面向目标 decoder 学习时，
无论初始化种子如何变化，都能收敛到同一个码字空间。

### 6.5 数据小结

```
独立训练同 seed:          -28 dB  ← 正常
独立训练跨 seed 拼接:      +24 dB  ← 灾难
后验 adapter:             -21 dB  ← 有限恢复，差 7 dB
后验 LoRA:                 -8 dB  ← 不够
冻结 decoder + 训练 encoder: -28.6 dB  ← 与 baseline 齐平
```

---

## 7. 对齐方法：如何缩小不同 Seed 的码字空间差异

### 7.1 方法分类框架

按作用阶段和约束强度，可以将所有对齐方法分为三大类：

```
训练前约束（Pre-training）
    ├─ 确定性初始化
    └─ 共享 bottleneck 规范化

训练中约束（In-training）
    ├─ 冻结 decoder + 训练 encoder        ← 最有效，已验证
    ├─ Teacher code 蒸馏损失              ← 需要谨慎调参
    ├─ 码字分布正则化（均值/方差/协方差）
    ├─ 对比学习锚定                       ← 复杂但灵活
    └─ 共享离散码本                       ← 最强约束但最复杂

训练后对齐（Post-training）
    ├─ 线性/仿射探针映射
    ├─ Adapter 残差翻译                   ← 当前实验
    ├─ LoRA 微调                         ← 当前实验
    └─ 解冻 encoder 微调到目标 decoder
```

### 7.2 训练前约束：确定性初始化

**思路**：如果所有训练都从相同的初始权重出发，差异来源就只剩 batch shuffle。

**实现方式**：

```python
def deterministic_init(model, fixed_seed=42):
    """所有训练共用同一个初始化种子，只让 batch shuffle 不同"""
    with torch.random.fork_rng():
        torch.manual_seed(fixed_seed)
        model._reset_parameters(model.init_strategy)
```

**效果预估**：
- 能缩小一部分差异，因为初始化相同意味着第一步梯度的差异仅来自 batch 顺序
- 但 batch shuffle 差异在 400 个 epoch 后仍然会积累为显著的参数空间差异
- **不能从根本上解决问题**，因为损失函数仍然不约束码字空间

**适用场景**：作为其他方法的基础，减小初始发散速度。

### 7.3 训练中约束（一）：冻结 Decoder + 训练 Encoder

**原理**：将 decoder 视为一个"固定的语言"，要求 encoder 学会说这种语言。

```python
# 已有实验验证的方法
decoder_target = load_decoder("seed42/transnet_hybrid/checkpoints/best_nmse.pth")
for param in decoder_target.parameters():
    param.requires_grad = False

# 只训练 encoder_new
loss = MSE(decoder_target(encoder_new(x)), x)
```

**在源码中的实现路径**：

```python
# utils/init.py, L121-L127
if args.pretrained_decoder is not None:
    decoder_state = _load_decoder_state_dict(args.pretrained_decoder)
    model.decoder.load_state_dict(decoder_state)
    for param in model.decoder.parameters():
        param.requires_grad = False
```

**为什么有效**：
- decoder 的 `token_projection`、Transformer 层、CNN refine head 的权重全部
  固定，它们定义了一个固定的 "码字 → CSI" 映射
- 不同 seed 的 encoder 被迫学习输出能让同一个 decoder 正确重建的码字
- 这相当于把码字空间的坐标系锁定为 decoder 的"输入语言"

**实验结果**：所有 seed 都收敛到 -28.6 dB（std=0.02 dB），完全解决了问题。

**局限性**：
- 需要从头训练 encoder，不能复用已有 encoder 权重
- 不同 seed 的 encoder 码字虽然能被同一个 decoder 解码，但码字本身的数值
  分布不一定完全相同（只是"同义不同形"变成了"同形同义"）

### 7.4 训练中约束（二）：Teacher Code 蒸馏

**原理**：选定一个 teacher encoder，要求所有新 encoder 的码字数值上接近
teacher 的码字。

```python
teacher = load_model("seed42/transnet_hybrid/checkpoints/best_nmse.pth")
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# 训练 encoder_new（decoder 可冻结或不冻结）
code_new = encoder_new(x)
code_teacher = teacher.encoder(x).detach()

loss = MSE(decoder(code_new), x) + lambda_t * MSE(code_new, code_teacher)
```

**关键细节**：

1. **`lambda_t` 的调度极其重要**。已有实验表明：
   - 固定 `lambda=0.1` → NMSE 劣化到 -0.027 dB（几乎无法重建）
   - 原因：过大的 code loss 压制了重建损失，encoder 学会复制 teacher code
     但 decoder 跟不上

2. **推荐的 λ 调度**：

   ```
   epoch   1 ~ 100:  lambda = 0           (先学会重建)
   epoch 100 ~ 250:  lambda 线性升到 1e-4  (逐步引入对齐)
   epoch 250 ~ 400:  lambda = 1e-4        (维持轻度对齐)
   ```

3. **与冻结 decoder 结合效果更好**：先冻结 decoder 保证重建质量，再用
   teacher code loss 进一步统一码字数值。

### 7.5 训练中约束（三）：码字分布正则化

**原理**：不指定"码字应该等于什么值"，而是约束"码字应该服从什么分布"。

```python
code = encoder(x)  # (B, code_dim)

# 均值约束：码字 batch 均值接近 0
mean_loss = code.mean(dim=0).pow(2).mean()

# 方差约束：码字 batch 方差接近 1
var_loss = (code.var(dim=0) - 1).pow(2).mean()

# 协方差约束（可选）：码字维度间去相关
# 计算成本较高，code_dim=512 时需要 512×512 协方差矩阵
code_centered = code - code.mean(dim=0, keepdim=True)
cov = (code_centered.T @ code_centered) / (B - 1)
cov_loss = (cov - torch.eye(code_dim).to(cov.device)).pow(2).mean()

loss = recon_loss + alpha * mean_loss + beta * var_loss + gamma * cov_loss
```

**效果预估**：
- 能消除码字的尺度漂移和均值漂移
- 去相关可以减少维度间的冗余纠缠
- **但不能保证码字的维度语义一致**——两个 seed 可以都输出零均值、单位方差
  的码字，但第 1 维的含义可能完全不同

**适用场景**：
- 作为辅助手段与 teacher code loss 或冻结 decoder 配合
- 独立使用时只能缩小分布差异，不能保证互换性

### 7.6 训练中约束（四）：对比学习锚定

**原理**：用样本级对比学习（InfoNCE）要求同一个 CSI 样本在不同 encoder 下
产生相近的码字，不同 CSI 样本的码字远离。

```python
# 需要同时有 teacher encoder 和 new encoder
code_new = encoder_new(x)          # (B, 512)
code_teacher = teacher.encoder(x)  # (B, 512)

# 计算相似度矩阵
sim = F.cosine_similarity(code_new.unsqueeze(1),
                          code_teacher.unsqueeze(0), dim=2) / tau
# InfoNCE loss
labels = torch.arange(B).to(sim.device)
contrastive_loss = F.cross_entropy(sim, labels)

loss = recon_loss + lambda * contrastive_loss
```

**优点**：
- 不强求码字数值完全相同，只要求保持样本间的邻域结构
- 对旋转/缩放等等价变换更鲁棒
- 理论上可以在不牺牲重建质量的前提下对齐结构

**缺点**：
- 实现复杂，需要调 temperature、batch size
- 对齐的是"结构"而不是"坐标"，decoder 可能仍需微调
- 训练成本更高

### 7.7 训练中约束（五）：共享离散码本（VQ-VAE 风格）

**原理**：强制所有 encoder 的输出映射到共享码本中的离散向量。

```python
class SharedCodebook(nn.Module):
    def __init__(self, num_codes=1024, code_dim=512):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)

    def forward(self, z):
        # z: (B, code_dim)，encoder 的连续输出
        distances = torch.cdist(z.unsqueeze(0),
                                self.codebook.weight.unsqueeze(0))
        indices = distances.argmin(dim=-1)
        quantized = self.codebook(indices)
        # Straight-through estimator
        return z + (quantized - z).detach()
```

**优点**：
- 码字空间被严格限制在有限个离散向量上
- 不同 seed 的 encoder 都只能使用同一套码本
- 天然保证互换性

**缺点**：
- 实现复杂度高（commitment loss、码本利用率、EMA 更新）
- 可能牺牲重建精度（量化误差）
- 需要大量实验确定码本大小
- 与当前 MSE 训练范式差距较大

### 7.8 训练后对齐：线性/仿射探针

**原理**：在训练完成后，学习一个线性（或仿射）映射，将 seed_A 的码字空间
映射到 seed_B 的码字空间。

```python
# 导出同一批数据的码字
code_A = encoder_A(X)  # (N, 512)
code_B = encoder_B(X)  # (N, 512)

# 线性最小二乘求解
# code_B ≈ code_A @ W + b
W, b = linear_regression(code_A, code_B)

# 推理时
code_adapted = encoder_A(x) @ W + b
output = decoder_B(code_adapted)
```

**效果取决于码字空间差异的性质**：
- 如果差异主要是线性（旋转/缩放/置换）：线性探针能恢复大部分性能
- 如果差异包含非线性扭曲：线性探针天花板很低
- **这是一个重要的诊断工具**：通过比较线性探针和 MLP 探针的结果，
  可以判断码字空间差异的主要性质

---

## 8. Adapter 路线可行性分析与改进

### 8.1 当前 CodeAdapter 的局限性

当前的 `CodeAdapter` 结构非常简单：

```python
# models/adapters/code_adapter.py
class CodeAdapter(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.norm = nn.LayerNorm(code_dim)
        self.proj = nn.Linear(code_dim, code_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, code):
        return code + self.proj(self.norm(code))
```

它是一个 **零初始化的残差线性变换**。能力分析：

| 能力 | 支持？ | 说明 |
|---|---|---|
| 线性旋转 | 部分 | `I + W` 可以近似正交变换，但零初始化开始困难 |
| 缩放 | ✓ | 对角化的 `W` 可以缩放 |
| 维度置换 | 部分 | 需要 `W` 学成置换矩阵，困难 |
| 均值偏移 | ✓ | bias 可以偏移 |
| 非线性扭曲 | ✗ | 单个 Linear 没有非线性 |

**关键瓶颈**：如果两个 seed 的码字空间差异包含 **非线性** 成分（高度可能），
单层线性 adapter 无法弥补。

### 8.2 增强 Adapter 的方案

#### 方案 A：多层 MLP Adapter

```python
class MLPAdapter(nn.Module):
    def __init__(self, code_dim, hidden_dim=1024, num_layers=2):
        super().__init__()
        layers = []
        in_dim = code_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, code_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(code_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        # 最后一层零初始化，保持残差连接初始为恒等
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, code):
        return code + self.net(self.norm(code))
```

**优势**：非线性映射能力，可以处理旋转 + 缩放 + 非线性扭曲的组合。
**风险**：过拟合，需要正则化。

#### 方案 B：分解式 Adapter（先对齐分布，再做精细映射）

```python
class TwoStageAdapter(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        # Stage 1: 分布对齐（可学习的 LayerNorm 参数）
        self.dist_align = nn.LayerNorm(code_dim)
        # Stage 2: 坐标变换（线性映射 + 残差）
        self.coord_transform = nn.Linear(code_dim, code_dim, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.eye_(self.coord_transform.weight)
        nn.init.zeros_(self.coord_transform.bias)

    def forward(self, code):
        aligned = self.dist_align(code)
        return self.coord_transform(aligned)
```

**优势**：分阶段处理——先用 LayerNorm 消除均值/方差漂移，再用线性变换
修正坐标旋转。注意力初始化为单位矩阵而不是零矩阵。

#### 方案 C：条件 Adapter（per-encoder 参数）

```python
class ConditionalAdapter(nn.Module):
    def __init__(self, code_dim, num_encoders):
        super().__init__()
        # 每个 encoder 有独立的 adapter 参数
        self.adapters = nn.ModuleDict({
            f"enc_{i}": CodeAdapter(code_dim)
            for i in range(num_encoders)
        })

    def forward(self, code, encoder_id):
        return self.adapters[f"enc_{encoder_id}"](code)
```

**优势**：每个 encoder 有独立的翻译模块，比共享 adapter 更灵活。
**代价**：参数量线性增长，部署时需要知道码字来源。

### 8.3 Adapter 训练策略改进

当前 adapter 训练的问题不仅是模型容量，还有训练策略：

#### 改进 1：先用线性探针热启动

```python
# Step 1: 导出码字对
code_src = encoder_src(X)
code_tgt = encoder_tgt(X)

# Step 2: 线性最小二乘求解
W, b = torch.linalg.lstsq(
    torch.cat([code_src, torch.ones(N,1)], dim=1),
    code_tgt
).solution

# Step 3: 用 (W, b) 初始化 adapter 的线性层
adapter.proj.weight.data = W[:512, :]
adapter.proj.bias.data = b[:512]
```

**优势**：adapter 从一个已经粗对齐的状态开始优化，而不是从零开始。

#### 改进 2：渐进式训练

```python
# Phase 1: 只训练 adapter，冻结 encoder 和 decoder (当前做法)
# Phase 2: 解冻 encoder，联合微调 adapter + encoder
# Phase 3: 可选地解冻 decoder 的部分层

for phase, (unfreeze_enc, unfreeze_dec, epochs) in enumerate([
    (False, False, 100),  # adapter only
    (True,  False, 200),  # adapter + encoder
    (True,  True,  100),  # adapter + encoder + decoder (partial)
]):
    set_requires_grad(encoder, unfreeze_enc)
    set_requires_grad(decoder, unfreeze_dec)
    train(model, epochs)
```

#### 改进 3：多任务码字空间正则化

在 adapter 训练中同时加入码字空间约束：

```python
code_adapted = adapter(encoder_src(x))
code_ref = encoder_ref(x).detach()

loss = MSE(decoder_ref(code_adapted), x)     # 重建损失
     + alpha * MSE(code_adapted, code_ref)    # 码字对齐
     + beta * distribution_loss(code_adapted) # 分布正则化
```

### 8.4 让 Adapter 达到同 Seed 水平的核心条件

基于以上分析，adapter 要达到同 seed 水平（-28 dB），需要满足以下条件之一：

1. **Adapter 容量足够大**（能完全建模码字空间的非线性映射）
   + 足够多的训练数据和 epoch
   + 不过拟合

2. **码字空间差异主要是线性的**（简单 adapter 就够用）
   + 需要通过线性探针实验验证
   + 如果差异主要是非线性的，需要增加 adapter 复杂度

3. **Adapter + encoder 联合微调**（相当于"用已有 encoder 权重热启动 +
   冻结 decoder 训练"）
   + 这本质上回到了"冻结 decoder 训练 encoder"方案
   + 但利用了已有 encoder 权重作为更好的初始化

**最有可能成功的路线是条件 3**——它同时利用了已有 encoder 的知识和目标
decoder 的约束。

---

## 9. 总结与建议

### 9.1 核心结论

| 层次 | 原因 | 能否消除？ |
|---|---|---|
| 表层 | 不同 seed → 不同初始化 + 不同 batch 顺序 | 可以固定初始化种子，但不解决根本问题 |
| 中层 | 非凸损失景观 + 等价变换对称性 → 不同极值 | 需要显式打破对称性 |
| **深层** | **MSE 损失不约束码字空间坐标系** | **只能通过额外约束解决** |

### 9.2 方法推荐优先级

| 优先级 | 方法 | 预期效果 | 实施难度 | 状态 |
|---|---|---|---|---|
| **1** | 冻结 decoder + 训练 encoder | -28.6 dB | 低 | ✅ 已验证 |
| **2** | 冻结 decoder + 加载已有 encoder 微调 | 预期 -28 dB | 低 | ⏳ 待验证 |
| **3** | 线性/MLP 探针诊断码字差异性质 | 诊断工具 | 中 | ⏳ 待做 |
| **4** | 多层 MLP adapter + 分阶段训练 | 预期 -24~-26 dB | 中 | ⏳ 待做 |
| **5** | Teacher code 蒸馏 (warmup λ) | 预期 -27~-28 dB | 中 | ⏳ 待做 |
| **6** | 码字分布正则化 | 缩小分布差异 | 低 | ⏳ 辅助手段 |
| **7** | 对比学习锚定 | 未知 | 高 | ⏳ 探索性 |
| **8** | 共享离散码本 | 最强保证 | 极高 | ⏳ 长期方向 |

### 9.3 对 Adapter 研究方向的具体建议

如果你的研究目标是"在不同 seed 训练出来的 encoder/decoder 中间加个 adapter，
使其达到同一个 seed 训练出来的水平"，那么最可能的成功路径是：

```
Step 1: 线性探针实验 → 判断码字差异是否主要为线性
  ├── 如果线性探针能到 -25 dB 以上：
  │     → 增强 adapter 为 2-3 层 MLP，预期可达 -27~-28 dB
  │
  └── 如果线性探针只能到 -15~-20 dB：
        → 差异包含大量非线性，纯 adapter 难以弥补
        → 需要解冻 encoder 联合微调（回到"冻结 decoder 训练 encoder"）
        → 或使用训练中约束（teacher code + frozen decoder）

Step 2: 用探针结果初始化 adapter 权重
Step 3: 分阶段训练（先 adapter-only → 再解冻 encoder）
Step 4: 加入轻量码字分布正则化辅助
```

### 9.4 一句话总结

> **不同 seed 的码字空间差异不是随机噪声，而是自编码器优化问题的数学本质——
> MSE 损失定义了无穷多等价但互不兼容的码字坐标系。要让不同 seed 的 encoder
> 输出相互兼容，必须在训练中引入额外约束来消除这种坐标系自由度。后验 adapter
> 的本质是在不改变 encoder/decoder 的前提下做码字空间翻译，其天花板取决于
> 码字差异的线性/非线性程度——首先做线性探针诊断，然后决定 adapter 复杂度
> 或是否需要回到训练中约束。**
