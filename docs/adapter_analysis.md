# Adapter 结构改进分析：跨 Encoder 码字空间对齐

## 1. 现有实验回顾

### 1.1 实验设置

`exps/COST2100/in/adapter/` 下有三组 adapter 实验：

| Adapter 类型 | 种子数 | Encoder（冻结） | Decoder（冻结） | 关键配置 |
|---|---|---|---|---|
| **mlp** (残差 MLP) | 0,42,520,796,1024,2026,3407 | transnet (seedX) | hybrid (seed42) | hidden_dim=4×code_dim=2048 |
| **mlp_direct** (无残差) | 同上 | 同上 | 同上 | 同上 |
| **transformer** | 3407 | 同上 | 同上 | d_model=64, nhead=2, hidden_dim=2048 |

**关键发现：Decoder 始终来自 seed42 的 checkpoint，Encoder 来自各自 seed 的 checkpoint。**

### 1.2 核心数据

| 配置 | 训练前 NMSE (dB) | 训练后 NMSE (dB) | 联合训练基线 (dB) |
|---|---|---|---|
| seed0 encoder + seed42 decoder (mlp) | **+29.23** | -20.51 | -26.60 |
| seed0 encoder + seed42 decoder (mlp_direct) | **+26.80** | -20.70 | -26.60 |
| seed3407 encoder + seed42 decoder (transformer) | **+23.85** | -20.54 | -27.09 |
| seed42 encoder + seed42 decoder (mlp) | ~0 (identity) | **-28.07** | -28.07 |

### 1.3 关键结论

1. **训练前 NMSE 为正（+23~+29 dB）**：说明即使同架构（transnet→hybrid）、仅因不同随机种子训练，编码器输出的码字空间也几乎正交——解码器根本无法理解来自“陌生”编码器的码字。

2. **Adapter 可以将 NMSE 从 +29 dB 拉回 -20.5 dB**：2 层 MLP（512→2048→512 + 残差）在 400 epoch 后能把两个几乎正交的空间桥接起来，效果显著。

3. **但仍有 6~8 dB 的差距**：与联合训练（-26~-28 dB）相比，仅靠码字空间内的 adapter 无法完全弥补 encoder/decoder 内部表征的协同适配损失。

4. **Seed42 → Seed42（同源）几乎无损**：当 encoder 和 decoder 来自同一次训练时，adapter 只需学习恒等映射，最终 NMSE 与联合训练持平。

5. **三种 adapter 最终性能接近**：MLP 残差、MLP 直接、Transformer 在充分训练后都收敛到 -20.5 dB 左右，说明瓶颈不在 adapter 的表达能力，而在于**仅修改码字向量的信息瓶颈**。

---

## 2. 问题本质分析

### 2.1 码字空间正交的成因

不同 encoder（无论是不同架构如 Swin/ConvNeXt/TransNet，还是同架构不同种子）将同一个 CSI 信号映射到 R⁵¹² 中的不同区域：

```
Encoder_A: CSI → code_A ∈ R⁵¹²
Encoder_B: CSI → code_B ∈ R⁵¹²
```

尽管 code_A 和 code_B 都编码了同一个 CSI 的“信息”，但它们在 R⁵¹² 中可能处于几乎正交的子空间。这是因为：

- **初始化不同**：不同种子 → 不同初始权重 → 收敛到不同的局部最优
- **归纳偏好不同**：CNN 偏好局部特征，Transformer 偏好全局交互，Swin 偏好窗口局部自注意力
- **Decoder 协同适配**：训练时 encoder 和 decoder 相互适应，形成“密码本”约定——decoder 的第一层 `fc_decoder` / `token_projection` 学会了特定于该 encoder 的基

### 2.2 为什么简单的 Adapter 不够

当前 adapter 做的是：

```
y = x + MLP(LayerNorm(x))     # MLP 残差
y = MLP(LayerNorm(x))          # MLP 直接
y = x + Transformer(LN(x))     # Transformer 残差
```

这本质上是用一个神经网络在 R⁵¹² 中学习一个映射。问题在于：

1. **起点太差**：残差 adapter 从恒等映射出发，但在正交空间中恒等映射给出 +29 dB NMSE。梯度需要穿过一个“高损失高原”才能找到正确的变换方向。

2. **线性变换与非线性修正未解耦**：码字空间的对齐首先需要一个大的线性变换（旋转/缩放/基底变换），然后才是架构差异带来的非线性修正。当前设计把两者混在一个 MLP 中。

3. **信息瓶颈**：adapter 在 512 维空间内操作，而 decoder 内部有大量参数（fc_decoder: 2048×512, token_mixer: 多层 Transformer 等）。仅修改 512 维输入无法重塑 decoder 内部的表征分布。

4. **缺乏结构化先验**：没有任何约束来利用“所有 encoder 都编码同一信号”这一内在联系。

---

## 3. Adapter 结构改进方案

以下方案按实现难度和预期收益排序，均不改动 encoder/decoder 权重。

### 方案 1：线性 + 非线性分解 Adapter（推荐优先尝试）

**核心思路**：将变换分解为一个显式的线性映射（学习基底变换/旋转）和一个小的非线性残差。

```python
class DecomposedAdapter(nn.Module):
    def __init__(self, adapter_dim, adapter_hidden_dim=None):
        super().__init__()
        if adapter_hidden_dim is None:
            adapter_hidden_dim = 4 * adapter_dim

        # 线性变换：学习码字空间的旋转/缩放
        self.linear = nn.Linear(adapter_dim, adapter_dim, bias=False)

        # 非线性修正：处理架构差异
        self.nonlinear = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, adapter_dim),
        )

        # 可学习的混合系数
        self.beta = nn.Parameter(torch.tensor(0.1))   # 非线性贡献从小开始

    def reset_parameters(self):
        # 线性层初始化为正交矩阵（保持范数，避免梯度消失/爆炸）
        nn.init.orthogonal_(self.linear.weight)
        # 非线性部分零初始化（初始为恒等偏置）
        nn.init.kaiming_uniform_(self.nonlinear[1].weight, a=0.3,
                                 nonlinearity='leaky_relu')
        nn.init.zeros_(self.nonlinear[1].bias)
        nn.init.zeros_(self.nonlinear[3].weight)
        nn.init.zeros_(self.nonlinear[3].bias)

    def forward(self, x):
        x_linear = self.linear(x)
        x_nonlinear = self.nonlinear(x_linear)
        return x_linear + self.beta * x_nonlinear
```

**为什么有效**：
- 正交初始化给线性层一个好的起点（接近正交矩阵保持范数，避免梯度消失/爆炸）
- 非线性部分从小开始（beta≈0.1），让模型先学会大致的线性对齐，再逐步加入非线性修正
- 显式分离旋转和扭曲，训练更稳定

**变体**：可以将 `alpha` 参数也用于输入支路：`y = alpha * x_linear + (1-alpha) * x_nonlinear`，或者完全去掉残差只用 `x_linear + beta * x_nonlinear`。

---

### 方案 2：更深残差 Adapter（ResNet-in-Code-Space）

**核心思路**：将单个 2 层 MLP 扩展为多个残差块，增加表达深度。

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # 零初始化最后一层 → 每个 block 从恒等开始
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        return x + self.mlp(self.norm(x))

class DeepResidualAdapter(nn.Module):
    def __init__(self, adapter_dim, hidden_dim=None, num_blocks=4):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * adapter_dim
        self.blocks = nn.Sequential(*[
            ResidualBlock(adapter_dim, hidden_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)
```

**为什么有效**：
- 更多层 = 更强的非线性表达能力
- 每个 block 从恒等开始，训练稳定
- 可以逐步学习越来越复杂的变换
- 实现简单，改动最小

**风险**：过深可能导致过拟合（512 维空间，100K 样本）。

---

### 方案 3：带统计对齐的 Adapter

**核心思路**：在变换前先对 encoder 输出做统计归一化，使其匹配 decoder 期望的分布。

```python
class StatisticsAlignedAdapter(nn.Module):
    def __init__(self, adapter_dim, adapter_hidden_dim=None):
        super().__init__()
        if adapter_hidden_dim is None:
            adapter_hidden_dim = 4 * adapter_dim

        # 可学习的仿射参数：将 encoder 输出分布对齐到 decoder 期望
        self.pre_norm = nn.LayerNorm(adapter_dim)
        self.shift = nn.Parameter(torch.zeros(adapter_dim))
        self.scale = nn.Parameter(torch.ones(adapter_dim))

        # 非线性变换
        self.mlp = nn.Sequential(
            nn.Linear(adapter_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, adapter_dim),
        )
        # 零初始化残差
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        # Step 1: 统计对齐
        x_norm = self.pre_norm(x)
        x_aligned = self.scale * x_norm + self.shift

        # Step 2: 非线性修正
        delta = self.mlp(x_aligned)
        return x_aligned + delta
```

**为什么有效**：
- 不同 encoder 输出的均值和方差差异可能很大
- 通过可学习的 shift/scale 先将分布大致对齐
- 然后用残差 MLP 做精细修正

**进阶**：可以维护 running_mean/running_var（类似 BatchNorm），在训练时用 batch 统计量，推理时用全局统计量。

---

### 方案 4：多尺度码字 Adapter

**核心思路**：在多个分辨率上处理码字，类似图像处理中的多尺度策略。

```python
class MultiScaleAdapter(nn.Module):
    def __init__(self, adapter_dim, adapter_hidden_dim=None, num_scales=3):
        super().__init__()
        if adapter_hidden_dim is None:
            adapter_hidden_dim = 4 * adapter_dim

        self.scales = nn.ModuleList()
        for i in range(num_scales):
            # 不同尺度的降采样/升采样
            scale_dim = max(adapter_dim // (2 ** i), 32)
            self.scales.append(nn.Sequential(
                nn.Linear(adapter_dim, scale_dim),
                nn.GELU(),
                nn.Linear(scale_dim, adapter_dim),
            ))
            # 零初始化最后一层
            nn.init.zeros_(self.scales[-1][2].weight)
            nn.init.zeros_(self.scales[-1][2].bias)

    def forward(self, x):
        out = x
        for scale_module in self.scales:
            out = out + scale_module(out)
        return out
```

**为什么有效**：
- 不同尺度捕获不同粒度的码字结构
- 粗尺度（小维度）学习全局对齐
- 细尺度（大维度）学习局部调整

---

### 方案 5：跨码字注意力 Adapter（增强 Transformer）

**核心思路**：在现有 TransformerAdapter 基础上加深、加宽，并引入位置编码。

```python
class EnhancedTransformerAdapter(nn.Module):
    def __init__(self, adapter_dim, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=1024):
        super().__init__()
        assert adapter_dim % d_model == 0
        self.num_tokens = adapter_dim // d_model
        self.d_model = d_model

        # 可学习的 token 位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, d_model) * 0.02)

        self.norm = nn.LayerNorm(adapter_dim)
        self.tokenize = nn.Linear(adapter_dim, adapter_dim, bias=False)

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer = TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model))

        self.out_proj = nn.Linear(adapter_dim, adapter_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        B = x.size(0)
        t = self.tokenize(self.norm(x))
        t = t.view(B, self.num_tokens, self.d_model)
        t = t + self.pos_embed  # 加入位置信息
        t = self.transformer(t)
        t = t.reshape(B, -1)
        return x + self.out_proj(t)
```

**为什么有效**：
- 自注意力可以捕获码字内部的长程依赖
- 位置编码帮助 token 区分不同位置
- 更深的 Transformer 提供更强的表达能力

---

### 方案 6：双阶段渐进训练 Adapter

**核心思路**：不是改结构，而是改训练策略。

**阶段 1（线性对齐，epoch 0~100）**：
- 只训练一个线性层 `Linear(code_dim, code_dim)`，初始化为随机正交矩阵
- 冻结非线性部分
- 训练到 NMSE 不再明显下降

**阶段 2（非线性精修，epoch 100~400）**：
- 解冻整个 adapter
- 用较小的学习率继续训练

```python
class ProgressiveAdapter(nn.Module):
    def __init__(self, adapter_dim, adapter_hidden_dim=None):
        super().__init__()
        if adapter_hidden_dim is None:
            adapter_hidden_dim = 4 * adapter_dim

        # Stage 1: 线性变换
        self.linear = nn.Linear(adapter_dim, adapter_dim, bias=False)
        nn.init.orthogonal_(self.linear.weight)

        # Stage 2: 非线性修正（初始冻结）
        self.nonlinear = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, adapter_dim),
        )
        nn.init.zeros_(self.nonlinear[3].weight)
        nn.init.zeros_(self.nonlinear[3].bias)

    def forward(self, x):
        x = self.linear(x)
        return x + self.nonlinear(x)

    def stage1_mode(self):
        """冻结非线性部分，只训练线性层"""
        for p in self.nonlinear.parameters():
            p.requires_grad = False

    def stage2_mode(self):
        """解冻全部参数"""
        for p in self.nonlinear.parameters():
            p.requires_grad = True
```

在 `main.py` 的训练循环中，epoch 0~100 调用 `model.code_adapter.stage1_mode()`，epoch 100~400 调用 `model.code_adapter.stage2_mode()`。

---

## 4. 方案对比与推荐路径

| 方案 | 实现复杂度 | 预期增益 | 训练稳定性 | 推荐优先级 |
|---|---|---|---|---|
| **方案 1：线性+非线性分解** | 低 | ★★★★ | ★★★★★ | 🥇 第一优先 |
| **方案 2：深层残差** | 低 | ★★★ | ★★★★ | 🥈 第二优先 |
| **方案 3：统计对齐** | 中 | ★★★ | ★★★★ | 🥈 第二优先 |
| **方案 4：多尺度** | 中 | ★★ | ★★★ | 🥉 探索性 |
| **方案 5：增强 Transformer** | 中 | ★★★ | ★★★ | 🥉 探索性 |
| **方案 6：双阶段训练** | 低 | ★★★ | ★★★★★ | 🥇 配合方案1 |

**推荐实施路径**：

1. **首先实现方案 1（线性+非线性分解）+ 方案 6（双阶段训练）的组合**。这是改动最小、最可能有效的方案——显式学习线性对齐 + 渐进训练。

2. **同时实现方案 2（深层残差）** 作为备选，代码改动很小，可以快速验证更多层是否有帮助。

3. 如果上述方案仍有 3dB 以上的 gap，说明瓶颈不在码字空间而在 decoder 内部，此时需要考虑：
   - 在 decoder 的中间层插入轻量 adapter（类似 LoRA，但这会“动”decoder）
   - 或者接受这个 gap 作为不改 encoder/decoder 的固有限制

---

## 5. 实现注意事项

### 5.1 参数初始化策略

对于正交码字空间问题，初始化至关重要：

- **线性层**：使用 `nn.init.orthogonal_()` 获得一个保持范数的旋转矩阵，比随机初始化和恒等初始化都更好
- **非线性残差**：最后一层零初始化，确保初始时非线性部分贡献为 0
- **LayerNorm**：使用 `elementwise_affine=True`（默认），让模型学习最优的归一化参数

### 5.2 学习率与优化

- adapter 训练时只有少量参数（相比 encoder+decoder），可以使用更高的学习率
- 推荐初始 lr = 1e-3 ~ 5e-4（当前 2e-4 可能偏保守）
- 对于方案 6 的阶段 2，将 lr 降为原来的 1/5 ~ 1/10

### 5.3 评估指标

除了 NMSE，建议监控以下指标来理解 adapter 的行为：

- **码字余弦相似度**：adapter 输出与“native decoder”期望码字的余弦相似度
- **码字范数比**：`||adapted_code|| / ||original_code||`
- **线性层权重奇异值分布**：判断线性变换是否接近正交
- **非线性残差占比**：`||delta|| / ||x||`，观察非线性的贡献

### 5.4 实验对照

每个新 adapter 方案应至少跑 3 个 seed（如 0, 42, 2026），并与以下基线对比：

- **下界**：不做 adapter，直接拼接 encoder_A + decoder_B（NMSE ~ +29 dB）
- **当前最佳**：MLP adapter（NMSE ~ -20.5 ~ -24.9 dB）
- **上界**：联合训练同源 encoder+decoder（NMSE ~ -26 ~ -28 dB）

---

## 6. 更深层的思考：是否应该接受这个 Gap？

从信息论角度看，encoder 将 CSI 张量（2048 维）压缩为码字（512 维）。这 512 维中：

- 一部分维度编码了“信号内容”（与 CSI 重建相关）
- 一部分维度编码了“encoder 特征”（与该 encoder 的训练轨迹相关）
- decoder 学会了解码特定的“encoder 特征”

Adapter 在 512 维空间内只能重新组合这些维度，无法凭空创造丢失的信息。如果 encoder_A 和 encoder_B 都把关键信息编码在它们各自不同的 200 维子空间中，adapter 最多只能做一个基底变换——而这种情况下线性变换应该就够了。

**如果线性变换 + 小非线性不足以弥合 gap，那么真正的问题在于：decoder 内部的前几层（如 `token_projection`、`semantic_projector`）已经过拟合到特定 encoder 的码字分布。** 这种情况下，可能需要在 decoder 的前几层插入小型 adapter（如 1×1 卷积或轻量 MLP），而不是仅在码字入口处做变换。

但这已经超出了“不动 encoder/decoder”的约束。作为一种折中，可以考虑 **Decoder-Aware Adapter**：

```python
class DecoderAwareAdapter(nn.Module):
    """
    在 decoder 的前几层插入可训练的轻量变换，但不修改原始权重。
    
    策略：对于 HybridDecoder，在 semantic_projector 和 token_projection 
    之后各插入一个低秩 adapter（类似 LoRA 但只加不乘）。
    """
    def __init__(self, code_dim, input_dim, rank=16):
        super().__init__()
        # 码字空间的 adapter（原有）
        self.code_adapter = DecomposedAdapter(code_dim)
        
        # Decoder 入口处的低秩修正（新增）
        # semantic_projector 之后的修正
        self.semantic_lora_A = nn.Linear(code_dim, rank, bias=False)
        self.semantic_lora_B = nn.Linear(rank, code_dim, bias=False)
        nn.init.zeros_(self.semantic_lora_B.weight)
        
        # token_projection 之后的修正  
        self.token_lora_A = nn.Linear(input_dim, rank, bias=False)
        self.token_lora_B = nn.Linear(rank, input_dim, bias=False)
        nn.init.zeros_(self.token_lora_B.weight)
```

这需要修改 `UniversalCSI.py` 的 forward 逻辑，在 decoder 内部插入 adapter 的调用——但 decoder 的原始权重仍然不变，只是在其输出上加了低秩修正项。这比完全“不动 decoder”稍微多了一点侵入性，但仍然保持了原始权重的完整性。
