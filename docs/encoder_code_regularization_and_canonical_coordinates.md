# Encoder 码字正则与公共坐标系分析

## 背景

当前 CSI 自编码训练形式为：

```text
x -> encoder -> z -> decoder -> x_hat
```

不同 seed 下训练出的 encoder/decoder 虽然自配对重建 NMSE 都不错，但跨 seed 互换时性能明显退化。这说明不同 seed 学到的码字 `z` 不在同一个坐标系中。即使两个模型都能很好重建，`encoder_seedA` 输出的第 `i` 维 code 不一定对应 `decoder_seedB` 理解的第 `i` 维语义。

## 为什么会出现坐标系旋转

自编码器的 latent 坐标本身不可辨识。对任意可逆矩阵 `A`，都可以构造：

```text
E'(x) = A E(x)
D'(z') = D(A^-1 z')
```

此时：

```text
D'(E'(x)) = D(E(x))
```

重建结果完全一样，但码字坐标系已经被旋转、缩放或线性混合。只要训练目标主要是：

```text
MSE(D(E(x)), x)
```

模型就没有理由选择某一个唯一坐标系。

因此，不引入任何锚点时，不能保证不同 seed 的 encoder 自行编码到同一个坐标系。

## 不能使用其他 seed encoder/decoder 时的约束思路

如果不想引入其他 seed 的 encoder/decoder 信息，可以引入“公共锚点”。这个锚点不是某个模型，而是公共结构、公共数据统计或数学基底。

可选方向包括：

```text
固定或共享的 projection head
PCA/DCT/Fourier 等公共 basis
固定 codebook
code 分布正则
非负/稀疏结构
orthogonal/normalized encoder head
```

但需要注意：单纯 code 正则只能减轻坐标漂移，通常不能完全定义唯一坐标系。

## 常见 Code 正则手段

### 1. 尺度/能量约束

形式：

```text
||z||^2
mean(||z||^2) -> target
var(z_i) -> target_i
```

作用：

```text
防止码字整体尺度漂移
稳定不同 batch、不同 seed 的 code 能量
```

局限：

```text
如果所有维度 target 相同，任意正交旋转后能量不变
无法保证第 i 维 code 的语义一致
```

### 2. 均值/方差/协方差约束

形式：

```text
mean(z) -> 0
cov(z) -> I
cov(z) -> diag(target_var)
offdiag(cov) -> 0
```

作用：

```text
让 code 去均值
减少维度相关性
控制每个维度的能量
```

如果使用非均匀 `target_var`，例如固定递减序列：

```text
target_var_i = exp(-i / tau)
```

可以给每个 code 维度一个固定“能量身份”，从而减少维度置换和旋转自由度。

局限：

```text
cov(z)=I 时，zQ 对任意正交矩阵 Q 仍满足 cov=I
非均匀 target_var 只能约束二阶统计，不能保证逐样本语义方向一致
decoder 仍可通过自身参数补偿 encoder 的坐标变换
```

### 3. 稀疏/非负/熵约束

形式：

```text
L1(z)
z >= 0
top-k sparsity
entropy penalty
```

作用：

```text
破坏很多旋转自由度
让 code 更像公共字典系数
```

原因是任意旋转后的 dense 实值 code 通常不再稀疏，也不再非负。

局限：

```text
可能牺牲自重建 NMSE
仍可能存在维度置换、局部混合、尺度补偿
不能单独保证跨 seed 坐标完全一致
```

### 4. 结构化先验

形式：

```text
orthogonal encoder head
normalized code
ordered variance
fixed basis coefficient
VQ / fixed codebook
```

作用：

```text
比普通 loss 更强
从架构层面减少坐标自由度
让 code 更接近某种公共规范下的系数
```

局限：

```text
如果 decoder 仍完全自由训练，它可以学习反向变换，绕开部分约束
强结构可能降低单模型重建性能
```

## 为什么 Code 正则仍不能完全约束码字坐标系

核心原因是正则大多约束的是“分布形状”，不是“语义坐标”。

例如：

```text
mean=0：旋转后仍为 0
L2 norm：正交旋转不变
cov=I：正交旋转不变
offdiag cov=0：在等方差情况下旋转后仍可能成立
```

即使使用非均匀方差排序，也只是减少旋转空间，而不是对每个样本定义唯一坐标。

只要 decoder 是自由训练的，它就可以适配 encoder 输出的私有坐标系：

```text
encoder 学一个变换 A
decoder 学 A^-1
整体 recon loss 不变
```

因此，code 正则通常只能“偏置模型倾向某类坐标”，不能严格保证不同 seed 得到相同坐标。

## 从 Encoder 下手的更合理方向

相比冻结随机 decoder 展开层，更合理的是约束 encoder 的输出头和 code 分布。

### 1. Encoder canonical head

将 encoder 结构改为：

```text
encoder backbone -> canonical head -> code
```

canonical head 可以包含：

```text
LayerNorm
orthogonal / normalized linear head
ordered scale
optional sparsity / non-negative activation
```

目标是让最后输出的 code 维度具有稳定身份。

### 2. Orthogonal / normalized encoder head

如果 encoder 最后一层是：

```text
features -> encoder.fc -> code
```

可对 `encoder.fc.weight` 加约束：

```text
W W^T -> I
```

作用：

```text
减少 code 维度间混合
避免某些维度尺度过大
让输出头更接近标准正交坐标
```

局限：

```text
正交不等于语义唯一
decoder 仍能适配旋转后的正交坐标
```

### 3. Ordered scale / target variance

在 code 输出端定义固定维度能量排序：

```text
target_var_i = exp(-i / tau)
```

或：

```text
code_i *= fixed_scale_i
```

作用：

```text
第 0 维、第 1 维、第 100 维有不同统计身份
减少维度置换和任意旋转
```

局限：

```text
只约束统计，不约束样本级语义
tau 过小会导致后面维度几乎不用
tau 过大则接近等方差，锚定能力弱
```

### 4. 稀疏/非负 code

可以让 code 更像公共字典系数：

```text
z = softplus(raw_z)
loss += lambda_l1 * z.mean()
```

或只加：

```text
loss += lambda_l1 * mean(abs(z))
```

优点：

```text
抑制任意旋转
让 code 更稳定、更可解释
```

风险：

```text
可能明显损伤重建能力
可能造成死维度或容量不足
```

## 关于固定随机 fc_decoder 的判断

从现有实验记录看，`fc_decoder` 或 decoder 的第一层展开映射非常关键。

正常 `transnet_transnet` 自配对训练结果大约为：

```text
seed42:   -28.126 dB
seed2026: -28.180 dB
seed3407: -26.868 dB
```

冻结 seed42 decoder body，只放开 `fc_decoder` 去适配其他 seed encoder 时：

```text
seed2026 -> seed42 decoder body: -22.561 dB
seed3407 -> seed42 decoder body: -22.083 dB
seed3407 lr=1e-3:               -22.211 dB
```

这说明即使 `fc_decoder` 可训练，在 decoder body 固定时也只能恢复到约 `-22 dB`，离完整端到端训练有明显差距。

因此，固定随机初始化的 `fc_decoder` 大概率不可行。它虽然能作为公共锚点，但会直接锁死最重要的重建展开层，严重限制重建能力。

更合理的公共锚点应是：

```text
PCA/DCT/Fourier 等有意义 basis
公共可训练 stem
fixed basis + trainable scale
fixed basis + low-rank residual
encoder canonical head
```

## 可行实验建议

### 第一阶段：只分析，不改结构

对已有模型导出 code，分析：

```text
不同 seed code 的 mean/std/cov
不同 seed code covariance eigenvalue
不同 seed code 子空间相似度
Procrustes 对齐前后误差
encoder_seedA + decoder_seedB 互换 NMSE
```

如果 Procrustes 对齐能显著改善，说明主要是线性旋转/坐标变换问题。

### 第二阶段：轻量 code 正则

保持 encoder/decoder 结构不变，只加正则：

```text
loss = recon_loss
     + lambda_mean * ||mean(z)||^2
     + lambda_var  * ||var(z) - target_var||^2
     + lambda_cov  * ||offdiag(cov(z))||^2
```

建议初始权重：

```text
lambda_mean = 1e-4
lambda_var  = 1e-4 或 1e-3
lambda_cov  = 1e-4 或 1e-3
```

观察：

```text
自配对 NMSE 是否明显下降
跨 seed 互换 NMSE 是否改善
code covariance 是否更稳定
```

### 第三阶段：Encoder canonical head

如果轻量正则有效，再考虑从架构层面加入：

```text
encoder backbone -> canonical head -> code
```

canonical head 可以包含：

```text
LayerNorm
orthogonal / normalized projection
ordered scale
optional L1 / non-negative activation
```

不要一开始做强制不可逆或强非负约束，应逐步消融。

### 第四阶段：公共结构锚点

如果只从 encoder 下手仍不足，可以考虑不使用其他 seed 模型信息的公共锚点：

```text
PCA basis from raw CSI data
DCT/Fourier basis
shared projection head initialized once and reused
VQ fixed codebook
```

这些锚点来自数据统计或数学结构，不来自其他 seed encoder/decoder。

## 结论

1. 不引入任何锚点时，无法保证不同 seed 的 encoder 自行学到同一个码字坐标系。
2. Code 正则可以减轻坐标漂移，但多数只能约束分布，不能唯一确定语义坐标。
3. 固定随机 decoder 展开层不合适，因为现有实验显示展开层对重建非常关键。
4. 更合理的方向是从 encoder 输出头下手：orthogonal/normalized head、ordered variance、code covariance 正则、稀疏/非负约束。
5. 如果需要真正稳定跨 seed 坐标，最终仍需要某种公共锚点，例如公共 canonical head、PCA/DCT basis 或 fixed codebook。
