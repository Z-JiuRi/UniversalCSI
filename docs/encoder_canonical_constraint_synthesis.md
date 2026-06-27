# Encoder 码字坐标约束综合分析

本文综合 `encoder_code_regularization_and_canonical_coordinates.md` 与
`encoder_architecture_constraints_analysis.md`，并结合当前代码结构和已有实验记录，重新判断：

```text
能否只从 encoder 下手，让不同 seed 的码字进入更一致的公共坐标系？
```

结论先行：

```text
只靠普通 code 正则不能唯一确定码字坐标系。
只靠固定随机投影也有明显风险，尤其可能损伤重建表达力。
更合理的方向是：在 encoder 输出头处引入“公共结构锚点 + 有限自由度”，再用 code 统计正则辅助。
```

## 1. 当前代码里的坐标自由度在哪里

当前统一接口在 `models/UniversalCSI.py`：

```text
CSI input -> encoder -> code -> decoder -> reconstructed CSI
```

`UniversalCSIModel.encode()` 当前只做：

```python
code = self.encoder(x)
```

如果启用 adapter，才会额外执行：

```python
code = self.code_adapter(code)
```

也就是说，基础训练路径没有任何公共 code 坐标约束。不同 seed 的 encoder 可以自由选择任意 latent 坐标，只要对应 decoder 能解码即可。

以 `transnet_transnet` 为例，关键线性对称结构非常明显：

```text
TransNetEncoder:
  CSI -> TransformerEncoder -> flatten(2048) -> encoder.fc(2048 -> 512) -> code

TransNetDecoder:
  code -> decoder.fc_decoder(512 -> 2048) -> TransformerDecoder -> CSI
```

这里的 `encoder.fc` 和 `decoder.fc_decoder` 构成一对可互相补偿的线性坐标变换。若 encoder 端输出被某个可逆矩阵 `A` 变换：

```text
z' = A z
```

decoder 第一层只需学习近似：

```text
fc_decoder' = fc_decoder A^-1
```

整体重建 loss 可以几乎不变。这是跨 seed 坐标系不一致的核心结构原因。

`hybrid` decoder 虽然没有名为 `fc_decoder` 的层，但存在等价的第一段码字解释模块：

```text
semantic_projector(code_dim -> code_dim)
token_projection(code_dim -> input_dim)
```

因此同样存在 code 坐标被 decoder 前端吸收和重解释的问题。

## 2. 实验数据说明了什么

### 2.1 自配对重建很好，不代表坐标一致

已有完整训练结果显示，同架构不同 seed 自配对都可以达到较好 NMSE：

```text
transnet_transnet:
  seed42:   -28.126 dB
  seed2026: -28.180 dB
  seed3407: -26.868 dB

transnet_hybrid:
  seed42:   -28.407 dB
  seed2026: best -28.207 dB, final -25.870 dB
  seed3407: -27.562 dB
```

这说明模型容量足够，单个 encoder/decoder 对可以很好重建。

但架构分析文档记录了跨 seed 码字空间的典型现象：

```text
跨 seed 质心余弦相似度约 0
跨 seed 逐维 Pearson 相关约 0
主子空间中位角约 72 度
特征值谱余弦相似度 > 0.97
线性回归 R² 为 0.96 ~ 0.99
Procrustes 旋转后残差约 12% ~ 20%
```

这些数据组合在一起，含义很清楚：

```text
信息内容大体相同，统计谱也类似；
但坐标轴方向几乎随机，主要差异是 seed 特异的线性旋转/仿射变换。
```

换句话说，不同 seed 不是没有学到 CSI 信息，而是把同一类信息放进了不同的 512 维坐标系。

### 2.2 只调整 decoder 第一层也远远不够

已有 `unfreeze_fc_decoder` 实验更能说明 decoder 前端的重要性：

```text
加载 seed42 transnet decoder body
加载其他 seed transnet encoder
冻结 decoder body，只训练 decoder.fc_decoder

seed2026 -> seed42 decoder body: -22.561 dB
seed3407 -> seed42 decoder body: -22.083 dB
seed3407 lr=1e-3:               -22.211 dB
```

这些结果比跨 seed 直接互换要好，但距离完整自配对的 `-27 ~ -28 dB` 仍有大约 `5 ~ 6 dB` 差距。

这说明两点：

1. `fc_decoder`/decoder stem 确实承载了大量坐标解释能力。
2. 坐标差异不只是一个能被第一层完全修复的简单缩放；后续 decoder body 也已经适配了私有码字分布。

因此，固定随机 `fc_decoder` 或固定随机展开层作为公共锚点并不理想。它会锁死最重要的重建入口，而已有实验显示即使这个入口可训练，恢复能力也有限。

## 3. 数学本质：不是优化问题，而是不可辨识性

基础自编码目标是：

```text
min MSE(D(E(x)), x)
```

对任意可逆矩阵 `A`：

```text
E'(x) = A E(x)
D'(z') = D(A^-1 z')
```

都有：

```text
D'(E'(x)) = D(E(x))
```

这意味着 loss 本身无法区分这些等价解。对 `code_dim=512`，这个自由度近似是：

```text
GL(512): 512 * 512 = 262144 个线性自由度
```

如果只要求正交或白化，最多把自由度压到：

```text
O(512): 512 * 511 / 2 = 130816 个旋转自由度
```

这仍然非常大。因此，“让 encoder 自己学到统一坐标系”不是一个普通正则就能解决的问题；必须人为破坏这些等价解。

## 4. 为什么普通 code 正则不够

Code 正则大致分为四类：

```text
能量/范数约束
均值/方差/协方差约束
稀疏/非负约束
结构化先验
```

它们的共同问题是：大多数只约束分布形状，不定义样本级坐标语义。

例如：

```text
mean(z)=0         旋转后仍成立
||z||2            正交旋转不变
cov(z)=I          任意正交旋转后仍成立
offdiag(cov)=0    等方差情况下仍有旋转不变性
```

非均匀方差目标，例如：

```text
var(z_i) -> exp(-i / tau)
```

可以减少维度置换和任意旋转，因为每个维度有不同能量身份。但它仍只约束二阶统计，并不保证：

```text
第 i 维在不同 seed 中表达同一种 CSI 语义
```

稀疏和非负约束更强，因为旋转后的 dense 实值向量通常不再稀疏/非负。但它也有两个问题：

```text
可能明显牺牲自重建 NMSE
仍不能排除维度置换、局部混合、decoder 反向补偿
```

所以 code 正则的定位应是：

```text
辅助压缩等价解空间，而不是单独定义 canonical 坐标系。
```

## 5. 架构约束方案的重新判断

第二份文档提出了五类方案。结合当前代码和实验数据，可以重新排序。

### 5.1 方案一：固定随机正交投影 Q

形式：

```text
features(2048) -> fixed Q -> code(512)
```

它的优点是明确给 code 维度定义了固定探针方向。不同 seed 不能自由改变 `encoder.fc` 的行空间，因此可以显著减少 `encoder.fc` 带来的 GL(512) 对称性。

但它有两个实际风险：

1. 当前 encoder 的 `fc` 是重要压缩头，直接替换成随机固定 Q 可能显著损伤表达力。
2. 如果 decoder 仍自由训练，它仍可能适配固定 Q 下的私有分布，跨 seed decoder body 兼容性未必完全解决。

因此固定 Q 可以作为诊断性 baseline，但不应直接作为最终方案。

### 5.2 方案二：固定随机 codebook

形式：

```text
features -> assignment logits -> softmax weights -> fixed codebook convex combination
```

它比固定 Q 更强，因为 code 被限制在固定词表的凸包中。随机 codebook 的对称群接近有限离散群，理论上更能压制旋转。

但它的问题也更明显：

```text
表达空间受凸包限制
K 的选择敏感
assignment 网络仍可学出 seed 特异分配方式
```

如果目标是工程上稳定提升跨 seed 兼容性，codebook 适合做探索性方案，而不是第一优先级。

### 5.3 方案三：固定 Q + 低秩可学习残差

形式：

```text
code = features @ Q.T + low_rank_residual(features)
```

其中低秩残差秩 `r << 512`，例如 `r=8/16/32`。

这是两份文档中最值得优先考虑的结构方向，因为它在数学和工程上最平衡：

```text
固定 Q 提供公共坐标锚点
低秩残差保留必要表达力
不同 seed 的个性化差异被限制在 r 维子空间
```

它不是“冻结随机层导致模型无路可走”，而是：

```text
大部分坐标由公共固定结构定义；
少量自由度用于补偿数据和优化差异。
```

对当前代码来说，它最适合替换或包裹 `TransNetEncoder.fc` 这类 encoder 输出头，而不是接在 encoder 输出后做外部后处理。

### 5.4 方案四：白化 + 固定 Q

形式：

```text
features -> whitening -> fixed Q -> code
```

它能消除均值和二阶统计差异。如果 batch 足够大，理论约束强于简单 LayerNorm/方差正则。

但当前训练 batch 通常是 `200/256`，而 features 维度是 `2048`。要估计 `2048 x 2048` 协方差，batch 明显不足。IterNorm 又会引入大量参数和复杂度。

因此它更适合作为后续高成本实验，不适合作为当前第一步。

### 5.5 方案五：正交初始化/频谱归一化/正交正则

这类方案能改善几何稳定性，但不能解决坐标唯一性。

原因是正交约束最多把自由度限制到 `O(512)`，而不同 seed 的正交矩阵仍可以相差很远。第二份文档中的 `||R - I|| ≈ 32` 已经说明跨 seed 旋转接近随机正交水平。

因此它只能作为辅助正则，例如配合方案三稳定训练。

## 6. 更合理的总体路线

综合代码结构、实验结果和数学约束，推荐路线不是“单纯 code 正则”，也不是“固定随机 decoder 层”，而是：

```text
从 encoder 输出头下手，引入公共结构锚点；
用低秩残差保留表达力；
用 code 正则辅助压缩剩余自由度；
用跨 seed 互换 NMSE 和 Procrustes 指标验证。
```

建议优先级：

### 第一优先级：Encoder canonical head with low-rank residual

替换 encoder 最后一层：

```text
features -> canonical projection head -> code
```

推荐结构：

```text
features
  -> LayerNorm 或 RMSNorm
  -> fixed orthogonal Q
  -> trainable bias / scale
  -> low-rank residual, rank=8/16/32
  -> optional code normalization
```

关键是残差必须受限：

```text
full trainable 2048 -> 512 会重新恢复 GL(512) 自由度
low-rank residual 才能把 seed 特异坐标变化限制住
```

### 第二优先级：配套 code 正则

不要指望它单独解决问题，而是作为辅助：

```text
mean(z) -> 0
offdiag(cov(z)) -> 0
var(z_i) -> fixed decreasing target_var_i
optional small L1
```

建议从很小权重开始，避免牺牲自重建：

```text
lambda_mean = 1e-4
lambda_cov  = 1e-4 ~ 1e-3
lambda_var  = 1e-4 ~ 1e-3
lambda_l1   = 0 或 1e-6 ~ 1e-5
```

### 第三优先级：公共数据 basis

如果固定随机 Q 表达力不稳，可以改为公共数据或物理 basis：

```text
PCA basis from raw CSI data
2D-DCT / Fourier basis
block-DCT basis
```

这类 basis 不来自其他 seed encoder/decoder，因此满足“公共锚点”要求，同时比随机 Q 更贴近 CSI 信号结构。

## 7. 评估不能只看自配对 NMSE

这类实验的核心指标不是单模型 NMSE，而是坐标一致性和跨 seed 可用性。

必须同时看：

```text
encoder_seedA + decoder_seedA 自配对 NMSE
encoder_seedA + decoder_seedB 互换 NMSE
跨 seed code 逐样本 cosine / L2
跨 seed 质心 cosine
code covariance eigen spectrum
Procrustes 对齐前后误差
线性回归 R² 和残差
```

如果一个方案自配对只掉 `0.5 ~ 1 dB`，但互换 NMSE 从崩溃或 `-22 dB` 提升到更接近 `-26 dB`，它就是有价值的。

反过来，如果自配对维持 `-28 dB`，但 Procrustes 仍显示接近随机旋转，跨 seed 互换仍差，那说明约束没有真正锚住坐标。

## 8. 最终判断

1. 当前代码中的 `encoder.fc` 与 `decoder.fc_decoder/token_projection` 构成了跨 seed 坐标漂移的核心自由度。
2. 实验数据显示不同 seed 自配对可达 `-27 ~ -28 dB`，但只训 `fc_decoder` 适配固定 decoder body 只能到约 `-22 dB`，说明码字私有坐标已经深入影响 decoder 前端和 body。
3. 普通 code 正则不能唯一约束坐标系，因为它主要约束分布统计，无法消除自编码器的可逆变换对称性。
4. 纯固定随机投影能提供锚点，但可能过度限制最重要的压缩头；更稳妥的是固定公共主投影 + 低秩可学习残差。
5. 最推荐的方向是 encoder 输出头结构化：`fixed/shared basis + low-rank residual + small code regularization`。
6. 如果坚持“不使用其他 seed encoder/decoder 信息”，公共锚点应来自数学结构或原始 CSI 数据统计，而不是某个训练好的模型。

一句话总结：

```text
跨 seed 不兼容不是 encoder 没学会 CSI，而是 encoder/decoder 成对学会了私有坐标协议；
要解决它，必须把这个协议的一部分变成公共协议，而最合适的切入点是 encoder 的最后坐标头。
```
