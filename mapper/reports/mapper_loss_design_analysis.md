# mapper decoder-aware 与 tail-aware loss 设计分析

本文基于 `mapper/reports/mapper_exps_full_analysis.md`、`mapper/reports/mapper_distribution_analysis/*.csv` 和 `mapper/reports/mapper_advanced_analysis/*.csv`，回答当前关于 mapper loss 设计的 11 个问题。本文只分析现有结果，不修改代码。

## 0. 当前实验证据摘要

固定 teacher 与 decoder：

```text
teacher code: exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt
fixed decoder: exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth
```

当前最好结果：

```text
mapper = hybrid
source = seed2026/transnet_transnet
mapped code MSE = 0.003117
mapped code cosine = 0.99767
fixed decoder NMSE = -25.39 dB
teacher code fixed decoder NMSE = -29.10 dB
gap = 3.72 dB
```

分布分析的关键证据：

```text
mapped residual: Laplace better than Normal = 10 / 10
mapped residual 是尖峰重尾，不是普通高斯小噪声
```

高级分析的关键证据：

```text
最佳 hybrid/transnet:
  teacher p95 sample NMSE = -24.57 dB
  mapped  p95 sample NMSE = -21.48 dB
  mapped  p99 sample NMSE = -17.89 dB
  mapped 超过 teacher p95 的样本比例 = 19.5%

mapped residual top50 PCA energy 平均值 = 0.119
mapped residual last256 PCA energy 平均值 = 0.348

fc_decoder.weight 列范数与逐维误差相关性平均值 ≈ -0.020
```

所以当前最主要的问题不是“整体 code 没对齐”。整体 code 分布、cosine、effective rank 都已经被拉近。真正拖累 fixed decoder NMSE 的是：

1. fixed decoder 放大的残余误差；
2. 样本级尾部误差；
3. teacher 低方差方向上的残差；
4. 少数维度/样本的重尾偏差。

## 1. `MSE(D_t(z_a), D_t(z_t))` 和 `MSE(D_t(z_a), x)` 哪个更好？

两个都应该用，但职责不同。

记号：

```text
z_s = source encoder code
z_t = teacher encoder code
z_a = mapper(z_s)
D_t = fixed seed42 decoder
x   = ground-truth CSI
```

### `MSE(D_t(z_a), D_t(z_t))`

这是 teacher reconstruction consistency，也可以叫 `recT loss`。

作用：

```text
让 mapped code 经过 fixed decoder 后，模仿 teacher code 经过 fixed decoder 的输出。
```

优点：

- 它直接约束 fixed decoder 真实看到的输出差异；
- 它等价于一种隐式 decoder-Jacobian 加权 code loss；
- 它不会要求 mapper 超越 teacher encoder/decoder 本身；
- 训练目标更稳定，因为 `D_t(z_t)` 是 fixed decoder 在 teacher code 上的自然输出。

数学上，在 `z_t` 附近：

```text
D_t(z_a) - D_t(z_t) ≈ J_D(z_t) (z_a - z_t)
```

所以：

```text
MSE(D_t(z_a), D_t(z_t))
```

比普通：

```text
MSE(z_a, z_t)
```

更关注 decoder 会放大的方向。

### `MSE(D_t(z_a), x)`

这是真实重建 loss，也可以叫 `rec loss`。

作用：

```text
直接优化最终任务指标：fixed decoder 重建 CSI。
```

优点：

- 和最终 NMSE 最一致；
- 可以修正 teacher reconstruction 本身不完美的部分；
- 对样本级尾部误差更直接。

风险：

- 如果单独使用，mapper 可能离开 teacher code manifold；
- fixed decoder 可能存在一些非自然 code 也能降低训练集重建误差，但泛化或稳定性不一定好；
- 它可能破坏 code-space 对齐，使 `z_a` 不再像 `z_t`。

### 结论

推荐顺序：

```text
主力：MSE(D_t(z_a), D_t(z_t))
辅助但很重要：MSE(D_t(z_a), x)
保底约束：MSE(z_a, z_t)
```

推荐组合：

```text
L = 0.1 * MSE(z_a, z_t)
  + 1.0 * MSE(D_t(z_a), D_t(z_t))
  + 0.5~1.0 * MSE(D_t(z_a), x)
```

如果二选一：

- 想稳定贴近 teacher decoder 行为：先选 `MSE(D_t(z_a), D_t(z_t))`。
- 想直接冲最终 NMSE：选 `MSE(D_t(z_a), x)`，但必须配 code/manifold 正则。

结合当前目标“fixed decoder NMSE 尽量接近 teacher 1 dB 内”，我建议两个都开。

## 2. hard sample weighting 应该怎么做？

hard sample weighting 是对“难样本”提高 loss 权重。当前分析显示样本尾部是主要问题之一：

```text
teacher p95 sample NMSE = -24.57 dB
best mapped p95 sample NMSE = -21.48 dB
best mapped p99 sample NMSE = -17.89 dB
```

这说明平均 loss 掩盖了尾部样本。

### 方式 A：基于 reconstruction error 的连续权重

对每个样本计算：

```text
e_i = mean((D_t(z_a_i) - x_i)^2)
```

然后构造权重：

```text
w_i = clamp(e_i / mean(e), w_min, w_max)
```

例如：

```text
w_min = 0.5
w_max = 3.0
```

loss：

```text
L_rec_weighted = mean(w_i * e_i)
```

实践中建议对 `w_i` 停止梯度：

```text
w_i = stopgrad(w_i)
```

否则模型可能通过操纵权重而不是修误差来降低 loss。

### 方式 B：top-k tail loss

对 batch 内样本误差排序，只取最差的 top `q%`：

```text
e_i = mean((D_t(z_a_i) - x_i)^2)
L_tail = mean(topk(e_i, k = q% * batch_size))
```

建议先用：

```text
q = 20%
```

总 loss：

```text
L = L_base + λ_tail * L_tail
```

推荐：

```text
λ_tail = 0.1 ~ 0.5
```

### 方式 C：基于 teacher tail 阈值

从 teacher code 解码得到 teacher 样本误差分布，取 teacher p95 作为阈值：

```text
τ = percentile(e_teacher, 95)
```

只惩罚超过 teacher p95 的 mapped 样本：

```text
L_tail = mean(relu(e_mapped - τ))
```

这个方式很贴合目标：“不要让太多 mapped 样本掉出 teacher 的正常尾部范围”。

### 推荐

先用方式 B，最简单、稳定、和当前问题直接相关：

```text
L_tail = mean(top20%(per-sample MSE(D_t(z_a), x)))
```

如果后续想更精细，再加 teacher p95 threshold。

## 3. dimension weighting 怎么做？

dimension weighting 是对 code 维度或 residual 维度加权。当前分析显示：

```text
dim RMSE max 远高于 dim RMSE p95
hybrid/csinet max dim RMSE = 0.5543
hybrid/transnet max dim RMSE = 0.3239
```

说明少数 code 维度残差很大。

### 方式 A：基于 mapped residual 的逐维误差权重

先统计每个 code 维度的 residual MSE：

```text
r_j = mean_i((z_a_ij - z_t_ij)^2)
```

构造权重：

```text
w_j = clamp(r_j / mean(r), w_min, w_max)
```

然后：

```text
L_dim = mean_j w_j * mean_i((z_a_ij - z_t_ij)^2)
```

风险：这个权重只看 code error，不一定等于 decoder 敏感方向。

### 方式 B：基于 teacher PCA / covariance 的维度权重

teacher covariance 的低方差方向如果被扰动，Mahalanobis 会放大它：

```text
diff = z_a - z_t
u = P_t^T diff
L_pca = mean_j u_j^2 / (λ_j + eps)
```

这里 `λ_j` 是 teacher covariance 的第 `j` 个特征值。低方差方向 `λ_j` 小，所以误差被更大惩罚。

这和现有分析更吻合，因为 mapped residual 的能量更多转移到 teacher 低方差方向：

```text
mapped residual last256 PCA energy 平均值 = 0.348
```

### 方式 C：基于 decoder 输出差异反推敏感维度

最直接但实现更复杂：

```text
想估计 w_j ≈ E ||∂D_t(z) / ∂z_j||^2
```

可以用 Jacobian、finite difference 或 autograd 近似。但 512 维 code 和大 decoder 下成本较高。

### 不推荐只用 `fc_decoder.weight` 列范数

现有分析显示：

```text
fc_sens_dim_mse_corr 平均 ≈ -0.020
范围 -0.103 到 0.265
```

这说明第一层 `fc_decoder.weight` 的列范数无法充分解释 decoder NMSE 差距。原因是后面还有 Transformer 层，敏感方向经过非线性和 attention 混合。

### 推荐

如果要做 dimension weighting，优先顺序：

```text
1. teacher PCA/Mahalanobis weighting
2. residual dim RMSE weighting
3. decoder Jacobian proxy
4. fc_decoder.weight column norm
```

最稳的实现是：

```text
L_dim = mean((P_t^T(z_a-z_t))^2 / (λ_t + eps))
```

但权重要小，作为 manifold 正则，不要压过 decoder-aware loss。

## 4. teacher covariance/manifold 正则作用是什么？怎么做？

作用是防止 mapper 为了降低重建误差，把 `z_a` 映射到 fixed decoder 没见过的 code 区域。

teacher code 不是任意 512 维向量，而是 teacher encoder 在数据集上产生的一片统计流形。fixed decoder 训练时只看过这片流形附近的 code。如果 `z_a` 离开这片流形，即使训练集某些样本重建变好，也可能导致：

- fixed decoder 输出不稳定；
- 样本尾部变差；
- 对不同 source/architecture 泛化差；
- code 分布看起来对齐但局部结构不自然。

### 可以约束哪些统计

1. 均值：

```text
mean(z_a) ≈ mean(z_t)
```

2. 方差：

```text
var(z_a) ≈ var(z_t)
```

3. 协方差：

```text
cov(z_a) ≈ cov(z_t)
```

4. PCA basis 下的能量：

```text
P_t^T(z_a - μ_t) 的分布 ≈ P_t^T(z_t - μ_t)
```

5. Mahalanobis distance：

```text
(z_a - μ_t)^T Σ_t^{-1}(z_a - μ_t)
```

### 推荐做法

不要一开始做完整 covariance MSE，因为 512x512 协方差容易带来噪声和计算开销。建议先做 PCA/Mahalanobis 弱正则：

```text
u_a = P_t^T(z_a - μ_t)
L_manifold = mean_j (var(u_a_j) - λ_j)^2 / λ_j^2
```

或者更简单：

```text
L_mahal = mean_i (z_a_i - μ_t)^T Σ_t^{-1}(z_a_i - μ_t)
```

但 `L_mahal` 单独最小化会把所有点推向均值，所以更合理的是约束到 teacher 的典型范围，而不是越小越好。

例如：

```text
d_a = Mahalanobis(z_a)
d_t = Mahalanobis(z_t)
L_mahal = MSE(log(d_a), log(d_t))
```

这比直接最小化 `d_a` 更合理。

### 权重

建议作为弱正则：

```text
λ_manifold = 1e-4 ~ 1e-3
```

它的作用是稳定，不是主力提升 NMSE。主力仍然是 decoder-aware loss。

## 5. `d_M(z_a) = (z_a - μ_t)^T Σ_t^{-1} (z_a - μ_t)` 是什么意思？能用来做 loss 吗？

这是 Mahalanobis distance，表示 `z_a` 离 teacher code 分布中心有多远，并且按 teacher 分布的协方差进行归一化。

普通 L2 是：

```text
||z_a - μ_t||^2
```

它认为所有方向同等重要。

Mahalanobis 是：

```text
(z_a - μ_t)^T Σ_t^{-1} (z_a - μ_t)
```

如果 teacher code 在某个方向方差很大，说明这个方向自然变化范围大，那么同样的偏移不太异常。如果 teacher code 在某个方向方差很小，说明这个方向本来很稳定，那么一点偏移就很异常。

PCA 形式更直观。设：

```text
Σ_t = P diag(λ) P^T
u = P^T(z_a - μ_t)
```

则：

```text
d_M(z_a) = Σ_j u_j^2 / λ_j
```

所以它会强烈惩罚 teacher 低方差方向上的偏移。

### 能不能做 loss？

能，但不能直接做：

```text
L = mean(d_M(z_a))
```

因为这样会把所有 `z_a` 往 teacher 均值 `μ_t` 拉，导致 code 塌缩。

更合理的做法是：

### 方式 A：匹配 teacher 的 Mahalanobis 分布

```text
d_a = d_M(z_a)
d_t = d_M(z_t)
L = MSE(log(d_a), log(d_t))
```

### 方式 B：只惩罚超出 teacher 正常范围的点

```text
τ = percentile(d_M(z_t), 95)
L = mean(relu(d_M(z_a) - τ))
```

### 方式 C：对 pair residual 做 whitened code loss

```text
diff = z_a - z_t
L = mean(diff^T Σ_t^{-1} diff)
```

这个更像 dimension weighting，强调 teacher 低方差方向的 pairwise 对齐。

### 推荐

如果是 mapper 的 pairwise 对齐任务，我更推荐方式 C：

```text
L_whiten_pair = mean((P_t^T(z_a-z_t))^2 / (λ_t + eps))
```

但权重必须小：

```text
λ_whiten = 1e-4 ~ 1e-3
```

否则会过度追低方差方向，影响主重建 loss。

## 6. 能不能用 `D_t(z_a)-D_t(z_t)` 来找哪些 code 维度误差最容易被放大，并加权？

直接从 `D_t(z_a)-D_t(z_t)` 不能唯一分解到 code 维度，因为 decoder 是多层非线性映射。输出误差是所有 code 维度共同作用的结果。

但可以用它做两类事情。

### 方式 A：直接作为 loss

这是最推荐的：

```text
L_recT = MSE(D_t(z_a), D_t(z_t))
```

它不需要显式知道哪个 code 维度敏感，反向传播会自动把梯度分配给导致输出误差的 code 维度和 mapper 参数。

### 方式 B：用梯度估计敏感维度

如果一定要得到维度权重，可以估计：

```text
s_j = E_i |∂ L_recT_i / ∂ z_a_ij|
```

或：

```text
s_j = E_i ||∂D_t(z_a_i) / ∂z_a_ij||^2
```

然后：

```text
L_code_weighted = mean_j s_j * mean_i((z_a_ij - z_t_ij)^2)
```

但这比直接 `L_recT` 更复杂，并且可能不稳定。

### 推荐

不要先做显式敏感维度加权。直接加入：

```text
MSE(D_t(z_a), D_t(z_t))
```

它就是最自然的 decoder sensitivity weighting。

如果后续仍想做显式维度权重，再用梯度统计，而不是 `fc_decoder.weight` 列范数。

## 7. 现在最主要是哪个方向的问题？

结合现有报告，我认为主因排序是：

### 第一主因：decoder 敏感方向没有被直接优化

证据：

```text
best mapped code MSE = 0.003117
best mapped cosine = 0.99767
但 fixed decoder NMSE 仍比 teacher 差 3.72 dB
```

说明 code-space average MSE 已经不足以描述 decoder 真实误差。必须直接优化：

```text
MSE(D_t(z_a), D_t(z_t))
MSE(D_t(z_a), x)
```

### 第二主因：少数样本尾部拖累

证据：

```text
teacher p95 sample NMSE = -24.57 dB
best mapped p95 sample NMSE = -21.48 dB
best mapped p99 sample NMSE = -17.89 dB
best mapped 超过 teacher p95 的比例 = 19.5%
```

说明 fixed decoder NMSE 不是所有样本均匀变差，而是尾部明显变差。需要 hard sample weighting/top-k tail loss。

### 第三主因：teacher 低方差方向上的残差

证据：

```text
mapped residual top50 PCA energy 平均值 = 0.119
mapped residual last256 PCA energy 平均值 = 0.348
```

主方向已经对齐，但低方差方向还有残差。这些方向在普通 code MSE 里不显眼，但可能对 decoder 有影响。需要 teacher PCA/Mahalanobis 弱正则。

### 第四主因：少数维度重尾

证据：

```text
mapped residual Laplace better than Normal = 10/10
residual kurtosis 远大于 3
dim RMSE max 远大于 dim RMSE p95
```

说明存在维度级别尾部误差。可以用 SmoothL1、top-k dim loss、gated residual correction head 处理。

### 不是最主要的方向

`fc_decoder.weight` 列范数解释力弱：

```text
fc_sens_dim_mse_corr 平均 ≈ -0.020
```

nearest-neighbor manifold distance 区分度也弱：

```text
nn_l2_mean ≈ 11.26~11.48
```

所以它们不应该作为下一轮主目标。

## 8. 既然 `fc_decoder.weight` 不够解释，那是不是应该加部分重建损失？

是，应该加。

原因是 transnet decoder 不是单层线性 decoder。它包含：

```text
code -> fc_decoder -> feature map -> Transformer decoder layers -> output CSI
```

`fc_decoder.weight` 只描述第一步线性变换。后面的 Transformer 会继续混合、放大、抑制不同方向的误差。所以仅用 `fc_decoder.weight` 列范数判断敏感维度不够。

更合理的 loss 是：

```text
MSE(fc_t(z_a), fc_t(z_t))
MSE(D_t(z_a), D_t(z_t))
MSE(D_t(z_a), x)
```

其中：

- `fc loss` 约束 decoder 第一层后的 coarse feature；
- `recT loss` 约束完整 decoder 的 teacher 行为；
- `rec loss` 约束最终真实重建。

推荐权重：

```text
λ_fc   = 1e-2
λ_recT = 1.0
λ_rec  = 0.5~1.0
```

`fc loss` 不要太大，因为它只是中间层辅助项。真正应该主导的是完整重建损失。

## 9. 怎么对 decoder error 高的样本加权？

训练时每个 batch 计算每个样本的 decoder reconstruction error：

```text
e_i = mean((D_t(z_a_i) - x_i)^2)
```

然后构造 sample weight。

### 简单版本

```text
w_i = clamp(e_i.detach() / mean(e.detach()), 0.5, 3.0)
L = mean(w_i * e_i)
```

含义：

- 比 batch 平均误差高的样本权重大；
- 比 batch 平均误差低的样本权重小；
- `clamp` 防止极端样本让训练不稳定；
- `detach` 防止模型通过权重路径作弊。

### 更稳版本

用 EMA 维护全局平均误差：

```text
ema_e = 0.99 * ema_e + 0.01 * mean(e)
w_i = clamp(e_i.detach() / ema_e, 0.5, 3.0)
```

这样 batch 波动更小。

### teacher threshold 版本

先统计 teacher reconstruction error：

```text
e_t_i = mean((D_t(z_t_i) - x_i)^2)
τ = percentile(e_t, 95)
```

训练时：

```text
w_i = 1 + α * indicator(e_i > τ)
```

或：

```text
L_tail = mean(relu(e_i - τ))
```

这个版本最贴近目标：让 mapped code 的尾部不要比 teacher code 差太多。

## 10. `L_tail = mean(topk(e_i, k=20%))` 是什么意思？

先对每个样本算 reconstruction error：

```text
e_i = mean((D_t(z_a_i) - x_i)^2)
```

假设 batch size 是 100，那么会得到 100 个误差：

```text
e_1, e_2, ..., e_100
```

`topk(e_i, k=20%)` 的意思是取误差最大的 20 个样本：

```text
top20 = 最大的 20 个 e_i
```

然后：

```text
L_tail = mean(top20)
```

也就是只看当前 batch 里最差的 20% 样本。

它的作用是强迫模型优化尾部，而不是只优化平均误差。

总 loss 可以写成：

```text
L = mean(e_i) + λ_tail * mean(top20%(e_i))
```

如果 `λ_tail=0.2`，意思是：

- 主 loss 仍然看所有样本；
- 额外给最差 20% 样本更多梯度；
- 不让少数很差样本被平均值掩盖。

注意事项：

- `topk` 比连续加权更激进；
- 初期训练不要 `λ_tail` 太大；
- 建议先从 `λ_tail=0.1~0.2` 开始；
- 最好在 mapper 已经粗对齐后再开 tail loss。

## 11. 最终推荐训练目标

结合所有分析，我建议下一轮不要单独测试某一个 trick，而是按两阶段做。

### Stage 1：粗对齐

保持现在类似的 code-space 训练：

```text
L_stage1 = MSE(z_a, z_t)
```

或者更稳一点：

```text
L_stage1 = 0.5 * MSE(z_a, z_t)
         + 0.5 * SmoothL1(z_a, z_t, beta=0.05)
```

目标是把 cosine 拉到 `0.995+`，把 mapped code 带到 teacher code 附近。

### Stage 2：decoder-aware + tail-aware finetune

从 Stage 1 checkpoint 开始，小学习率 finetune：

```text
L = 0.1 * MSE(z_a, z_t)
  + 1.0 * MSE(D_t(z_a), D_t(z_t))
  + 0.5~1.0 * MSE(D_t(z_a), x)
  + 1e-2 * MSE(fc_t(z_a), fc_t(z_t))
  + 1e-4~1e-3 * L_whiten_pair
  + 0.1~0.2 * L_tail
```

其中：

```text
L_whiten_pair = mean((P_t^T(z_a-z_t))^2 / (λ_t + eps))
L_tail = mean(top20%(mean((D_t(z_a_i)-x_i)^2)))
```

推荐第一组实验：

```text
λ_code = 0.1
λ_recT = 1.0
λ_rec  = 1.0
λ_fc   = 1e-2
λ_whiten = 1e-4
λ_tail = 0.1
tail_ratio = 0.2
lr = 5e-5 或 1e-4
```

### 为什么这个组合最符合现有证据

| 观察到的问题 | 对应 loss |
|---|---|
| decoder 对 code 方向敏感，code MSE 不足以解释 NMSE | `MSE(D_t(z_a), D_t(z_t))` |
| 最终指标是重建 NMSE | `MSE(D_t(z_a), x)` |
| 样本 p95/p99 明显变差 | `L_tail` 或 hard sample weighting |
| mapped residual 落到 teacher 低方差方向 | `L_whiten_pair` / Mahalanobis 正则 |
| 需要保持 teacher code manifold | 小权重 code loss + manifold 正则 |
| fc 权重解释力弱但中间层仍有用 | 小权重 `fc loss` |

## 12. 最短结论

1. `MSE(D_t(z_a), D_t(z_t))` 和 `MSE(D_t(z_a), x)` 都应该用；前者稳定贴近 teacher decoder 行为，后者直接优化真实 NMSE。
2. hard sample weighting 应该基于 `D_t(z_a)` 的 per-sample reconstruction error，而不是 code error。
3. dimension weighting 优先用 teacher PCA/Mahalanobis，不建议只用 `fc_decoder.weight` 列范数。
4. teacher covariance/manifold 正则的作用是防止 decoder-aware finetune 把 code 拉出 teacher decoder 熟悉的 code 分布。
5. Mahalanobis 可以做 loss，但不要直接最小化 `d_M(z_a)`，更建议做 pairwise whitened loss 或匹配 teacher Mahalanobis 分布。
6. `D_t(z_a)-D_t(z_t)` 最好直接作为 `recT loss`，不要先绕一圈手工估计敏感维度。
7. 现有证据表明最主要方向是 decoder-aware + hard sample tail，其次是 teacher low-variance/manifold 约束和维度尾部修正。
