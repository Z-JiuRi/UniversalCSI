# TransNet-TransNet Decoder 跨 Seed 权重差异分析报告

## 结论摘要

本报告分析 `exps/COST2100/in/base/seed*/transnet_transnet/checkpoints/best_nmse.pth` 中 243 个 seed 的 `transnet_transnet` decoder 权重。统计只包含真实 decoder 参数，不包含 checkpoint 里的 `total_ops`、`total_params` 统计标量。

核心结论：

- 同一份 COST2100/in 数据、同一 `transnet_transnet` 架构、仅 seed 不同，decoder 参数差异非常大。
- 全 decoder 参数量为 1,646,592，243 个 seed 形成 29,403 个 seed 对。
- 全 decoder 两两相对 L2 距离均值为 1.3127，median 为 1.3126；按每个参数维度计的差值 RMS 均值为 0.05527，而权重自身 RMS 均值为 0.04210。
- 除 LayerNorm 权重和最后 `decoder.norm` 外，大多数大参数层的相对 L2 接近 `sqrt(2)=1.4142`，cosine 接近 0，说明不同 seed 学到的权重方向几乎互不相关。
- 中心化后，全 decoder seed 对 cosine 均值为 -0.00413，接近 `-1/(243-1)`，说明去掉公共均值后，各 seed 权重偏移方向近似高维各向同性分布。
- PCA 结果显示前 10 个主成分只解释 4.52% 方差，解释 90% 方差需要 217 个成分，participation dimension 为 241.62，说明这些 decoder 权重不是一个明显低维流形。
- 因此，若目标是“只给一个 decoder 参数，确定性变换到另一个特定 seed 的 decoder 参数”，从当前统计看信息不足，不太可行；若目标是“用 encoder 输出码字作为条件，生成与该 encoder/code 空间匹配的 decoder”，这是更合理的问题。这个方向有可行性，但条件信息不能只用简单全局统计，建议用固定样本顺序的 codebook 或 `(code, CSI)` 功能约束来指导参数生成。

## 数据与统计口径

实验目录：

```text
exps/COST2100/in/base/seed*/transnet_transnet/checkpoints/best_nmse.pth
```

样本数：

| 项目 | 数值 |
|---|---:|
| checkpoint 数 | 243 |
| seed 对数 | 29,403 |
| decoder 参数张量数 | 40 |
| decoder 参数总数 | 1,646,592 |
| checkpoint epoch 范围 | 400 - 400 |
| best epoch 范围 | 201 - 400 |
| best NMSE 均值 | -26.4758 dB |
| best NMSE 最小值 | -28.4560 dB |
| best NMSE 最大值 | -23.4900 dB |

指标定义：

```text
rel_l2(theta_i, theta_j) = ||theta_i - theta_j||_2 / ((||theta_i||_2 + ||theta_j||_2) / 2)
cosine(theta_i, theta_j) = <theta_i, theta_j> / (||theta_i||_2 ||theta_j||_2)
delta_rms(theta_i, theta_j) = sqrt(mean((theta_i - theta_j)^2))
centered_cosine = cosine(theta_i - mean(theta), theta_j - mean(theta))
```

所有统计均为 243 个 seed 的全量两两组合精确计算，不是抽样估计。

## 全 Decoder 差异

| 指标 | mean | std | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel_l2 | 1.312707 | 0.002293 | 1.304458 | 1.311095 | 1.312574 | 1.314196 | 1.322512 |
| cosine | 0.138474 | 0.002978 | 0.125483 | 0.136551 | 0.138643 | 0.140568 | 0.149195 |
| centered_cosine | -0.004131 | 0.001846 | -0.011726 | -0.005394 | -0.004131 | -0.002888 | 0.003577 |
| delta_rms | 0.055270 | 0.000537 | 0.053829 | 0.054881 | 0.055229 | 0.055610 | 0.057664 |
| weight_rms | 0.042103 | 0.000490 | 0.041144 | 0.041750 | 0.042040 | 0.042439 | 0.043667 |
| center_rel_l2 | 0.926260 | 0.001861 | 0.922862 | 0.924864 | 0.925977 | 0.927349 | 0.932363 |

最近 seed 对：

| seed_a | seed_b | rel_l2 | cosine | L2 |
|---:|---:|---:|---:|---:|
| 15257 | 15953 | 1.304458 | 0.149195 | 69.2480 |

最远 seed 对：

| seed_a | seed_b | rel_l2 | cosine | L2 |
|---:|---:|---:|---:|---:|
| 17723 | 39533 | 1.322512 | 0.125483 | 73.9945 |

解释：

- 全 decoder 的 `rel_l2=1.31` 已经接近两个高维随机方向的典型距离。
- 原始 cosine 还有 0.138 的公共相似度，主要来自各 seed 共享的训练后参数均值和归一化类参数；一旦减去 seed 均值，`centered_cosine` 约为 -0.00413，几乎正好是有限样本中心化向量之间的平均负相关。
- 这说明不同 seed 的 decoder 权重不是“同一权重附近的小扰动”，而是围绕公共均值分散在高维空间中的不同方向。

## 分模块差异

| 模块 | 参数量 | rel_l2 mean | rel_l2 median | cosine mean | centered cosine mean | delta_rms mean |
|---|---:|---:|---:|---:|---:|---:|
| fc_decoder | 1,050,624 | 1.414363 | 1.414357 | 0.000004 | -0.004130 | 0.042327 |
| layer0.self_attn | 16,640 | 1.414275 | 1.414293 | -0.000048 | -0.004132 | 0.129944 |
| layer0.multihead_attn | 16,640 | 1.414273 | 1.414265 | -0.000035 | -0.004132 | 0.131280 |
| layer0.linear1 | 133,120 | 1.414101 | 1.414096 | 0.000270 | -0.004131 | 0.057082 |
| layer0.linear2 | 131,136 | 1.414364 | 1.414362 | -0.000026 | -0.004131 | 0.055753 |
| layer0.norm1 | 128 | 0.087370 | 0.086282 | 0.996158 | -0.003963 | 0.060103 |
| layer0.norm2 | 128 | 0.067524 | 0.067196 | 0.997706 | -0.004048 | 0.047610 |
| layer0.norm3 | 128 | 0.063522 | 0.063160 | 0.997968 | -0.003895 | 0.045434 |
| layer1.self_attn | 16,640 | 1.414223 | 1.414218 | 0.000008 | -0.004132 | 0.132491 |
| layer1.multihead_attn | 16,640 | 1.414330 | 1.414354 | 0.000007 | -0.004131 | 0.142107 |
| layer1.linear1 | 133,120 | 1.414210 | 1.414202 | 0.000322 | -0.004130 | 0.063368 |
| layer1.linear2 | 131,136 | 1.414353 | 1.414322 | -0.000024 | -0.004131 | 0.064870 |
| layer1.norm1 | 128 | 0.058191 | 0.057705 | 0.998342 | -0.003381 | 0.041820 |
| layer1.norm2 | 128 | 0.077672 | 0.077282 | 0.996968 | -0.003889 | 0.056289 |
| layer1.norm3 | 128 | 0.060519 | 0.055416 | 0.998046 | -0.003168 | 0.042303 |
| decoder.norm | 128 | 0.842087 | 0.843769 | 0.644152 | -0.004129 | 0.409892 |

关键观察：

- `fc_decoder` 占 decoder 参数量 63.81%，其跨 seed `rel_l2=1.41436`、cosine 约 0，基本等同于高维随机正交方向。
- 两层 Transformer 的 attention 和 FFN 大权重也都接近 `rel_l2=sqrt(2)`、cosine 约 0。
- LayerNorm 的 `weight` 非常稳定，rel_l2 通常只有 0.05 到 0.09，cosine 接近 1。
- 最后的 `decoder.norm` 比中间 LayerNorm 差异更大，rel_l2 为 0.842，但仍明显比大矩阵权重更有公共结构。
- 从参数变换角度看，真正难变换的是 `fc_decoder`、attention 和 FFN 大矩阵；LayerNorm 参数反而可以通过均值模板或简单归一化较好建模。

## 单张量差异

差异最大的参数张量主要是 bias 类参数和大矩阵权重，rel_l2 接近或略高于 `sqrt(2)`。

| 参数张量 | 参数量 | rel_l2 mean | cosine mean | centered cosine mean | delta_rms mean |
|---|---:|---:|---:|---:|---:|
| decoder.layers.0.norm1.bias | 64 | 1.432218 | 0.000357 | -0.003787 | 0.011727 |
| decoder.layers.0.norm2.bias | 64 | 1.425244 | -0.001123 | -0.003931 | 0.010818 |
| decoder.layers.0.self_attn.in_proj_bias | 192 | 1.425224 | 0.000530 | -0.003885 | 0.034989 |
| decoder.layers.1.norm1.bias | 64 | 1.424986 | -0.000014 | -0.003877 | 0.009156 |
| decoder.layers.1.norm2.bias | 64 | 1.424334 | -0.001432 | -0.003929 | 0.024425 |
| decoder.layers.0.multihead_attn.out_proj.bias | 64 | 1.421337 | -0.000814 | -0.004009 | 0.010521 |
| decoder.layers.0.multihead_attn.in_proj_bias | 192 | 1.421021 | 0.000257 | -0.004003 | 0.011533 |
| decoder.layers.1.self_attn.in_proj_bias | 192 | 1.420717 | -0.000280 | -0.004024 | 0.017029 |

差异最小的参数张量主要是 LayerNorm weight。

| 参数张量 | 参数量 | rel_l2 mean | cosine mean | centered cosine mean | delta_rms mean |
|---|---:|---:|---:|---:|---:|
| decoder.layers.1.norm3.weight | 64 | 0.051491 | 0.998511 | -0.001898 | 0.050900 |
| decoder.layers.1.norm1.weight | 64 | 0.057467 | 0.998384 | -0.003362 | 0.058405 |
| decoder.layers.0.norm3.weight | 64 | 0.062474 | 0.998033 | -0.003872 | 0.063191 |
| decoder.layers.0.norm2.weight | 64 | 0.066615 | 0.997766 | -0.004044 | 0.066422 |
| decoder.layers.1.norm2.weight | 64 | 0.073779 | 0.997257 | -0.003790 | 0.075607 |
| decoder.layers.0.norm1.weight | 64 | 0.086478 | 0.996234 | -0.003958 | 0.084128 |
| decoder.norm.weight | 64 | 0.258453 | 0.966361 | -0.003777 | 0.145372 |

注意：LayerNorm weight 的 `delta_rms` 不一定最小，因为这些参数本身的均值约在 1 附近；相对 L2 和 cosine 更能反映其跨 seed 方向稳定性。

## PCA 与权重分布维度

全 decoder 中心化权重的 PCA 统计：

| 指标 | 数值 |
|---|---:|
| top1 explained variance | 0.004579 |
| top5 explained variance | 0.022787 |
| top10 explained variance | 0.045230 |
| top20 explained variance | 0.089482 |
| top50 explained variance | 0.219155 |
| 90% 方差所需成分数 | 217 |
| 95% 方差所需成分数 | 230 |
| participation dimension | 241.621 |

解释：

- 243 个样本中心化后最多只有 242 个非零 PCA 方向；participation dimension 达到 241.62，几乎用满可用维度。
- top10 只解释 4.52% 方差，top50 只解释 21.92% 方差。
- 因此，当前这些 seed 的 decoder 参数云不是一个明显的低维连续轨迹，而更接近高维随机散布。

## 能否把一个 Decoder 参数变换到另一个 Decoder 参数

### 1. 不带目标条件的确定性映射不成立

如果输入只有一个 decoder 参数 `theta_a`，希望确定性输出某个特定 `theta_b`，这个问题本身缺少目标信息。当前 243 个 seed 在同一训练设置下是可交换样本：任意一个 `theta_a` 都可以对应 242 个不同目标 `theta_b`。没有目标 seed、目标性能、目标 encoder/code 分布或其他条件时，映射不是函数。

统计上也支持这一点：

- 大矩阵层之间的 cosine 接近 0。
- 中心化后的全 decoder cosine 均值为 -0.00413，方向几乎无关。
- PCA 没有低维主方向可以解释大部分 seed 差异。

因此，“从一个 seed 的 decoder 权重直接推断另一个特定 seed 的 decoder 权重”不应作为主要路线。

### 2. 可行的是学习 decoder 权重分布，而不是一对一确定性变换

更合理的问题是：

```text
给定同架构、同数据训练得到的一批 decoder 权重，学习其分布；
再从该分布中生成一个新的 decoder，或在给定条件下生成满足条件的 decoder。
```

这时 diffusion 或 flow-matching 可以尝试，但需要强约束：

- 不建议直接在 1,646,592 维原始参数上训练生成模型。样本只有 243 个，维度远大于样本数，直接建模会严重欠定。
- 应先做分层标准化，例如对每个参数张量使用 `theta = mu_layer + sigma_layer * z_layer`，在标准化后的 `z_layer` 上建模。
- 优先在 PCA 子空间、低秩分解参数、或者 block-wise latent 中建模；例如每层保留 PCA top-k，再对拼接 latent 做 flow matching。
- 对 LayerNorm weight、bias 和大矩阵权重应分开处理，因为它们的跨 seed 差异尺度和结构完全不同。
- 生成后必须用重建 NMSE 或少量 finetune 验证，不能只看参数距离。

### 3. 可能有价值的变换类型

建议按从简单到复杂的顺序验证：

| 方法 | 目标 | 预期风险 |
|---|---|---|
| 分层均值模板 | `theta_new = mu` 或 `mu + noise` | 能给出公共结构，但可能性能差，需要评估 |
| 分层白化 + 高斯采样 | 每个张量独立建模均值/方差 | 忽略层间相关，可能破坏功能 |
| PCA latent 采样 | 在 242 维以内的经验子空间生成权重 | 当前 PCA 方差很分散，需要较高维 |
| 低秩 delta 建模 | 对大矩阵建模 `Delta W` 的低秩因子 | 若真实差异近似满秩，效果有限 |
| 条件 flow matching | 条件包括目标 seed/code 分布/性能指标 | 样本数少，需要强正则和验证集 |
| 参数空间 diffusion | 直接生成完整参数 | 当前样本维度比过低，不建议作为第一步 |

### 4. 针对当前统计的判断

当前权重差异呈现两个层次：

1. 稳定公共结构：LayerNorm weight、部分归一化统计、训练后全局均值。
2. 高维随机偏移：`fc_decoder`、attention、FFN 大矩阵。

这意味着参数变换的可行方向不是“直接把 A 旋转/平移到 B”，而是“抽取公共结构 + 建模 seed 随机偏移”。flow-matching 可以作为偏移分布建模工具，但它需要在低维 latent 或分块标准化空间中做。

## 建议的后续实验

建议做 4 个最小验证实验：

1. 均值 decoder 评估：构造 `theta_mean = mean_seed(theta)`，加载到同架构 decoder，直接评估 NMSE。
2. 线性插值评估：选择最近 seed 对和最远 seed 对，评估 `theta(t) = (1-t)theta_a + t theta_b` 在 `t=0,0.25,0.5,0.75,1` 的 NMSE，判断是否存在 mode connectivity。
3. 分层 PCA 重建：对每层保留 top-k PCA 成分重建原 checkpoint，测试 k 对 NMSE 的影响。
4. 条件/无条件 latent flow matching：先在 PCA latent 上训练 flow matching，生成 decoder 后做直接评估和少量 finetune，比较是否优于均值模板和高斯采样基线。

如果均值 decoder 或 PCA 低维重建的 NMSE 明显退化，说明权重空间可生成性较弱，后续应转向功能空间约束，例如用固定 code 输入的 decoder 输出一致性、Jacobian 约束或少量数据蒸馏来训练参数生成器。

## 加入 Encoder Codeword 条件后的再分析

上面的无条件分析确实过于弱：decoder 不是凭空生成的，它应该和对应 encoder 的 code 空间匹配。对 `transnet_transnet` 来说，每个 seed 都有：

```text
train_code.pt: (100000, 512)
decoder 参数: 1,646,592
```

因此更合理的目标是：

```text
给定某个 encoder 在固定训练集上的 codebook C_s = E_s(X_train)，
生成一个 decoder 参数 theta_s，使 D_theta_s(C_s) 能重建 X_train。
```

这和“从一个 decoder 变到另一个 decoder”不同。这里的条件 `C_s` 携带了 encoder 学到的 latent 坐标系、尺度、分布和样本排列信息，理论上可以约束 decoder。

### 码字条件的初步实证信号

我额外读取了 243 个 `train_code.pt`，对每个 seed 的完整 `[100000, 512]` codebook 精确计算了每维均值和标准差，然后分析这些码字分布摘要与 decoder 参数距离的相关性。这里没有抽样，但条件只用了 `mean(code)` 和 `std(code)`，不是完整 codebook。

码字统计：

| 指标 | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| code RMS | 0.754751 | 0.041180 | 0.636407 | 0.751247 | 0.887423 |
| code mean pair rel_l2 | 1.416060 | 0.031200 | 1.288322 | 1.416406 | 1.551036 |
| code mean pair cosine | -0.000234 | 0.044066 | -0.186164 | -0.000246 | 0.174832 |
| code std pair rel_l2 | 0.202987 | 0.026847 | 0.128368 | 0.198989 | 0.400629 |
| code std pair cosine | 0.982149 | 0.003474 | 0.966035 | 0.982507 | 0.991761 |
| `[mean,std]` summary pair rel_l2 | 0.621606 | 0.026142 | 0.526662 | 0.620237 | 0.762410 |
| `[mean,std]` summary pair cosine | 0.809154 | 0.015598 | 0.709526 | 0.810082 | 0.863172 |

码字摘要距离与 decoder 参数距离的相关性：

| 条件摘要 | decoder rel_l2 Pearson | decoder rel_l2 Spearman | decoder cosine Pearson | decoder cosine Spearman |
|---|---:|---:|---:|---:|
| code mean | 0.00637 | 0.00481 | 0.00123 | 0.00028 |
| code std | -0.19464 | -0.22032 | -0.33965 | -0.33584 |
| `[mean,std]` summary | -0.03801 | -0.04202 | -0.06006 | -0.05878 |

解释：

- `mean(code)` 的跨 seed 方向几乎随机，和 decoder 参数距离基本无关。
- `std(code)` 在 seed 间非常相似，cosine 均值 0.982；它和 decoder cosine 有一定负相关，但幅度不足以单独预测 164 万维 decoder 参数。
- `[mean,std]` 合并摘要与 decoder 参数距离相关性仍很弱。
- 所以，简单全局码字分布统计不是充分条件。要让 codeword 条件真正有用，必须保留更多结构，尤其是“每个训练样本对应的 code”这一 sample-wise 对齐关系。

### 为什么完整 codebook 比全局统计更有价值

decoder 学的是从 code 到 CSI 的函数：

```text
z_i = E_s(x_i)
x_i_hat = D_s(z_i)
```

如果只看 code 的均值和方差，会丢掉大量信息：哪个 code 对应哪个 CSI、局部邻域如何组织、不同维度组合如何编码空间结构等。完整 codebook `C_s` 在固定训练集顺序下包含更强的条件：

- 行索引和训练样本 `x_i` 对齐，可以隐式给出 code-to-data 的对应关系。
- code 空间的局部几何能反映 encoder 的坐标系。
- 不同 seed 的 encoder 可能实现了功能等价但坐标不同的表示，decoder 参数需要适配这个坐标系。

因此，条件生成时不应只输入一个 1024 维 `[mean,std]` 摘要，而应使用 code set encoder 或固定 probe code 子集。

### 推荐的条件生成形式

更合理的生成器结构是：

```text
C_s = E_s(X_train)                  # (N, code_dim)
h_s = CondEncoder(C_s, optional X)  # codebook 条件表示
theta_s = ParamGenerator(h_s, noise)
loss = L_param(theta_s, theta_s*) + lambda_rec * ||D_theta_s(C_s_probe) - X_probe||^2
```

其中：

- `CondEncoder` 可以是 DeepSets、Set Transformer、Perceiver、随机投影统计、PCA/协方差摘要，或固定 probe 子集上的小 Transformer。
- `ParamGenerator` 不建议直接生成 164 万维全参数；应分层生成，或者先生成 PCA/低秩 latent，再还原到参数。
- `noise` 仍然需要保留，因为同一类 code 条件下可能存在多个功能近似的 decoder 参数。
- `L_param` 可作为监督信号，但不能只优化参数 MSE；最终必须加 `D_theta(C)-X` 的功能重建损失。

### 条件信息应该怎么选

建议按信息量从低到高验证：

| 条件 | 维度/成本 | 预期 |
|---|---:|---|
| `mean(code), std(code)` | 1024 | 当前统计显示不够强，只能做弱 baseline |
| code covariance / correlation | 512 x 512 | 能保留维度相关性，但仍没有 sample-to-CSI 对齐 |
| 固定 K 个 probe code | K x 512 | 保留样本级结构，建议 K=256/1024 起步 |
| `(probe code, probe CSI)` 成对条件 | K x (512+2048) | 最直接约束 decoder 功能，推荐 |
| 完整 codebook set encoder | 100000 x 512 | 信息最全，但 IO/显存成本高，需要采样或分块 |

如果目标是生成“能用的 decoder”，最推荐的条件不是纯 `C_s`，而是固定 probe 集：

```text
{(E_s(x_i), x_i)}_{i in probe}
```

这会把问题从“猜 decoder 参数”变成“生成一个能把这些 code 映射回 CSI 的函数参数”。这比只用 code 分布更有辨识度。

### Flow-Matching / Diffusion 的具体落点

在有 code 条件后，flow-matching/diffusion 可以这样做：

1. 先把真实 decoder 参数按层标准化：

```text
theta_s^l = mu_l + sigma_l * z_s^l
```

2. 对每层 `z_s^l` 做 PCA 或低秩分解，得到低维参数 latent `u_s`。
3. 用 code condition encoder 得到 `h_s`。
4. 训练条件 flow matching：

```text
v_t = F_phi(u_t, t, h_s)
u_0 ~ N(0, I), u_1 = u_s
```

5. 从条件 `h_new` 生成 `u_new`，还原为 decoder 参数，再用 probe reconstruction loss 做筛选或少量 finetune。

关键点：

- 条件 flow 不是从 `theta_a` 变到 `theta_b`，而是从噪声生成“与 codebook 条件匹配”的 decoder latent。
- 由于只有 243 个 seed，参数 latent 维度必须很小。建议先试 16、32、64、128 维，而不是直接 242 维或原始 164 万维。
- 训练时应同时报告参数 MSE 和重建 NMSE；如果二者冲突，以重建 NMSE 为准。

### 更新后的判断

加入 encoder codeword 条件后，问题明显更有意义，但当前统计给出一个边界：

- 只用 `mean/std` 这种全局码字摘要，不足以预测 decoder 参数。
- 使用固定样本顺序的 codebook 或 `(code, CSI)` probe 对，才可能给出足够条件。
- 最可行的路线不是“参数空间无条件生成”，而是“条件参数 latent 生成 + decoder 功能损失”。
- 如果生成器只看 code 而不看对应 CSI，它最多学习 code 分布到参数分布的相关性；如果同时用 probe CSI 做功能约束，就能直接学习 inverse mapping，成功概率更高。

## 从已有 Decoder 参数出发的条件生成

相比从随机噪声开始，另一个更可行的设定是：

```text
source: theta_a, C_a = E_a(X)
target condition: C_b = E_b(X), optional X_probe
output: theta_b_hat
```

也就是用已有 decoder 参数 `theta_a` 作为初始化/锚点，再结合目标 encoder 的 codeword 条件 `C_b`，生成或编辑出适配目标 code 空间的 decoder 参数。这个问题比无条件生成更有结构，因为 `theta_a` 已经包含了 CSI 重建任务的大量公共知识，生成模型只需要学习“如何把 decoder 从 source code 坐标系迁移到 target code 坐标系”。

### 可行性判断

这个方向可行性高于纯噪声生成，但不能简单理解成 `theta_b = theta_a + small_delta`。

原因是前面的统计显示：

- 大矩阵层跨 seed 差异接近随机正交，`fc_decoder`、attention、FFN 的 `rel_l2` 都接近 `sqrt(2)`。
- 全 decoder 最近 seed 对的 `rel_l2` 也有 1.304，说明不同 seed 的原始参数并不接近。
- 但各 decoder 功能相同，都是把同一数据集的 encoder code 重建为 CSI，因此它们可能存在功能空间的可迁移结构，只是这种结构不表现为原始参数空间的近邻。

所以，从 `theta_a` 出发是有价值的，但模型应学习“条件编辑方向”，而不是假设目标参数就在 source 参数附近。

### 推荐建模方式：条件参数桥接

可以把它建成 conditional bridge / conditional rectified flow：

```text
theta_t = (1 - t) * theta_a + t * theta_b
v_target = theta_b - theta_a
v_phi = F_phi(theta_t, t, cond)
cond = Enc(C_a, C_b, optional X_probe)
```

训练目标：

```text
L_flow = ||v_phi(theta_t, t, cond) - (theta_b - theta_a)||^2
L_rec  = ||D_theta_hat(C_b_probe) - X_probe||^2
L_total = L_flow + lambda_rec * L_rec + lambda_anchor * ||theta_hat - theta_a||_regularized
```

推理时从 `theta_a` 开始积分：

```text
theta_0 = theta_a
d theta / dt = F_phi(theta_t, t, cond)
theta_1 = generated target decoder
```

这里的 flow 不是从噪声到参数分布，而是从一个真实 decoder 到另一个条件匹配 decoder。它更像参数空间的条件传输映射。

### 条件必须包含什么

仅输入目标 `C_b` 仍然可能不够，因为模型不知道 source decoder `theta_a` 对应的 source code 坐标系。更完整的条件应该包含：

```text
cond = {
  source code probe: C_a_probe,
  target code probe: C_b_probe,
  optional CSI probe: X_probe,
  optional source decoder outputs: D_theta_a(C_a_probe)
}
```

推荐最小条件：

```text
{(z_a_i, z_b_i, x_i)} for i in probe
```

其中：

```text
z_a_i = E_a(x_i)
z_b_i = E_b(x_i)
```

这比只给 `C_b` 更强，因为它显式告诉模型同一个 CSI 样本在 source encoder 和 target encoder 下的 code 坐标如何变化。

### 为什么需要 source-target code 对齐

decoder 迁移的本质不是“改参数”，而是“适配 latent 坐标变换”：

```text
D_a(E_a(x)) ~= x
D_b(E_b(x)) ~= x
```

如果存在某种 code-space 映射 `T_{b->a}`：

```text
E_a(x) ~= T_{b->a}(E_b(x))
```

那么一个强 baseline 是：

```text
D_b(z_b) ~= D_a(T_{b->a}(z_b))
```

这说明在直接生成 decoder 参数前，应该先测试 code-space adapter：

```text
z_b -> adapter -> z_a -> frozen D_a
```

如果这个 adapter 能取得不错 NMSE，说明 seed 间主要差异确实是 code 坐标系变换，此时再做 decoder 参数编辑更有依据。如果 adapter 都做不好，直接在参数空间生成 decoder 会更难。

### 比纯噪声生成更适合的路线

建议按以下顺序实验：

1. Code adapter baseline：

```text
Train A_{b->a}: E_b(x) -> E_a(x)
Evaluate D_a(A_{b->a}(E_b(x)))
```

2. Decoder finetune baseline：

```text
Initialize theta = theta_a
Finetune theta on (C_b_probe, X_probe)
```

3. Delta predictor：

```text
Delta_theta = G(theta_a, C_a_probe, C_b_probe, X_probe)
theta_hat = theta_a + Delta_theta
```

4. Conditional bridge flow：

```text
theta_a -> theta_b with condition (C_a_probe, C_b_probe, X_probe)
```

5. Layer-wise bridge flow：

对 `fc_decoder`、attention、FFN、LayerNorm 分别建模，避免一个生成器同时处理尺度和结构差异很大的参数。

### 训练数据如何构造

243 个 seed 可以构造有向 source-target 对：

```text
243 * 242 = 58,806 pairs
```

这比无条件生成的 243 个样本多很多，但这些 pair 不是完全独立样本，因为底层 decoder 只有 243 个。训练时需要按 target seed 或 source seed 做拆分，避免模型记住 seed 身份。

推荐拆分：

```text
train target seeds / val target seeds / test target seeds
```

而不是随机拆 pair。否则同一个 target decoder 会同时出现在训练和测试里，评估会虚高。

### 关键风险

- 原始参数空间不对齐：不同 seed 的 Transformer 层可能存在隐藏单元置换、注意力头等价变换、尺度对称性，直接预测参数 delta 会被这些等价性干扰。
- source decoder 不一定是好初始化：如果 `theta_a` 与 `C_b` 坐标系差异很大，直接参数编辑可能需要大幅移动，优化难度仍高。
- 参数距离不等于功能距离：`theta_hat` 参数上接近 `theta_b` 不一定 NMSE 好，反之也可能成立。
- 只给 `C_b` 不够：必须至少给 source/target probe code 对，最好给 `X_probe` 做功能监督。

### 更新后的推荐结论

从已有 decoder 参数出发、再用 code 条件指导，是比从随机噪声开始更推荐的方向。最合理的形式不是传统无条件 diffusion，而是：

```text
source decoder 参数 + source/target code probe 对 + CSI probe
-> 条件参数编辑 / conditional bridge flow
-> target-compatible decoder
```

这个方向的第一步不应直接训练大 flow，而应先做两个 sanity check：

1. `E_b -> E_a` code adapter 加 frozen `D_a` 是否能重建。
2. 从 `theta_a` 用少量 `(E_b(x), x)` probe finetune 是否能快速接近 `theta_b` 或达到可用 NMSE。

如果这两个 baseline 有效，再上 conditional bridge flow 才有充分依据。
