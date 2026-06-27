# 码字中信道信息与厂家信息的占比、形式和分离可行性分析

## 结论先行

基于 `exps/COST2100/in/seed<seed_id>/transnet_hybrid/codewords/train_code.pt` 的 59 个 seed 码字分析，当前证据更支持：

```text
码字 = seed/厂家特异的坐标系变换(信道内容)
```

而不是简单的：

```text
码字 = 信道内容 + 厂家偏置
```

更具体地说，比较符合数据的形式是：

```text
z_i(x) ≈ A_i z_42(x) + b_i
```

或者反过来：

```text
z_42(x) ≈ adapter_i(z_i(x))
```

其中：

- `x` 是同一个 CSI 信道样本。
- `z_i(x)` 是 seed/vendor `i` 的 encoder 输出码字。
- `z_42(x)` 是 seed42 canonical code。
- `A_i, b_i` 是 seed/vendor 特异的坐标系变换。

这意味着：信道信息确实保留在码字里，但在原始坐标轴上，不同 seed 的“信道方向”并不对齐。厂家信息也不是一个可以简单切掉的独立字段，而是体现在整个 latent 坐标系的形式上。

因此，更准确的研究目标不是：

```text
从 z 中扣掉厂家信息，剩下信道信息
```

而是：

```text
学习一个厂家特异的坐标变换，把该厂家的 code 翻译到 seed42 decoder 能读懂的 canonical code 空间
```

## 1. “信道信息”和“厂家信息”的占比能不能分析？

可以分析，但必须先声明假设。

码字是一个连续高维向量，不存在天然标签告诉我们哪几维是信道、哪几维是厂家。因此“占比”不是物理真值，只能在某种分解模型下估计。

这里使用二因素方差分解：

```text
z_{seed, sample}
  = 全局均值
  + seed 主效应
  + sample 主效应
  + seed × sample 交互残差
```

解释：

- `seed 主效应`：不同厂家/seed 的全局偏移、尺度、均值差异。
- `sample 主效应`：所有 seed 共享的同一样本信道信息。
- `seed × sample 交互残差`：同一个信道样本在不同 seed 坐标系下的非加法差异。

如果码字真的是：

```text
z = 信道内容 + 厂家偏置
```

那么 `seed 主效应 + sample 主效应` 应该解释绝大部分方差，交互残差应该很小。

如果码字是：

```text
z_i = T_i(信道内容)
```

那么在原始坐标下，sample 主效应会很小，交互残差会很大；但把不同 seed 线性对齐到同一个 canonical 坐标后，sample 主效应会大幅上升。

## 2. 原始码字空间的占比结果

使用 59 个 seed，每个 seed 抽取 4096 个对齐样本，原始码字空间的方差分解结果：

| 分量 | 方差占比 |
|---|---:|
| seed/厂家主效应 | 72.61% |
| sample/信道主效应 | 0.46% |
| seed × sample 交互残差 | 26.93% |

如果对所有维度做全局 z-score 后再分解：

| 分量 | 方差占比 |
|---|---:|
| seed/厂家主效应 | 60.32% |
| sample/信道主效应 | 0.67% |
| seed × sample 交互残差 | 39.02% |

这说明在原始坐标系下，码字里最显眼的是 seed/vendor 差异，而不是跨 seed 共享的信道方向。

但这不能解释成“码字里只有厂家信息、没有信道信息”。原因是：信道信息可能存在于每个 seed 自己的坐标系里，在跨 seed 直接平均时被互相抵消了。

## 3. 去掉简单厂家统计后是否能显出信道信息？

进一步做 per-seed 标准化：

```text
z_i' = (z_i - mean_i) / std_i
```

也就是去掉每个 seed 的均值和尺度差异。

分解结果：

| 分量 | 方差占比 |
|---|---:|
| seed/厂家主效应 | 约 0.00% |
| sample/信道主效应 | 1.68% |
| seed × sample 交互残差 | 98.32% |

这很关键。

如果厂家信息主要只是：

```text
均值偏移 + 尺度差异
```

那么 per-seed 标准化后，同一样本的跨 seed 表示应该明显靠近。

但结果不是这样。去掉均值/尺度后，sample 主效应仍然只有 1.68%，绝大多数差异仍然在交互残差里。

这说明厂家差异不只是简单 bias 或 normalization，而是更深的坐标轴旋转、混合、投影或仿射变换。

## 4. 只做厂家偏置对齐是否足够？

测试一个 bias-only 对齐：

```text
z_i -> z_i - mean_i + mean_42
```

也就是只把每个 seed 的均值平移到 seed42。

结果：

| 指标 | 数值 |
|---|---:|
| bias-only 到 seed42 的 R2 均值 | -0.933 |
| bias-only 到 seed42 的 R2 最好值 | -0.733 |
| bias-only 后 sample 主效应 | 1.68% |
| bias-only 后交互残差 | 98.30% |

R2 为负说明：只做均值平移比直接用 seed42 均值预测还差。

因此：

```text
码字 ≠ 信道信息 + 简单厂家偏置
```

这个 additive 模型明显不够。

## 5. 线性坐标系对齐后的结果

用前 2048 个配对样本学习每个 seed 到 seed42 的线性映射：

```text
z_42 ≈ z_i W_i + b_i
```

然后在 4096 个样本上分析线性对齐后的码字。

结果：

| 分量 | 方差占比 |
|---|---:|
| seed/厂家主效应 | 0.00045% |
| sample/信道主效应 | 98.63% |
| seed × sample 交互残差 | 1.37% |

线性映射到 seed42 的 R2：

| 指标 | R2 |
|---|---:|
| min | 0.931 |
| mean | 0.970 |
| median | 0.971 |
| max | 0.991 |

同一样本跨 seed 距离也发生明显变化：

| 空间 | 同一样本跨 seed 距离 | 不同样本跨 seed 距离 | 比值 |
|---|---:|---:|---:|
| 原始码字 | 40.43 | 40.80 | 0.991 |
| bias-only 对齐 | 20.72 | 21.21 | 0.977 |
| 线性对齐 | 2.25 | 21.56 | 0.104 |

解释：

- 原始空间下，同一样本和不同样本几乎一样远。
- bias-only 只能把整体距离变小，不能真正让同一样本靠近。
- 线性对齐后，同一样本跨 seed 距离只有不同样本距离的 10.4%，说明信道样本结构被恢复出来了。

这强烈支持：

```text
不同 seed 的码字差异主要是坐标系/线性仿射变换，而不是简单加法厂家偏置。
```

## 6. 当前模式更像乘法/坐标变换，还是加法？

从结果看，更像：

```text
码字 = seed 特异性的线性/仿射坐标变换(信道内容)
```

而不是：

```text
码字 = 信道内容 + seed 特异性向量
```

注意这里说的“乘法”不是逐元素相乘，而是广义的线性变换：

```text
z_i = h A_i + b_i
```

其中 `A_i` 可以包含：

- 旋转；
- 缩放；
- 维度混合；
- 方向翻转；
- 子空间重排；
- 非正交投影。

因此更准确的写法是：

```text
z_i = T_i(h)
```

其中 `T_i` 近似可以被仿射映射解释。

## 7. 是否应该让不同 seed 的“信道信息”尽量一致，厂家信息尽量不一致？

如果你显式分成两个表示：

```text
content code: 信道内容
domain code: 厂家/seed 信息
```

那么目标确实应该是：

```text
content: 同一样本跨 seed 尽量一致
domain: 不同 seed 尽量可区分
```

也就是：

```text
content_i(x) ≈ content_j(x)
domain_i != domain_j
```

但要注意：这个目标不应该直接施加在 decoder 输入码字 `z` 上。

原因是固定 seed42 decoder 需要读取的是 canonical code：

```text
z_42
```

如果你强行让原始 `z_i` 同时保留强厂家差异，那么 `D_42` 可能仍然无法解码。更合理的是：

```text
raw code z_i: 可以有厂家特异形式
adapter 参数或 domain embedding: 表示厂家差异
adapter 输出 z_hat_42: 尽量只含 canonical 信道内容
```

也就是：

```text
z_i --adapter_i--> z_hat_42

z_hat_42(x) ≈ z_42(x)
```

在这个结构中：

- 厂家信息不应该留在 `z_hat_42` 里。
- 厂家信息应该进入 `adapter_i` 的参数、vendor embedding 或低秩变换矩阵。
- 信道信息应该在 `z_hat_42` 中对齐到 seed42 canonical space。

## 8. 和 LDA 的类比对不对？

有相似之处，但不能完全等同。

LDA 的典型目标是：

```text
类内距离尽量小
类间距离尽量大
```

如果把“同一个信道样本在不同 seed 下的码字”看成同一类，那么你想要的是：

```text
同一信道样本跨 seed 距离小
不同信道样本距离大
```

这对应的是信道内容对齐，类似 supervised metric learning：

```text
positive pair: 同一样本，不同 seed
negative pair: 不同样本
```

可以用：

```text
contrastive loss
triplet loss
supervised contrastive loss
```

但如果把“seed/厂家”看成类，那么 LDA 会鼓励不同 seed 分得更开。这和固定 decoder 目标是冲突的，因为 decoder 希望看到的是统一 canonical code，而不是更强的厂家差异。

所以正确的类比应该是双空间：

```text
content space:
  同一信道样本跨 seed 类内距离小
  不同信道样本距离大
  seed 不可分类

domain space:
  seed/厂家可分类
  用于生成 adapter 参数
```

不要在同一个 decoder code 空间里同时追求：

```text
信道一致
厂家差异也尽量大
```

否则 decoder 输入会继续混入厂家域信息，影响固定 decoder 互操作。

## 9. “信道信息不用管，只让 adapter 学厂家信息相互转化”可行吗？

部分可行，但要换一种表述。

更准确地说，adapter 学的不是“码字里某一段厂家信息的相互转化”，而是：

```text
该厂家 latent 坐标系到 seed42 canonical 坐标系的变换
```

如果这个变换是全局的、与具体信道样本无关：

```text
z_42(x) ≈ z_i(x) W_i + b_i
```

那么 adapter 的确主要在学厂家信息：

```text
W_i, b_i 表示厂家坐标系
z_i(x) 携带信道内容
adapter_i 把信道内容从厂家坐标系翻译到 seed42 坐标系
```

这种情况下，adapter 不需要理解每个信道的物理细节，只需要学会坐标变换。

当前数据支持这个方向，因为线性映射到 seed42 的平均 R2 达到 0.970。

但是有两个限制：

### 9.1 adapter 不能完全“忽略信道信息”

adapter 的输入仍然是：

```text
z_i(x)
```

它必须对不同 `x` 输出不同的 `z_42(x)`。所以 adapter 不能只根据 vendor id 输出一个固定向量。

正确理解是：

```text
adapter 参数由厂家信息决定
adapter 前向计算作用在信道码字上
```

也就是：

```text
theta_i = f(vendor_i)
z_hat_42(x) = Adapter(z_i(x); theta_i)
```

### 9.2 陌生厂家时，必须能估计 vendor_i

对已知 seed/vendor：

```text
直接训练 adapter_i
```

是可行的。

对陌生厂家，如果没有任何校准样本或 support code，就无法知道它的 `W_i, b_i` 是什么。

如果有一组该厂家无标签 codeword：

```text
{z_i(x_1), ..., z_i(x_m)}
```

可以从分布统计中估计 vendor embedding：

```text
vendor_embedding_i = SetEncoder({z_i})
theta_i = HyperNet(vendor_embedding_i)
```

这时“只学厂家变换”的思路才比较成立。

## 10. 推荐的分离与适配方案

### 10.1 第一阶段：证明 affine adapter 上限

先不要直接做复杂生成式模型。先做最小可解释 adapter：

```text
z_hat_42 = z_i W_i + b_i
```

或 residual 形式：

```text
z_hat_42 = z_i + z_i A_i B_i + b_i
```

推荐实验：

```text
full affine: W_i, b_i
low-rank affine: rank = 8 / 16 / 32 / 64 / 128
```

如果 affine 或 low-rank affine 已经接近 MLP adapter 和重训练 decoder 上限，说明“厂家差异主要是坐标变换”，后续生成 adapter 参数会更有理论支撑。

### 10.2 第二阶段：显式 content/domain 分离

构建：

```text
content_i = C(z_i)
domain_i = V({z_i samples})
theta_i = G(domain_i)
z_hat_42 = Adapter(content_i or z_i; theta_i)
```

损失：

```text
L_code = MSE(z_hat_42, z_42)
L_recon = MSE(D_42(z_hat_42), x)
L_content = MSE(content_i(x), content_42(x))
L_domain_cls = CE(domain_i, seed_id)
L_adv = 让 content_i 不能预测 seed_id
```

目标：

```text
content 表示信道，跨 seed 一致
domain 表示厂家，用于生成 adapter
adapter 输出 canonical code，给 D_42 解码
```

### 10.3 第三阶段：生成陌生厂家 adapter

不要输入单条 code，建议输入 support set：

```text
{z_new(x_k)}_{k=1}^m
```

生成：

```text
theta_new = HyperNet(SetEncoder({z_new}))
```

优先生成小参数：

```text
low-rank affine adapter
decoder LoRA/delta
small affine adapter
```

而不是生成 hidden=2048 的 full MLP adapter。

## 11. 最终回答

### 11.1 码字里信道信息和厂家信息占比是多少？

在原始公共坐标下，按 additive ANOVA 分解：

```text
seed/厂家主效应:        72.61%
sample/信道主效应:       0.46%
seed×sample 交互残差:   26.93%
```

但这不是说信道信息只有 0.46%。它说明：

```text
信道信息没有以跨 seed 共享坐标轴的方式出现。
```

线性对齐到 seed42 后：

```text
sample/信道主效应:      98.63%
seed/厂家主效应:         0.00045%
交互残差:                1.37%
```

这说明信道信息是完整且强烈存在的，只是被不同 seed 的坐标系包裹了。

### 11.2 当前更像 additive 还是 multiplicative/coordinate transform？

更像：

```text
coordinate transform / affine transform
```

不支持简单 additive bias。

证据：

```text
bias-only R2 平均为 -0.933
linear transform R2 平均为 0.970
```

### 11.3 是否应该让不同 seed 的信道信息一致、厂家信息不一致？

如果做 content/domain 分离，是的：

```text
content: 同一样本跨 seed 一致
domain: 不同 seed 可区分
```

但 adapter 输出给 decoder 的 canonical code 里，不应该继续保留强厂家差异。

厂家信息应进入：

```text
adapter 参数
vendor embedding
low-rank delta/LoRA 参数
```

而不是留在 `D_42` 的输入里。

### 11.4 只让 adapter 学厂家信息相互转化，可行性高吗？

如果“厂家信息”指的是：

```text
该厂家 code 坐标系到 seed42 code 坐标系的变换
```

那么可行性较高。当前线性对齐 R2 平均 0.970，说明这个方向很有希望。

如果“厂家信息”指的是：

```text
从单条 code 中抽出一个厂家字段，再生成通用 adapter
```

那可行性低，因为单条 code 中信道内容和厂家形式不可辨识。

最稳的路线是：

```text
support set 估计厂家域
生成 low-rank/affine adapter 参数
adapter 把该厂家 code 映射到 seed42 canonical code
```

也就是：

```text
{z_i(x_k)} -> vendor embedding -> adapter_i
z_i(x) -> adapter_i -> z_42(x) -> D_42 -> x_hat
```

这比直接让 adapter 在单条 code 上“识别并转换厂家信息”更合理。

