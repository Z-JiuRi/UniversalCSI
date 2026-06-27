# 全局厂家变换与固定 Decoder Adapter 策略分析

## 结论

你的理解基本是对的，但需要把“厂家信息”和“信道信息”的位置说清楚。

当前多 seed 码字更符合：

```text
z_i(x) = 第 i 个厂家/seed 的 latent 坐标系下的信道表示
```

而 adapter 要学的是：

```text
z_i(x) -> z_42(x)
```

也就是：

```text
厂家 i 的 latent 坐标系 -> seed42 decoder 能读懂的 canonical 坐标系
```

这个变换在当前数据上很接近“全局、样本无关”的变换。也就是说，对一个 seed/vendor，不需要每个信道样本单独学一个变换；一个固定的 adapter 就可以作用于该 seed/vendor 的所有码字。

因此最终部署形式应该是：

```text
固定 E_i
固定 D_42
只更新或生成 Adapter_i

z_i = E_i(x)
z_hat_42 = Adapter_i(z_i)
x_hat = D_42(z_hat_42)
```

这个方向是合理的。

但有一个关键边界：

```text
如果陌生厂家有一批码字 support set，可以推断该厂家坐标系；
如果只有单条码字，或者没有任何 support set，很难可靠推断完整 adapter。
```

## 1. 变换是不是全局的、与具体信道样本无关？

用现有 59 个 seed 的 `train_code.pt` 做了直接检验。

对每个 seed `i`，用前 2048 个配对样本拟合一个固定线性映射：

```text
z_42(x) ≈ z_i(x) W_i + b_i
```

然后在后 2048 个未参与拟合的样本上测试。

结果如下：

| 指标 | min | mean | median | max |
|---|---:|---:|---:|---:|
| 测试 R2 | 0.931 | 0.970 | 0.971 | 0.991 |
| 测试同样本 cosine | 0.989 | 0.995 | 0.996 | 0.999 |
| 4 个测试分块 R2 标准差 | 0.00027 | 0.00136 | 0.00129 | 0.00322 |

解释：

- 一个 `W_i, b_i` 在未见过的信道样本上仍然有效。
- 分块 R2 很稳定，说明不是只对某一段样本有效。
- 同样本 cosine 接近 1，说明映射后的码字方向非常接近 seed42 code。

所以当前证据支持：

```text
厂家/seed 差异主要是全局 latent 坐标系变换
```

而不是：

```text
每个信道样本都需要一个不同的厂家变换
```

这对 adapter 方案是好消息。

## 2. “信道一致、厂家差异大”这个目标怎么理解？

你说的：

```text
追求信道一致，厂家差异也尽量大
```

可以成立，但最好分到两个不同空间里。

### 2.1 Content space

这个空间用于 decoder 解码，应该尽量去厂家化：

```text
content_i(x) ≈ content_j(x)
```

尤其应该满足：

```text
Adapter_i(z_i(x)) ≈ z_42(x)
```

在这个空间里，同一个信道样本跨 seed 应该尽量近，不同信道样本应该可区分。

### 2.2 Domain / vendor space

这个空间用于表示厂家差异，应该能区分 seed/vendor：

```text
vendor_embedding_i != vendor_embedding_j
```

这个 vendor embedding 不直接喂给 decoder，而是用于生成 adapter 参数：

```text
vendor_embedding_i -> theta_adapter_i
```

所以更合理的结构是：

```text
support codewords of vendor i
    -> Vendor Encoder
    -> vendor embedding_i
    -> HyperNet
    -> Adapter_i 参数

single codeword z_i(x)
    -> Adapter_i
    -> z_hat_42(x)
    -> D_42
```

也就是说：

```text
厂家差异应该大，但大在 vendor embedding / adapter 参数空间；
decoder 输入的 canonical code 里，厂家差异应该小。
```

## 3. 和 LDA 的关系

这个问题确实有点像 LDA，但不能直接把 seed 当成 decoder code 空间里的类别。

如果把“信道样本”当类别，目标是：

```text
同一信道样本跨 seed 距离小
不同信道样本距离大
```

这对应 content alignment，类似：

```text
supervised contrastive learning
triplet loss
metric learning
```

如果把“厂家/seed”当类别，目标是：

```text
同一厂家靠近
不同厂家分开
```

这对应 domain/vendor embedding。

因此不是在同一个空间里同时做两个目标，而是：

```text
content space:
  类内 = 同一信道跨厂家
  类间 = 不同信道
  厂家不可分

vendor space:
  类内 = 同一厂家不同码字
  类间 = 不同厂家
  用来生成 adapter
```

如果直接让 decoder 输入码字同时“信道一致”和“厂家差异大”，会有冲突。因为固定 `D_42` 希望输入尽量像 `z_42`，不希望里面还保留强厂家域偏移。

## 4. Adapter 只学厂家差异转化是否可行？

可行，但精确定义应该是：

```text
Adapter 学厂家坐标系到 seed42 坐标系的变换
```

不是：

```text
Adapter 从码字里抽出一个厂家字段，然后只改这个字段
```

因为现有证据显示，厂家信息不是一个独立字段，而是整个坐标系：

```text
z_i(x) = T_i(channel)
```

因此 adapter 的参数表示厂家：

```text
theta_i ≈ T_i -> T_42 的变换
```

adapter 的输入仍然携带信道：

```text
z_i(x)
```

adapter 前向输出对应信道的 canonical code：

```text
Adapter_i(z_i(x)) ≈ z_42(x)
```

所以可以说：

```text
adapter 参数学厂家差异
adapter 计算过程保留并转换信道内容
```

这正是你想要的“不动 encoder/decoder，只更新 adapter 参数”。

## 5. 陌生厂家有完整码字，能不能直接做？

要分两种情况。

### 5.1 有一一对应的 seed42 code 或 reference code

如果陌生厂家的完整码字和 seed42 encoder 的码字是一一对应的：

```text
{z_new(x_k), z_42(x_k)}
```

那么问题就很简单。可以直接估计：

```text
z_42 ≈ z_new W + b
```

这就是校准，只是不需要动 encoder/decoder。

此时甚至不一定需要生成式模型，直接用最小二乘或训练一个小 adapter 就可以：

```text
W, b = argmin MSE(z_new W + b, z_42)
```

然后部署：

```text
D_42(z_new W + b)
```

这条路线成功概率很高，因为当前实验已经证明线性 R2 平均 0.970。

### 5.2 只有陌生厂家自己的完整码字，没有 seed42 配对 code

如果只有：

```text
{z_new(x_k)}
```

但没有：

```text
{z_42(x_k)}
```

那就不能直接拟合 `W, b`。

这时你想做的是：

```text
从 {z_new(x_k)} 的分布统计中推断 W_new, b_new
```

也就是：

```text
{z_new(x_k)} -> vendor embedding_new -> adapter_new
```

这个目标有可行性，但难度比有配对 code 高得多。因为不同厂家的码字分布确实有明显统计差异，可以被 100% 分类；这些统计差异可以作为生成 adapter 的线索。

但它不是必然成功，因为：

- `z_new` 中混有信道样本分布和厂家坐标系。
- 如果陌生厂家的坐标系超出训练 seed 分布，generator 会外推失败。
- 只看 `z_new` 分布，无法唯一确定它到 seed42 的精确方向对应关系。

因此更稳的做法是让 generator 输出低维 adapter，并允许少量自监督/无标签约束。

## 6. 为什么“完整码字里有信道信息和厂家信息，所以只变厂家信息”还不够？

关键在于：

```text
厂家信息不是可从单条码字中直接剥离的一段向量。
```

它更像一个坐标系：

```text
同一个 h，在厂家 i 下写成 z_i；
同一个 h，在 seed42 下写成 z_42。
```

这类似同一个三维点可以用不同坐标系表达：

```text
坐标值变了，但物理点没变。
```

adapter 做的是坐标变换：

```text
坐标系 i 的坐标 -> 坐标系 42 的坐标
```

而不是只修改一个“厂家字段”。

所以你的目标应该表述为：

```text
用陌生厂家的一组码字估计它的 latent 坐标系；
生成该坐标系到 seed42 坐标系的 adapter；
adapter 对每条码字做坐标变换；
固定 D_42 解码。
```

这个表述是合理且自洽的。

## 7. 推荐实验路线

### 7.1 已知 seed：先验证全局 adapter 上限

对每个已知 seed：

```text
train split: 学 W_i, b_i
test split: D_42(z_i W_i + b_i) 评估 NMSE
```

这一步回答：

```text
线性 code 对齐的高 R2 是否能真正转化为 decoder NMSE
```

如果 NMSE 接近 seed42 或原 seed 自身上限，说明全局坐标变换假设足够强。

### 7.2 已知 seed：比较 adapter 参数化

按参数量从小到大：

```text
bias-only
diagonal scale+bias
low-rank residual affine
full affine
small MLP
decoder LoRA/delta
```

其中重点是：

```text
low-rank residual affine:
z_hat = z + z A B + b
```

因为它既符合“全局坐标变换”，又适合后续生成参数。

### 7.3 陌生 seed：meta-learning / hypernetwork

训练任务：

```text
输入: seed i 的 support code set {z_i}
输出: Adapter_i 参数
监督: Adapter_i(z_i_query) ≈ z_42_query
```

训练时做 leave-one-seed-out：

```text
训练 seed: 58 个
测试 seed: 1 个完全未见
```

这样才能验证陌生厂家泛化，而不是记忆 seed。

### 7.4 最终部署结构

部署时：

```text
陌生厂家 UE encoder 固定
BS 端 seed42 decoder 固定
收集该厂家一批 codeword support set
generator 生成 Adapter_new
后续所有 codeword 经过 Adapter_new 再给 D_42
```

形式：

```text
support set:
  {z_new(x_k)}
      -> SetEncoder
      -> vendor embedding
      -> HyperNet
      -> theta_adapter_new

query:
  z_new(x)
      -> Adapter(theta_adapter_new)
      -> z_hat_42(x)
      -> D_42
      -> x_hat
```

## 8. 对你当前想法的判断

你的最终目标：

```text
不动 encoder
不动 decoder
只更新 adapter 参数
让不同 seed/vendor 的码字变换到 seed42 decoder 能认识的码字
```

这个目标是合理的，而且当前码字分析支持这个方向。

最强证据是：

```text
全局线性变换 z_i -> z_42
测试 R2 平均 0.970
同样本 cosine 平均 0.995
分块 R2 非常稳定
```

这说明 adapter 学到的主要可以是厂家级全局变换。

但是要避免一个误区：

```text
不要把厂家信息理解成码字里某个可以单独替换的字段。
```

更准确是：

```text
厂家信息 = latent 坐标系
adapter = 坐标系翻译器
信道信息 = 被翻译的内容
```

所以最终方案应该是：

```text
用 support set 估计厂家坐标系
生成或训练 Adapter_i
Adapter_i 对每条 code 做坐标变换
固定 D_42 解码
```

这条路线比“让单条码字自己告诉模型怎么生成完整 adapter”更可靠。

