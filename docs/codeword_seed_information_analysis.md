# COST2100/In 多 seed 码字信息分析

## 结论

对 `exps/COST2100/in/seed<seed_id>/transnet_hybrid/codewords/train_code.pt` 下的 59 个 seed 码字文件做抽样分析后，结论是：

1. 码字中明确包含信道样本信息。
2. 码字中也包含非常强的 seed/厂家域信息。
3. 不同 seed 的码字不是同一个公共坐标系下的同义表达，而是 seed 专属坐标系下的信道表示。
4. 直接把 seedA encoder 的码字喂给 seed42 decoder 失败，不是因为 seedA encoder 没学到信道，而是因为 seed42 decoder 无法解释 seedA 的码字坐标系和分布。

更具体地说，码字不是简单的：

```text
code = channel_info
```

而更接近：

```text
code_seed_i = T_i(channel_info)
```

其中 `T_i` 由随机初始化、训练轨迹和 decoder 协同学习决定。不同 seed 的 `T_i` 差异足够大，因此 seed42 decoder 只能解读 `T_42(channel_info)`，不能直接解读 `T_i(channel_info)`。

## 分析对象

分析的文件模式：

```text
exps/COST2100/in/seed<seed_id>/transnet_hybrid/codewords/train_code.pt
```

共找到 59 个 seed，每个文件形状均为：

```text
(100000, 512), dtype=float32
```

参考 seed：

```text
seed42
```

抽样设置：

```text
统计样本数: 4096
线性对齐训练样本数: 2048
对齐测试样本数: 512
```

这些抽样均使用相同样本 index，因此不同 seed 的第 `k` 行码字对应同一个训练集 CSI 样本。

## 码字分布差异

不同 seed 的码字范数和均值差异很明显：

| 指标 | min | mean | median | max |
|---|---:|---:|---:|---:|
| 样本码字 L2 norm 均值 | 22.49 | 28.96 | 28.84 | 45.57 |
| 码字均值向量 norm | 16.71 | 24.61 | 24.76 | 42.42 |
| 每维标准差均值 | 0.593 | 0.669 | 0.669 | 0.790 |
| 方差参与维度 | 489.74 | 499.04 | 500.07 | 503.45 |

解释：

- 所有 seed 的码字都不是低维塌缩表示，512 维基本都在使用。
- 但各 seed 的全局均值、范数尺度和分布位置不同。
- 这已经足以让固定 decoder 把其他 seed 的码字视为 out-of-distribution 输入。

## 码字是否包含 seed/厂家信息

用每个 seed 的前 2048 个样本估计码字中心，再在后 512 个未见过的样本上做最近中心 seed 分类：

| 测试 | 准确率 |
|---|---:|
| 随机猜测 | 1.69% |
| 最近中心分类 | 100.00% |
| cosine 最近中心分类 | 100.00% |
| 先减去同一样本的跨 seed 平均，再分类 | 100.00% |

这里最后一项很关键。它先对每个样本 index 计算跨 seed 平均码字，再把每个 seed 的码字减掉这个“共同样本成分”，然后再分类 seed。结果仍然是 100%。

这说明码字中存在非常强的 seed 专属残差模式。这个模式不是简单由信道样本差异造成的，而是和 seed/厂家域强相关。

因此，可以说：

```text
codeword 确实带有可被轻易识别的 seed/厂家域信息。
```

但这不一定表示模型显式编码了一个“seed id 字段”。更合理的解释是：每个 seed 训练出的 encoder-decoder 对形成了自己的 latent 坐标系和分布签名，seed 信息隐式存在于码字的全局坐标、均值、尺度和方向结构中。

## 码字是否包含信道信息

### 原始空间下的同一样本检索

以 seed42 的码字作为 reference，对每个其他 seed 的测试码字直接做 cosine 最近邻检索。检索目标是：同一个样本 index 的 seed42 码字是否是最近邻。

候选集大小为 512，因此随机 top-1 命中率约为：

```text
1 / 512 = 0.195%
```

实际结果：

| 指标 | min | mean | median | max |
|---|---:|---:|---:|---:|
| raw top-1 同样本检索 | 0.00% | 0.18% | 0.20% | 0.59% |
| raw 同样本 cosine | -0.073 | 0.0005 | 0.0013 | 0.083 |
| raw 非同样本 cosine | -0.071 | 0.0002 | 0.0020 | 0.080 |

解释：

- 不做对齐时，同一个信道样本在不同 seed 下的码字并不更接近。
- raw top-1 基本等于随机猜测。
- 同样本 cosine 和非同样本 cosine 几乎没有差异。

这直接解释了为什么：

```text
seed_i encoder + seed42 decoder
```

几乎没有重建能力。seed42 decoder 看到的不是它熟悉的 seed42 code 分布。

### 线性对齐后的同一样本检索

然后用 2048 个配对样本学习一个线性映射：

```text
z_seed_i -> z_seed42
```

在未参与拟合的 512 个测试样本上，再做同样本检索和 R2 评估。

结果：

| 指标 | min | mean | median | max |
|---|---:|---:|---:|---:|
| 线性对齐后 top-1 同样本检索 | 100.00% | 100.00% | 100.00% | 100.00% |
| 线性对齐 R2 | 0.929 | 0.968 | 0.970 | 0.991 |
| linear CKA | 0.855 | 0.958 | 0.966 | 0.979 |

解释：

- 线性映射后，所有 seed 都能 100% 找回同一个信道样本。
- R2 普遍很高，说明很多 seed 到 seed42 的码字差异可以被线性/仿射映射解释。
- CKA 很高，说明不同 seed 的码字表示几何中保留了相似的样本结构，只是原始坐标系不一致。

因此，可以说：

```text
codeword 明确包含信道信息。
```

只是这个信道信息被包在 seed 专属的表示坐标系中，不能被另一个 seed 的 decoder 直接读取。

## 典型 seed 对齐结果

以下均以 seed42 为 reference decoder/code 空间。

| seed | raw top-1 | raw 同样本 cosine | 线性 top-1 | 线性 R2 | CKA | 自身 best NMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.20% | -0.0126 | 100.00% | 0.948 | 0.962 | -26.601 |
| 1024 | 0.00% | -0.0288 | 100.00% | 0.982 | 0.970 | -26.037 |
| 2026 | 0.20% | -0.0271 | 100.00% | 0.990 | 0.978 | -28.207 |
| 3407 | 0.59% | 0.0572 | 100.00% | 0.958 | 0.972 | -27.562 |
| 520 | 0.39% | 0.0040 | 100.00% | 0.981 | 0.963 | -25.757 |
| 796 | 0.20% | -0.0067 | 100.00% | 0.981 | 0.950 | -27.478 |
| 31415 | 0.00% | -0.0095 | 100.00% | 0.947 | 0.974 | -26.312 |

这些结果支持同一个判断：

```text
raw code 空间几乎不可互操作；
paired linear alignment 后，信道样本结构高度可恢复。
```

## 对“码字是否同时包含信道信息和厂家信息”的回答

可以分两层理解。

### 1. 是否包含信道信息

是。

证据是：经过少量配对样本训练的线性映射后，不同 seed 的码字可以 100% 找回 seed42 中同一个信道样本，且 R2 平均约 0.968。

如果码字没有信道信息，线性映射不可能在未见过样本上稳定恢复 seed42 code。

### 2. 是否包含 seed/厂家信息

也是。

证据是：用未见过的样本做 seed 分类，准确率为 100%，随机只有 1.69%。即使先减掉同一样本的跨 seed 平均，分类准确率仍然是 100%。

这说明 seed/厂家域信息非常强，并且不是简单由样本内容造成。

### 3. 二者是什么关系

更准确的说法不是：

```text
code = 信道信息 + 厂家id
```

而是：

```text
code = 厂家/seed 专属坐标系中的信道表示
```

seed 信息不是显式字段，而是整个表示空间的坐标、尺度、均值、方向和高阶结构。信道信息存在于这个空间里，但 decoder 必须知道这个空间的“语言”才能解码。

## 对当前多厂家适配实验的启发

### 1. 直接拼接 encoder 和固定 decoder 本质上不应该工作

不同 seed 的 transnet_hybrid 即使架构完全相同，code 空间也几乎不对齐。raw 同样本检索只有随机水平。

所以：

```text
E_seed_i(x) -> D_seed42
```

失败是预期现象，不是某个 checkpoint 训练坏了。

### 2. adapter 方向是合理的，但应先做强线性 baseline

本次分析显示：

```text
z_seed_i -> z_seed42
```

存在很强的线性可对齐性。建议先加一个最简单的 affine adapter baseline：

```text
z' = z W + b
```

参数量：

```text
512 * 512 + 512 = 262,656
```

这个比当前 hidden=2048 的 MLP adapter 小很多：

```text
MLP adapter = 2,100,736
```

如果 affine adapter 的重建 NMSE 已经接近 MLP，那么说明主要问题是坐标系对齐，不需要复杂非线性 adapter。

### 3. code loss 的权重需要谨慎

因为 seed_i code 和 seed42 code 之间存在强线性关系，直接用：

```text
MSE(adapter(z_i), z_42)
```

是合理的。但如果 adapter 容量大、lr 偏高，可能把 code 推出 seed42 decoder 的有效流形。前面高 lr 实验已经看到类似现象。

建议先用稳定的小模型：

```text
affine adapter
low-rank affine adapter
small MLP adapter, hidden=128/256/512
```

再逐步增加容量。

### 4. 若目标是无校准样本的厂家泛化，仅靠固定 decoder 很难

当前证据说明每个 seed 都有非常强的域签名。若完全没有厂家校准数据，BS 端很难知道某个新 encoder 的 `T_i` 是什么。

如果允许少量校准数据，则可学习：

```text
T_i^{-1} 或 T_i -> T_42
```

如果不允许校准数据，则需要在训练阶段就约束所有厂家 encoder 使用同一个公共 code 协议，例如：

```text
shared decoder 训练
code distribution alignment
teacher code distillation
canonical codebook / quantization
contrastive sample alignment
```

## 后续建议

优先做三个实验：

1. **Affine adapter baseline**

   ```text
   z' = LayerNorm(z) W + b
   或
   z' = z W + b
   ```

   目标是验证线性对齐能否直接转化为 NMSE 提升。

2. **Low-rank affine adapter**

   ```text
   z' = z + z A B
   rank = 8 / 16 / 32 / 64
   ```

   这能把每厂家参数量从 262K 进一步降到几十 K。

3. **训练阶段加入 canonical code 约束**

   固定 seed42 encoder 产生 teacher code：

   ```text
   z_ref = E_42(x)
   ```

   训练其他厂家 encoder 时加入：

   ```text
   loss = recon_loss(D_42(E_i(x)), x) + lambda * MSE(E_i(x), z_ref)
   ```

   这比 post-hoc adapter 更接近真正的“固定 BS decoder、多厂家 UE encoder”协议。

## 总体判断

现有多个 seed 的码字同时满足：

```text
信道信息强：线性对齐后同样本检索 100%，R2 平均 0.968
厂家信息强：seed 分类 100%，随机只有 1.69%
原始互操作性弱：raw 同样本检索约 0.18%，接近随机
```

因此，当前问题的核心不是 encoder 没学到信道，而是：

```text
每个厂家/seed 学到了自己的 latent 语言；
BS 固定 decoder 只会读 seed42 这门语言。
```

多厂家适配的关键，就是把不同厂家 UE 的码字翻译到 BS 固定 decoder 能读懂的 canonical code 空间。
