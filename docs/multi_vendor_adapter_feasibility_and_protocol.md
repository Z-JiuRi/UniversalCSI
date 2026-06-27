# 多厂家码字分离、无校准泛化与公共 Code 协议分析

## 背景

当前设定是：

```text
UE 端: 不同厂家各自训练 encoder
BS 端: 只保留一个固定不变的 decoder，例如 seed42 decoder
目标: 让不同厂家 encoder 输出的 codeword 都能被固定 BS decoder 正常解码
```

现有实验已经说明：

```text
E_seed_i(x) -> D_seed42
```

直接拼接几乎不可用。原因不是 `E_seed_i` 没学到信道，而是不同 seed/厂家学到了不同 latent 语言。码字同时包含：

```text
信道样本信息
seed/厂家域信息
```

更准确的表达是：

```text
z_i = T_i(h)
```

其中：

- `h` 是信道样本的底层信息。
- `T_i` 是第 `i` 个厂家/seed 训练出来的表示坐标系。
- 固定 decoder `D_42` 只会解读 `T_42(h)`。

因此多厂家适配的本质是：

```text
把 z_i 翻译到 D_42 能读懂的 canonical code space
```

也就是：

```text
adapter_i(z_i) ≈ z_42
```

## 1. 能不能分离码字中的信道信息和厂家信息？

可以尝试，但要明确：这不是一个天然可辨识的问题。

码字不是简单的：

```text
z = channel_part + vendor_part
```

而更像：

```text
z_i = T_i(channel)
```

厂家信息不是一个独立字段，而是体现在整个坐标系、尺度、均值、方向和分布结构中。因此“分离”更现实的做法不是把码字拆成两个物理独立向量，而是学习两个表示：

```text
content code:  尽量只保留信道信息，跨厂家相近
domain code:   表示厂家/seed 的风格、坐标系或映射信息
```

目标形式可以写成：

```text
z_i -> content_i, domain_i

content_i(x) ≈ content_j(x)       对同一个信道样本，跨厂家一致
domain_i    能区分厂家/seed       表示厂家差异
```

### 1.1 有配对样本时：可行性较高

如果有同一批 CSI 样本经过不同厂家 encoder 的码字：

```text
z_i(x_k), z_j(x_k)
```

那么可以显式构造监督信号。

最直接的目标：

```text
content_i(x_k) ≈ content_j(x_k)
domain_i(x_k)  可预测 seed/vendor
```

同时要求：

```text
adapter(content_i, domain_i) -> z_42
D_42(adapter(...)) -> x
```

可以用以下损失：

```text
L_recon = MSE(D_42(z_hat_42), x)
L_code  = MSE(z_hat_42, z_42)
L_align = MSE(content_i(x), content_42(x))
L_domain_cls = CE(domain_i, vendor_id)
L_adv = 让 content_i 无法预测 vendor_id
```

其中：

```text
z_hat_42 = F(content_i, domain_i)
```

解释：

- `L_align` 让信道内容表示跨厂家靠近。
- `L_domain_cls` 让 domain 表示保留厂家信息。
- `L_adv` 通过梯度反转或 adversarial classifier，逼迫 content 去厂家化。
- `L_recon/L_code` 保证分离后的表示仍然能用于 seed42 decoder。

### 1.2 无配对样本时：难度明显上升

如果没有同一样本在多个厂家 encoder 下的码字，只看到：

```text
厂家 A 的一批 code
厂家 B 的另一批 code
```

那就很难知道哪些差异是信道内容，哪些差异是厂家风格。

这种情况下只能做弱约束，例如：

```text
分布对齐
domain adversarial training
cycle consistency
contrastive/self-supervised structure preservation
```

但这些都不能保证真正的样本级语义对齐。对于 CSI 重建这种高精度任务，风险很大。

### 1.3 当前实验提示：先做 affine/low-rank 分离很值得

前面的码字分析显示，不同 seed 到 seed42 的 code 存在很强线性可对齐性：

```text
z_42 ≈ z_i W_i + b_i
```

因此一个很自然的分离方式是：

```text
vendor-specific part: W_i, b_i
channel-specific part: aligned code z_i W_i + b_i
```

这不是显式把单个 code 拆成两段，而是把厂家信息放进 adapter 参数里，把对齐后的 code 作为公共信道表示。

推荐先做：

```text
Affine adapter:
z_hat = z W_i + b_i

Low-rank affine adapter:
z_hat = z + z A_i B_i + b_i
```

如果这种简单结构已经接近性能上限，说明厂家差异主要是坐标系变换，没必要先做复杂的 content/domain disentanglement。

## 2. 无校准样本、生成式模型读码字生成 adapter 参数，能否泛化到陌生厂家？

这里要区分两个目标。

### 2.1 目标 A：给定陌生厂家的少量码字样本，生成 adapter

如果生成式模型可以读取陌生厂家的多条 codeword，例如：

```text
{z_new(x_1), z_new(x_2), ..., z_new(x_m)}
```

并据此生成 adapter 参数：

```text
G({z_new}) -> adapter_new
```

这个目标有一定可行性。

因为多条码字样本可以暴露厂家 code 分布的统计特征，例如：

```text
mean
covariance
norm scale
principal directions
higher-order moments
```

这些统计特征可以帮助 generator 推断该厂家 latent 坐标系的大致变换。

更合理的输入不是单条 code，而是一组 support code：

```text
support set -> set encoder / transformer / DeepSets -> vendor embedding
vendor embedding -> adapter weights
```

形式：

```text
e_vendor = SetEncoder({z_new})
theta_adapter = HyperNet(e_vendor)
z_hat = Adapter(z_new; theta_adapter)
```

这个目标的成功概率取决于训练时见过多少种厂家/seed 分布。如果训练只包含几十个 seed，泛化到全新厂家会有不小风险，但比完全无信息的单条码字更现实。

### 2.2 目标 B：只读取单条陌生码字，直接生成完整 adapter

如果输入只有一条 codeword：

```text
G(z_new(x)) -> adapter_new
```

并希望这个 adapter 能适配该厂家的所有信道样本，这个目标非常难，成功概率低。

原因是单条 code 同时包含：

```text
该样本的信道内容
厂家/seed 的表示风格
```

从一条样本里很难判断：

```text
哪些维度变化来自 channel
哪些维度变化来自 vendor
```

这存在不可辨识性。比如同一个 code 分布中的某个方向，可能是：

- 某个信道结构变化；
- 某个厂家坐标系偏移；
- 两者混合。

没有多样本统计或配对校准时，模型很难可靠分离。

因此：

```text
单条 code -> 完整 adapter
```

更像是在记忆训练过的厂家模式，而不是真正泛化到陌生厂家。

### 2.3 目标 C：没有任何陌生厂家码字，直接生成 adapter

如果连陌生厂家的一批 codeword 都没有，只知道它是一个没见过的厂家，那么无法生成有意义的厂家 adapter。

因为没有任何信息可以识别该厂家的 latent 坐标系。

这时除非训练阶段已经强制所有厂家使用公共 code 协议，否则固定 decoder 没有依据去解码未知 encoder 的输出。

### 2.4 对最终可实现概率的判断

按可行性从高到低排序：

| 目标 | 可实现概率 | 说明 |
|---|---:|---|
| 见过厂家，生成或查表 adapter | 高 | 本质是多任务/条件生成 |
| 陌生厂家 + 少量无标签 codeword support set | 中等 | 可利用分布统计，但依赖训练 seed 覆盖度 |
| 陌生厂家 + 少量配对校准样本 | 较高 | 可直接估计 adapter 或 fine-tune |
| 陌生厂家 + 单条 codeword | 低 | 信道内容和厂家风格不可辨识 |
| 陌生厂家 + 无任何 code/sample | 基本不可行 | 没有识别厂家坐标系的信息 |

如果你的最终目标是：

```text
完全没见过的厂家
只读取它输出的 codeword
无需 paired calibration
直接生成 adapter
```

那么需要把目标改成：

```text
读取一组该厂家 codeword，而不是一条
生成低维/低秩 adapter，而不是生成 full MLP adapter
训练时使用大量 seed/厂家任务做 meta-learning
```

这样可实现性会明显提高。

### 2.5 更推荐的生成式 adapter 形式

不要让 generator 输出 2.1M 的 full MLP adapter 参数。

更推荐：

```text
输出 affine adapter:
W, b

或输出 low-rank adapter:
A, B, b

或输出 decoder LoRA/delta:
LoRA matrices for selected decoder layers
```

如果 code_dim=512：

```text
full affine: 512*512 + 512 = 262,656
rank-16 low-rank: 512*16 + 16*512 + 512 = 16,896
rank-32 low-rank: 33,280
rank-64 low-rank: 66,048
```

这比当前 hidden=2048 MLP adapter：

```text
2,100,736
```

更适合参数生成，也更不容易过拟合训练 seed。

## 3. 不允许校准数据时，训练阶段如何约束公共 code 协议？

如果 BS 端 decoder 固定不变，而且部署后不希望对陌生厂家做校准，那么唯一稳妥路线是在训练阶段就定义一个公共 code 协议。

也就是所有 UE encoder 都必须输出 decoder 能理解的 canonical code：

```text
E_vendor(x) -> z_canonical
D_shared(z_canonical) -> x
```

这样 decoder 不需要知道厂家是谁。

### 3.1 方案一：固定 reference decoder，训练所有 encoder 对齐它

先训练一个 reference autoencoder：

```text
E_ref, D_ref
```

然后固定 `D_ref`，训练每个厂家 encoder：

```text
E_i(x) -> D_ref -> x_hat
```

训练目标：

```text
L = MSE(D_ref(E_i(x)), x)
```

这样每个 encoder 被迫输出 `D_ref` 能读懂的 code。

可以再加入 teacher code：

```text
z_ref = E_ref(x)
L_code = MSE(E_i(x), z_ref)
```

总损失：

```text
L = L_recon + lambda_code * L_code
```

优点：

- 最贴合固定 BS decoder 场景。
- 部署时不需要 adapter。
- code 空间从训练开始就被 canonical 化。

缺点：

- 各厂家必须接受同一个 reference decoder/code 协议。
- 如果厂家 encoder 架构差异很大，训练可能更难。

### 3.2 方案二：共享 decoder 联合训练多个 encoder

同时训练多个厂家 encoder 和一个共享 decoder：

```text
E_1, E_2, ..., E_N, D_shared
```

目标：

```text
L = sum_i MSE(D_shared(E_i(x)), x)
```

可以加入 code 一致性：

```text
L_align = sum_i MSE(E_i(x), mean_j E_j(x))
```

或选择一个 teacher：

```text
L_align = MSE(E_i(x), E_ref(x))
```

总目标：

```text
L = L_recon + lambda_align * L_align
```

优点：

- decoder 从一开始就见过多种 encoder 分布。
- 比固定单一 reference decoder 更灵活。

缺点：

- 如果不加 code 对齐，共享 decoder 可能仍然隐式学习多域输入。
- 如果部署时出现训练外厂家，仍可能 OOD。

### 3.3 方案三：厂家不变的 content code + 厂家对抗去域

训练 encoder 输出：

```text
z = E_i(x)
```

要求 `z` 能重建信道，但不能预测厂家：

```text
L_recon = MSE(D_shared(z), x)
L_vendor = CE(C(z), vendor_id)
```

通过 gradient reversal 训练：

```text
encoder 让 vendor classifier 失败
classifier 尽力预测 vendor
```

总目标：

```text
min_encoder,decoder L_recon - lambda_adv * L_vendor
min_classifier L_vendor
```

目的：

```text
让 z 尽量只含信道信息，不含厂家信息
```

优点：

- 明确压制 code 中的厂家域信息。

缺点：

- 对抗训练不稳定。
- 去掉厂家信息后不一定保证 decoder 可读，还需要重建和 code 对齐约束。

### 3.4 方案四：分布规范化和 code 统计协议

规定每个 encoder 输出必须满足统一统计：

```text
mean(z) ≈ 0
std(z) ≈ 1
cov(z) ≈ I
```

或更强的：

```text
z follows N(0, I)
```

损失包括：

```text
L_mean = ||mean(z)||^2
L_cov = ||cov(z) - I||^2
```

也可以使用 VAE-style KL：

```text
KL(q(z|x) || N(0, I))
```

优点：

- 降低不同厂家 code 分布的均值/尺度差异。
- 有利于固定 decoder 泛化。

缺点：

- 只对齐分布，不保证同一样本语义对齐。
- 对高精度重建可能损伤性能。

### 3.5 方案五：公共 codebook / 量化协议

引入共享 codebook：

```text
z_continuous -> quantize -> code indices
```

所有厂家 encoder 必须输出同一个 codebook 的索引或同一套量化向量。

类似：

```text
VQ-VAE codebook
product quantization
learned vector quantization
```

优点：

- 协议最明确。
- BS decoder 接收的是公共离散符号，而不是任意连续 latent。

缺点：

- 训练复杂。
- 量化可能带来性能损失。
- codebook 设计需要非常谨慎。

### 3.6 推荐优先级

如果目标是论文实验，建议优先做：

```text
固定 D_ref + 多 seed encoder 对齐
```

也就是：

```text
先训练 seed42 autoencoder 得到 E_42, D_42
固定 D_42
训练其他 seed encoder，使 D_42(E_i(x)) 重建 x
可选加入 MSE(E_i(x), E_42(x))
```

这最直接证明：

```text
只要训练阶段定义公共 code 协议，多厂家 UE encoder 可以共享一个 BS decoder
```

然后再做 post-hoc adapter/generator，作为解决“已有厂家模型未遵守协议”的补救方案。

## 总体建议

### 对问题 1 的建议

可以做“有效分离”，但不要假设 code 自然可拆成信道段和厂家段。

更稳的做法：

```text
用 adapter 参数承载厂家坐标系信息
用 adapter 输出承载 canonical 信道 code
```

优先实验：

```text
affine adapter
low-rank affine adapter
small MLP adapter
content/domain adversarial disentanglement
```

### 对问题 2 的建议

完全无校准、单条陌生 code 生成 adapter，成功概率低。

更可行的目标是：

```text
陌生厂家的一组无标签 codeword -> vendor embedding -> low-rank adapter 参数
```

这属于 meta-learning / hypernetwork 问题。要提高泛化概率，需要：

```text
大量训练 seed/厂家任务
低维 adapter 参数化
support set 输入
分布统计输入
对 adapter 输出做强约束
```

### 对问题 3 的建议

如果部署时真的不允许校准数据，必须在训练阶段定义公共 code 协议。

最推荐的最小方案：

```text
1. 训练 reference seed42 autoencoder
2. 固定 D_42
3. 训练其他 encoder 直接适配 D_42
4. 加 E_42 teacher code loss 让 code 空间更一致
```

这样最终部署时：

```text
任意遵守协议的 UE encoder -> D_42
```

不需要 adapter。

## 最后判断

当前多 seed 实验表明：

```text
信道信息存在，但被厂家/seed 坐标系包裹；
厂家信息强到可以 100% 分类；
直接互操作失败是必然结果；
线性对齐有效，说明 adapter/generator 方向有价值；
无校准陌生厂家泛化的关键，是从单条 code 改成 support set，并把 adapter 参数化做小。
```

如果论文目标是“陌生厂家适配”，建议把主线设计成两层：

```text
训练期公共 code 协议: 证明理想情况下多厂家共享 decoder 可行
部署后生成式 adapter: 解决未遵守协议的 legacy 厂家模型适配
```

这样逻辑最完整，也最容易解释为什么直接拼接失败、为什么 adapter 有必要、为什么 generator 需要 support code 来推断厂家域。
