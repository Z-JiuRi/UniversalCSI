# 整体矩阵条件生成 LoRA 的技术路线

## 背景

用户澄清：不是用单个样本的 compressed code 生成一套样本级 LoRA，而是使用整个 encoder 输出矩阵作为条件：

```text
C = Encoder(X) ∈ R^{N x compressed_dim}
```

其中：

```text
X: 某个数据集 / 场景 / 厂家 / encoder 对应的一组 CSI 样本
C: 该 domain 下所有样本的 compressed code 矩阵
```

目标是：

```text
C
  -> condition encoder
  -> domain embedding
  -> LoRA generator
  -> decoder LoRA
```

这和“单样本 code_i -> LoRA_i”不是一回事。前者是 domain-level / dataset-level / vendor-level adaptation，方向是合理的。

## 与单样本 LoRA 的区别

不推荐的形式：

```text
code_i
  -> generator
  -> LoRA_i
  -> decoder(code_i; LoRA_i)
```

问题：

- decoder 参数每个样本都变化。
- generator 容易成为第二个 decoder。
- 样本瞬时信道信息和 domain / vendor 差异混在一起。
- 推理开销和部署复杂度高。
- 泛化解释困难。

推荐的形式：

```text
C_support = {code_1, ..., code_K}
  -> domain embedding
  -> LoRA_domain

for code_q in query samples:
    H_hat_q = Decoder(code_q; LoRA_domain)
```

这表示：

```text
一组 calibration compressed codes 生成一套 domain-level LoRA，
该 LoRA 在该 dataset/vendor/session 内保持固定。
```

## CCPG 项目分析

用户之前的 CCPG 项目路径：

```text
/home/z-jiuri/workspace/Huawei/CCPG
```

该项目本质是条件 LoRA 参数生成模型：

```text
condition features
  -> PerceiverResampler
  -> DiT denoiser
  -> GaussianDiffusion
  -> LoRA tokens
```

核心代码位置：

- `data/dataloader.py`
  - 读取 `cond`、`mask` 和 LoRA 矩阵 `a1,b1,a2,b2`。
  - 将 LoRA 矩阵 flatten 后 token 化。
  - 为 token 附加 `layer_ids` 和 `matrix_ids`。
- `models/resampler.py`
  - 使用 `PerceiverResampler` 将 `(cond_len, cond_dim)` 条件矩阵压缩为固定长度 latent condition tokens。
- `models/dit.py`
  - DiT denoiser 使用 self-attention、cross-attention、MLP 和 AdaLN。
  - 条件 latent tokens 通过 cross-attention 影响 LoRA token 去噪。
- `models/ddpm.py`
  - Gaussian diffusion 训练 LoRA token 分布，支持 `eps/x/v` prediction。

CCPG 里的关键路径可以概括为：

```text
cond: (cond_len, cond_dim)
  -> PerceiverResampler
  -> cond_feats: (latent_cond_len, hidden_dim)
  -> DiT cross-attention
  -> generated LoRA token sequence
```

这与 UniversalCSI 中想做的：

```text
compressed code matrix: (N, compressed_dim)
  -> condition encoder
  -> generated decoder LoRA
```

在思想上是一致的。

## CCPG 中值得保留的设计

### 1. 整体矩阵作为条件

CCPG 不用单个 scalar 或单个样本作为条件，而是用一个条件矩阵：

```text
cond: (cond_len, cond_dim)
```

这对 UniversalCSI 很有参考意义。对于多数据集 / 多厂家泛化，单个 code 不足以描述 domain 分布，而一组 code 的矩阵可以包含：

```text
均值
方差
相关结构
流形形状
encoder 输出尺度
domain 特征
```

### 2. 条件压缩器

CCPG 的 `PerceiverResampler` 用 latent queries 通过 cross-attention 压缩条件矩阵，这比简单 flatten 更合理。

抽象为：

```text
variable or fixed length condition set
  -> fixed length latent condition tokens
```

UniversalCSI 也需要类似模块。

### 3. 结构化 LoRA token 生成

CCPG 将多个 LoRA 矩阵 token 化，并附加：

```text
layer_ids
matrix_ids
```

这适合后续 UniversalCSI 生成多层 LoRA，例如：

```text
fc_decoder LoRA
Transformer FFN LoRA
attention projection LoRA
```

每个 target token 可以知道自己属于哪一层、哪一种矩阵。

### 4. 先生成 LoRA 而不是完整模型

CCPG 生成的是 LoRA 参数，而不是完整模型权重。

这与 UniversalCSI 的 BS 端部署逻辑一致：

```text
fixed base decoder W0
small generated adapter ΔW
```

## CCPG 中不能直接照搬的部分

### 1. 条件矩阵的行是否有顺序

CCPG 的 `PerceiverResampler` 对条件 token 加了 learned positional embedding：

```python
x = x + self.pos_embbed[:, :cond.size(1), :]
```

如果 `cond_len` 的顺序有物理意义，这没问题。

但 UniversalCSI 中：

```text
C = (N, compressed_dim)
```

其中 `N` 是样本数。样本集合通常没有固定顺序。

因此：

```text
code_1, code_2, ..., code_N
```

应该被视为集合，而不是序列。

不推荐直接使用 order-sensitive positional embedding。

推荐：

```text
mean/std statistics
DeepSets
attention pooling without absolute sample position
Set Transformer
Perceiver without sample positional embedding
```

或至少加入：

```text
sample-order shuffle augmentation
permutation consistency loss
```

### 2. random split 不能证明泛化

CCPG 当前 dataloader 使用 `random_split`。这在 IID 验证上可用，但不能证明跨 domain 泛化。

UniversalCSI 的目标是：

```text
多数据集 / 多厂家 / 多 encoder 泛化
```

因此必须建立 group split：

```text
held-out dataset
held-out scenario
held-out encoder
held-out resolution
held-out vendor
```

如果仍使用 random split，generator 可能只是记住已见 domain 的 LoRA 分布。

### 3. 不能只拟合 A/B token

CCPG 主要训练目标是 LoRA token diffusion loss，也就是让生成的 `A/B` 坐标接近 target。

但 LoRA 的真实作用是：

```text
DeltaW = B @ A
```

而 `A/B` 分解并不唯一。

对于 UniversalCSI，最终关心的是：

```text
decoder(code; generated LoRA) 的重建 NMSE
```

因此训练和评估不应只看：

```text
generated A/B MSE
```

更应关注：

```text
downstream reconstruction loss
NMSE
DeltaW gap
static LoRA upper bound
generated LoRA gap
```

### 4. 不建议一开始使用大 DiT diffusion

CCPG 默认配置较大：

```text
latent_cond_len = 128
hidden_dim = 1024
DiT depth = 12
mlp_ratio = 4
```

UniversalCSI 初期不建议直接上这个规模。

应该先验证最小闭环：

```text
domain condition matrix
  -> lightweight condition encoder
  -> MLP generator
  -> fc_decoder LoRA
  -> query reconstruction NMSE
```

如果这个闭环无收益，复杂 diffusion / flow-matching 通常也不会稳。

## 推荐技术路线

### 总体路线

推荐路线：

```text
Stage 0: 固定数据范围和 decoder 输出协议
Stage 1: 训练 fixed base decoder
Stage 2: 验证 static per-domain LoRA
Stage 3: 用整体 code 矩阵生成 domain embedding
Stage 4: MLP 生成 fc_decoder LoRA
Stage 5: 扩展到更多 LoRA 层
Stage 6: 再引入 Perceiver / DiT / diffusion / flow-matching
```

不要一开始直接：

```text
(N, compressed_dim) -> large diffusion -> all decoder LoRA
```

## Stage 0：统一数据和输出协议

当前 UniversalCSI 应保持：

```text
数据范围: [-0.5, 0.5]
decoder 输出: return out
NMSE: 不再 -0.5
```

这一步已经基本完成。

## Stage 1：训练 fixed base decoder

推荐 base decoder：

```text
HybridDecoder:
  code_adapter
  -> Transformer global alignment
  -> reshape to CSI map
  -> lightweight CNN residual refinement
  -> return out
```

也保留 baseline：

```text
TransNetDecoder
CNNResidualDecoder
```

训练设置：

```text
先固定 shape，例如 (N, 2, 32, 32)
多 dataset / 多 encoder 训练或分组训练
不使用 LoRA generator
```

目标：

```text
得到一个稳定的 base decoder W0
```

## Stage 2：static per-domain LoRA

冻结：

```text
base decoder W0
```

对每个 domain 训练一套 LoRA：

```text
LoRA_COST2100
LoRA_WAIR_D
LoRA_DeepMIMO
LoRA_encoder_csinet
LoRA_encoder_crnet
...
```

domain 可以定义为：

```text
dataset
dataset + scenario
dataset + encoder
dataset + encoder + resolution
vendor
```

需要先清楚定义 domain，否则 LoRA 适配对象会混乱。

优先 LoRA 位置：

```text
fc_decoder / initial projection
```

原因：

```text
这是 compressed code 进入 BS decoder latent space 的入口。
```

成功标准：

```text
static LoRA 明显优于 no-LoRA
LoRA 参数量较小
不同 domain 上收益稳定
```

如果 static LoRA 都没有收益，不应继续训练 LoRA generator。

## Stage 3：用整体 code 矩阵生成 domain embedding

对每个 domain，准备 support set：

```text
C_support = Encoder(X_support) ∈ R^{K x D}
```

其中：

```text
K << N
D = compressed_dim
```

不要每次都用完整训练集 `N`。实际部署中更合理的是 calibration：

```text
新 UE / 新数据集接入时，发送 K 个 calibration codes
BS 用这 K 个 code 估计 domain embedding
```

推荐 K：

```text
K = 8, 16, 32, 64
```

### Condition encoder baseline 1：mean/std statistics

最简单、最稳：

```python
mu = C_support.mean(dim=0)
std = C_support.std(dim=0)
stats = torch.cat([mu, std], dim=-1)
z_domain = MLP(stats)
```

优点：

- 排列不变。
- 参数少。
- 解释性强。
- 很适合作为第一版 baseline。

### Condition encoder baseline 2：DeepSets

```python
z_i = phi(code_i)
z_domain = rho(mean_i(z_i))
```

优点：

- 排列不变。
- 比 mean/std 更灵活。
- 实现简单。

### Condition encoder baseline 3：Perceiver / Set Transformer

```text
C_support
  -> input projection
  -> latent queries cross-attend to codes
  -> latent domain tokens
```

注意：

```text
不要使用绝对样本位置编码，
或必须通过 shuffle augmentation 保证顺序不敏感。
```

## Stage 4：MLP 生成 fc_decoder LoRA

先只生成一层 LoRA：

```text
fc_decoder LoRA
```

例如 32x32、cr=4：

```text
input_dim = 2 * 32 * 32 = 2048
code_dim = input_dim / 4 = 512
fc_decoder: 512 -> 2048
rank r = 8

A: (r, 512)
B: (2048, r)
generated params = r * 512 + 2048 * r = 20480
```

生成器：

```text
z_domain -> MLP -> flatten(A, B)
```

这比生成整个 decoder 的 LoRA 更容易训练和分析。

## Stage 5：Episodic training

推荐训练 episode：

```text
选择一个 domain d
采样 support set S: K 个样本
采样 query set Q: M 个样本

C_S = Encoder_d(X_S)
z_d = ConditionEncoder(C_S)
LoRA_d = Generator(z_d)

C_Q = Encoder_d(X_Q)
H_hat_Q = Decoder_W0+LoRA_d(C_Q)
loss = MSE(H_hat_Q, H_Q)
```

也就是：

```text
support set 用于生成 LoRA
query set 用于评价该 LoRA
```

这比直接拟合 offline LoRA 参数更贴近最终目标。

### 可选训练目标

可以组合：

```text
L = L_reconstruction
  + lambda_lora * L_static_lora_match
  + lambda_delta * L_deltaW
```

但第一版建议只用：

```text
L_reconstruction = MSE / NMSE-related loss
```

因为最终任务是重建 CSI。

## Stage 6：扩展 LoRA 插入点

如果 `fc_decoder` LoRA 有收益，再扩展：

```text
Transformer FFN linear1 / linear2
attention q/k/v/out projections
CNN head last Conv2d
```

优先级：

```text
fc_decoder
  -> Transformer FFN
  -> attention projections
  -> CNN head
```

不要一开始全加。

## Stage 7：引入 CCPG 风格生成器

当以下条件满足时，再引入 CCPG 风格复杂生成器：

```text
static LoRA 有明显收益
MLP generator 有明显收益
MLP generator 与 static LoRA 上限仍有差距
训练 domain 数量足够
LoRA 分布可能多模态
```

可迁移结构：

```text
Set/Perceiver condition encoder
  -> latent condition tokens
  -> DiT / diffusion / flow-matching
  -> structured LoRA tokens
```

但要针对 UniversalCSI 修改：

```text
条件矩阵按集合处理
group/OOD split
downstream NMSE 评估
DeltaW / functional metrics
较小模型起步
```

## 32x32 与 64x64 的处理

不同 shape 对应不同 `input_dim`：

```text
(2, 32, 32): input_dim = 2048
(2, 64, 64): input_dim = 8192
```

如果压缩率相同：

```text
cr = 4
32x32 code_dim = 512
64x64 code_dim = 2048
```

这意味着：

```text
fc_decoder shape 不同
LoRA A/B shape 不同
generator output head 不同
```

建议：

```text
先固定 32x32 跑通完整路线。
```

然后扩展为：

```text
shared condition encoder
  -> generator_head_32
  -> generator_head_64
```

或分别训练：

```text
decoder_32 + generator_32
decoder_64 + generator_64
```

不要一开始强行一个 head 同时生成两种 resolution 的 LoRA。

## 推荐实验顺序

### Experiment 0：确认 static LoRA 上限

```text
base decoder frozen
per-domain static LoRA
fc_decoder only
```

目的：

```text
判断 LoRA 是否能适配 domain shift。
```

### Experiment 1：mean/std condition -> MLP LoRA

```text
C_support
  -> mean/std
  -> MLP
  -> fc_decoder LoRA
  -> query reconstruction
```

目的：

```text
验证整体矩阵条件是否足够生成有效 LoRA。
```

### Experiment 2：DeepSets condition encoder

```text
C_support
  -> phi(code_i)
  -> mean pooling
  -> rho
  -> LoRA
```

目的：

```text
验证比 mean/std 更灵活的 set encoder 是否提升。
```

### Experiment 3：Perceiver without sample positional embedding

```text
C_support
  -> Perceiver latent queries
  -> LoRA generator
```

目的：

```text
迁移 CCPG 的条件压缩能力，同时避免样本顺序依赖。
```

### Experiment 4：多 LoRA 层

```text
fc_decoder + FFN LoRA
```

目的：

```text
验证更大适配空间是否提升。
```

### Experiment 5：Diffusion / Flow-Matching

```text
condition tokens
  -> DiT / flow model
  -> structured LoRA tokens
```

目的：

```text
在 MLP generator 有瓶颈时提高生成能力。
```

## 评价协议

必须包含：

```text
random split: IID sanity check
held-out dataset split
held-out encoder split
held-out scenario split
held-out resolution split
```

核心指标：

```text
no-LoRA NMSE
static-LoRA NMSE
generated-LoRA NMSE
generated vs static gap
LoRA rank / parameter count
support size K sensitivity
seen-domain vs unseen-domain gap
```

如果条件生成器只在 random split 上好，在 held-out domain 上差，则不能称为泛化提升。

## 最终推荐路线

最推荐的路线是：

```text
1. 固定 32x32 数据，统一 [-0.5,0.5] 坐标。
2. 训练 HybridDecoder base W0。
3. 冻结 W0，训练每个 domain 的 static fc_decoder LoRA。
4. 用 K 个 support compressed codes 的 mean/std 生成 domain embedding。
5. 用 MLP 生成 fc_decoder LoRA。
6. 用 query reconstruction NMSE 训练和评估。
7. 换 DeepSets / Perceiver condition encoder。
8. 扩展到 FFN LoRA。
9. 最后再迁移 CCPG 的 DiT diffusion / flow-matching。
10. 32x32 跑通后，再扩展 64x64，并使用 resolution-specific generator head。
```

一句话总结：

```text
CCPG 的“整体矩阵条件生成 LoRA”方向是合理的；
UniversalCSI 应将其改造成 permutation-invariant domain condition encoder
+ episodic query reconstruction loss
+ static LoRA upper-bound validation
的框架，而不是直接照搬 order-sensitive Perceiver + random split + A/B token diffusion。
```

