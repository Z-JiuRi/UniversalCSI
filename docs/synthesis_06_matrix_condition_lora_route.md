# 整体矩阵条件生成 LoRA：Codex 与 Gemini 合并路线

## 背景

本文件合并以下两份材料：

```text
docs/codex/07_matrix_condition_lora_route.md
docs/gemini/06_condition_processing_and_technical_roadmap.md
```

讨论对象是：在 UniversalCSI 的多数据集、多厂家、多 encoder 泛化任务中，如何使用整个 encoder 输出矩阵作为条件生成 BS decoder 的 LoRA。

用户澄清后的设定不是单样本级动态 LoRA，而是：

```text
C = Encoder(X) ∈ R^{N x compressed_dim}
```

其中：

```text
N: 某个 domain 中的样本数
compressed_dim: encoder 输出码字维度
domain: 数据集 / 场景 / 厂家 / encoder / resolution 的某种组合
```

目标是：

```text
C
  -> condition encoder
  -> domain embedding
  -> LoRA generator
  -> one LoRA for this domain
```

然后该 LoRA 用于该 domain 下的 query 样本：

```text
H_hat_q = Decoder_W0+LoRA_domain(code_q)
```

## 核心结论

整体矩阵条件是合理的，但不能直接将 `(N, compressed_dim)` flatten 后喂给 MLP 或 diffusion。

正确路线应是：

```text
support compressed codes: (K, compressed_dim)
  -> permutation-invariant condition encoder
  -> domain embedding
  -> lightweight LoRA generator first
  -> generated LoRA
  -> query reconstruction loss / NMSE
```

其中 `K` 是 calibration/support 样本数，通常远小于全量 `N`：

```text
K = 8, 16, 32, 64
```

## 为什么不能直接输入原始 `(N, compressed_dim)` 矩阵

Gemini 材料强调了三个关键问题。

### 1. N 可变

不同数据集、不同 UE 接入阶段、不同校准预算下，`N` 或 `K` 可能不同：

```text
COST2100 domain: K = 16
WAIR-D domain: K = 32
DeepMIMO domain: K = 64
```

普通 MLP 不能直接处理可变长度输入。

### 2. 样本集合无序

对于：

```text
C = {code_1, code_2, ..., code_K}
```

样本顺序通常没有语义。交换 `code_i` 和 `code_j` 不应改变生成的 LoRA。

因此 condition encoder 应满足：

```text
permutation invariance / permutation equivariance
```

不推荐：

```text
flatten(K, D) -> MLP
```

也不推荐对样本位置加入绝对 positional embedding。

### 3. 参数和显存爆炸

当 `N` 很大时：

```text
N * compressed_dim
```

可能达到几十万甚至百万维。如果直接 flatten，generator 第一层参数量会失控，且极易过拟合。

## 与单样本 LoRA 的区别

不推荐主线：

```text
code_i
  -> generator
  -> LoRA_i
  -> decoder(code_i; LoRA_i)
```

这会变成 sample-level hypernetwork，容易把样本瞬时信道信息和 domain/vendor 差异混在一起。

推荐主线：

```text
support codes C_support
  -> LoRA_domain

query codes C_query
  -> decoder with same LoRA_domain
```

这表示：

```text
一组校准码字生成一套 domain-level LoRA，
该 LoRA 在同一 domain/session 内保持固定。
```

## CCPG 经验的可迁移部分

用户之前的 CCPG 项目：

```text
/home/z-jiuri/workspace/Huawei/CCPG
```

核心路径：

```text
condition features
  -> PerceiverResampler
  -> DiT denoiser
  -> GaussianDiffusion
  -> LoRA tokens
```

值得迁移的设计：

### 1. 整体条件矩阵

CCPG 使用：

```text
cond: (cond_len, cond_dim)
```

而不是单个标量或单个样本。这和 UniversalCSI 的：

```text
compressed code matrix: (K, compressed_dim)
```

高度对应。

### 2. 条件压缩器

CCPG 的 `PerceiverResampler` 用 latent queries 通过 cross-attention 压缩条件矩阵。

这比直接 flatten 更合理，因为它能将可变长度或较长条件压缩成固定长度 latent tokens。

### 3. 结构化 LoRA token

CCPG 将 LoRA 矩阵拆成 token，并带上：

```text
layer_ids
matrix_ids
```

这适合 UniversalCSI 后续生成多层 LoRA：

```text
fc_decoder
Transformer FFN
attention projections
optional CNN head
```

### 4. 生成 LoRA 而不是完整 decoder

这与 BS 端部署一致：

```text
fixed base decoder W0
small generated ΔW_lora
```

## CCPG 不能直接照搬的部分

### 1. 去掉输入样本位置编码

CCPG 的 PerceiverResampler 对输入 condition 加 learned positional embedding。

如果 condition token 是序列或有固定语义位置，这是合理的。

但 UniversalCSI 中：

```text
C_support: (K, compressed_dim)
```

其中 K 维是样本集合，不是有序序列。

因此移植时必须：

```text
remove input positional embedding
```

或者至少加入：

```text
sample shuffle augmentation
permutation consistency loss
```

Gemini 建议将其改造成：

```text
Set Transformer Extractor
```

也就是剥离输入位置编码的 Perceiver/Set Encoder。

### 2. 不要直接用 random split 证明泛化

CCPG 当前有 random split 逻辑。它适合 IID sanity check，但不能证明跨 domain 泛化。

UniversalCSI 应采用：

```text
held-out dataset
held-out scenario
held-out encoder
held-out resolution
held-out vendor
```

否则条件生成器可能只是在插值已见 domain。

### 3. 不要只拟合 A/B token

LoRA 真正作用是：

```text
DeltaW = B @ A
```

而不是 raw A/B 坐标本身。

UniversalCSI 最终评价必须看：

```text
Decoder_W0+LoRA(code) 的重建 NMSE
```

可以记录 A/B 误差，但不能只优化或只报告它。

### 4. 不要一开始使用大 DiT diffusion

CCPG 默认模型较大。UniversalCSI 初期应该先验证最小闭环：

```text
mean/std or DeepSets
  -> MLP generator
  -> fc_decoder LoRA
  -> query reconstruction NMSE
```

只有当 MLP generator 有收益但有瓶颈，再引入 Perceiver/DiT/diffusion/flow-matching。

## 推荐 Condition Encoder

### Baseline 1：Mean/Std Statistics

最优先推荐的第一版：

```python
mu = C_support.mean(dim=0)
std = C_support.std(dim=0)
stats = torch.cat([mu, std], dim=-1)
z_domain = MLP(stats)
```

优点：

- 排列不变。
- 实现简单。
- 参数少。
- 解释性强。
- 适合作为 sanity baseline。

### Baseline 2：DeepSets

```python
z_i = phi(code_i)
z_domain = rho(mean_i(z_i))
```

优点：

- 仍然排列不变。
- 比 mean/std 更灵活。
- 适合作为第二阶段 condition encoder。

### Baseline 3：Set Transformer / Perceiver without Input Positional Embedding

```text
C_support
  -> input projection
  -> latent queries cross-attend to codes
  -> latent domain embedding
```

Gemini 建议：

```text
latent_cond_len = 1
```

即将整个 domain 压成单个：

```text
Domain_Embedding: (1, hidden_dim)
```

Codex 建议：

```text
latent_cond_len = 1 可作为第一版；
后续也可测试 latent_cond_len = 4/8，以保留更丰富 domain token。
```

但无论取多少 latent tokens，都不应使用输入样本绝对位置编码。

## 推荐 LoRA 生成目标

第一版只生成：

```text
fc_decoder / initial projection LoRA
```

原因：

- 它是 code 进入 BS decoder latent space 的入口。
- domain shift 首先体现在 code 分布与 decoder latent space 的对齐。
- 该层 LoRA 参数量可控。

例如 32x32、cr=4：

```text
input_dim = 2 * 32 * 32 = 2048
code_dim = input_dim / 4 = 512
fc_decoder: 512 -> 2048
rank = 8

A: (8, 512)
B: (2048, 8)
generated params = 20480
```

后续再扩展：

```text
Transformer FFN
attention projection
CNN head final conv
```

## 推荐训练协议：Episodic Support/Query

每个训练 episode：

```text
1. 选择一个 domain d。
2. 从 domain d 采样 support set S，大小 K。
3. 从 domain d 采样 query set Q，大小 M。
4. C_S = Encoder_d(X_S)
5. z_d = ConditionEncoder(C_S)
6. LoRA_d = Generator(z_d)
7. C_Q = Encoder_d(X_Q)
8. H_hat_Q = Decoder_W0+LoRA_d(C_Q)
9. loss = MSE(H_hat_Q, H_Q)
```

核心思想：

```text
support set 用来生成 LoRA；
query set 用来评估 LoRA 是否能泛化到同 domain 的未见样本。
```

这比直接拟合 offline LoRA 的 A/B 参数更贴近最终任务。

## 推荐阶段路线

### Stage 0：统一基础设定

```text
数据范围: [-0.5, 0.5]
decoder 输出: return out
NMSE: 不再 -0.5
```

### Stage 1：训练 base decoder

推荐：

```text
HybridDecoder:
  code_adapter
  -> Transformer global alignment
  -> reshape to CSI map
  -> CNN residual refinement
  -> return out
```

目标：

```text
得到固定 BS base decoder W0。
```

### Stage 2：验证 static per-domain LoRA

冻结：

```text
W0
```

为每个 domain 训练静态 LoRA：

```text
LoRA_COST2100
LoRA_WAIR_D
LoRA_DeepMIMO
LoRA_encoder_csinet
...
```

目的：

```text
证明低秩适配空间本身有效。
```

如果 static LoRA 无明显收益，不应继续做生成器。

### Stage 3：Mean/Std + MLP Generator

```text
C_support
  -> mean/std
  -> z_domain
  -> MLP
  -> fc_decoder LoRA
  -> query reconstruction
```

这是最推荐的第一版 generator。

### Stage 4：DeepSets / Set Transformer Extractor

替换 mean/std：

```text
C_support
  -> DeepSets or Set Transformer
  -> z_domain
  -> MLP LoRA generator
```

目标：

```text
提高 domain representation 表达能力。
```

### Stage 5：扩展 LoRA 层

从：

```text
fc_decoder only
```

扩展到：

```text
fc_decoder + Transformer FFN
```

再考虑：

```text
attention projections
CNN head final conv
```

### Stage 6：Diffusion / Flow-Matching

只有当以下条件满足时才引入：

```text
static LoRA 有收益
MLP generator 有收益
MLP generator 与 static LoRA 上限仍有差距
LoRA 分布可能多模态
domain 数量足够
```

条件输入仍应是：

```text
Domain_Embedding
```

而不是原始 `(N, compressed_dim)`。

## 32x32 与 64x64 的处理

不同 resolution 对应不同维度：

```text
(2, 32, 32): input_dim = 2048
(2, 64, 64): input_dim = 8192
```

如果压缩率相同：

```text
32x32 code_dim = 2048 / cr
64x64 code_dim = 8192 / cr
```

因此：

```text
decoder fc_decoder shape 不同
LoRA A/B shape 不同
generator output head 不同
```

推荐：

```text
先固定 32x32 跑通完整路线。
```

后续扩展：

```text
shared condition encoder
  -> generator_head_32
  -> generator_head_64
```

或：

```text
decoder_32 + generator_32
decoder_64 + generator_64
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
support size K sensitivity
LoRA rank / parameter count
seen-domain vs unseen-domain gap
```

如果只在 random split 上提升，不能声称多厂家泛化提升。

## 最终推荐路线

最推荐的执行顺序：

```text
1. 固定 32x32 和 [-0.5,0.5] 数据坐标。
2. 训练 HybridDecoder base W0。
3. 冻结 W0，训练每个 domain 的 static fc_decoder LoRA。
4. 用 K 个 support compressed codes 的 mean/std 生成 domain embedding。
5. 用 MLP 生成 fc_decoder LoRA。
6. 用 query reconstruction NMSE 训练和评估。
7. 将 mean/std 替换为 DeepSets。
8. 将 DeepSets 替换为无输入位置编码的 Set Transformer / Perceiver。
9. 扩展到 Transformer FFN LoRA。
10. 最后再迁移 CCPG 的 DiT diffusion / flow-matching。
11. 32x32 跑通后，再扩展 64x64，并使用 resolution-specific generator head。
```

一句话总结：

```text
整体矩阵条件生成 LoRA 是合理方向；
但 UniversalCSI 中应使用 permutation-invariant domain condition encoder
+ episodic query reconstruction loss
+ static LoRA upper-bound validation，
而不是直接 flatten 原矩阵，也不是直接照搬 CCPG 的 order-sensitive Perceiver、random split 和 A/B token diffusion 协议。
```

