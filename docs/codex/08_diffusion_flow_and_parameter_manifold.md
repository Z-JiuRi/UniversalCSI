# Diffusion、Flow-Matching 与 Decoder Adapter 参数流形分析

本文档总结后续围绕“多厂家 encoder 适配、固定 BS decoder、用 encoder 输出矩阵条件生成 decoder LoRA 参数”的讨论。重点包括：

- 在该任务中 diffusion 与 flow-matching 的适用性比较。
- 如何从“参数流形”的角度重新理解 decoder adapter 生成。
- 已经获得 LoRA 参数后，如何判断该参数集合是否低维、平滑、单峰或多模态。
- 如果不同厂家对应的 decoder adapter 差异很大，应该如何解释和处理。

## 1. 任务重新表述

当前研究目标可以表述为：

```text
不同 UE 厂家 / 设备 / encoder 产生不同压缩码分布；
BS 侧希望固定一个共享 decoder；
对 decoder 注入小规模 LoRA / adapter；
用 encoder 输出的压缩码矩阵作为条件，生成或预测 decoder adapter 参数。
```

设某个 domain / 厂家 / encoder 为 `d`，原始 CSI 样本为：

```text
X_d ∈ R^{N x 2 x Nt x Nc}
```

encoder 输出压缩码矩阵：

```text
C_d = Encoder_d(X_d) ∈ R^{N x compressed_dim}
```

实际部署或训练时更推荐使用 support 子集：

```text
C_support ∈ R^{K x compressed_dim}
```

然后通过条件编码器得到 domain embedding：

```text
z_d = ConditionEncoder(C_support)
```

再由生成器产生 decoder adapter：

```text
phi_d = Generator(z_d)
```

最终 decoder 形式为：

```text
Decoder_{W0 + phi_d}(C_query) -> reconstructed CSI
```

其中 `W0` 是固定共享 decoder，`phi_d` 是 domain-specific LoRA / adapter。

## 2. Diffusion 与 Flow-Matching 的核心差异

### 2.1 不建议一开始直接上复杂生成模型

在讨论 diffusion 和 flow-matching 之前，首先需要建立几个更基础的 baseline：

```text
1. no-LoRA 固定 decoder baseline
2. 每个 domain 单独训练 static LoRA 的 upper bound
3. DeepSets / Perceiver 条件编码器 + MLP 参数生成器
4. Flow-Matching 参数生成器
5. Diffusion 参数生成器
```

原因是 diffusion / flow-matching 是否必要，取决于 static LoRA 参数集合是否真的呈现复杂、多模态、非线性结构。如果简单 deterministic generator 已经接近 static LoRA upper bound，那么直接上大规模 diffusion 反而会增加训练和解释成本。

### 2.2 Flow-Matching 更适合作为主线候选

从任务性质看，该问题更像：

```text
条件化的结构化参数运输
```

也就是在给定 `z_d` 的情况下，把一个简单初始分布或基点运输到目标 adapter 参数：

```text
x0 ~ N(0, I)
x1 = target adapter
x_t = (1 - t) x0 + t x1
v_target = x1 - x0
v_theta(x_t, t, z_d) ≈ v_target
```

训练目标可以写成：

```text
L_flow = ||v_theta(x_t, t, z_d) - (x1 - x0)||^2
```

推理时通过少量 Euler / ODE steps 生成 adapter：

```text
x_{t+dt} = x_t + v_theta(x_t, t, z_d) dt
```

Flow-Matching 更适合该任务的原因：

- decoder adapter 通常应是稳定、低延迟、接近确定性的 domain compensation。
- LoRA 本身可以理解为固定 decoder `W0` 附近的低秩切向更新。
- 如果不同厂家差异是沿某些结构化方向变化，那么 flow 的连续向量场假设很自然。
- 采样步数通常可以少于 diffusion，便于实际部署。
- 对“从共享 decoder 到 domain-specific adapter 的连续运输”解释更直接。

推荐的 flow baseline：

```text
input:  adapter state x_t, time t, domain embedding z_d
output: velocity v_theta

x0 ~ N(0, I)
x1 = normalized target adapter
t ~ Uniform(0, 1)
x_t = (1 - t) x0 + t x1
loss = ||v_theta(x_t, t, z_d) - (x1 - x0)||^2
```

### 2.3 Diffusion 更适合作为强 baseline 或多模态建模

Diffusion 的典型训练方式是逐步加噪，再学习去噪：

```text
x_t = sqrt(alpha_t) x0 + sqrt(1 - alpha_t) eps
eps_theta(x_t, t, z_d) ≈ eps
```

或者使用 `v-prediction` / `x0-prediction` 形式。

Diffusion 的优势：

- 对复杂分布和多模态分布建模能力强。
- 已有不少“神经网络参数生成”“LoRA 参数生成”的相关工作可以参考。
- 如果同一个 condition 下存在多个功能不同但都有效的 adapter，diffusion 可以采样多个候选。
- 可以通过 classifier-free guidance、DDIM、latent diffusion 等成熟机制提升可控性。

Diffusion 的劣势：

- 推理通常需要更多采样步数。
- 在 LoRA `A/B` 坐标中直接扩散会受到参数重参数化不唯一问题影响。
- 如果任务实际接近确定性映射，diffusion 可能会学习到不必要的随机性。
- 训练、调参、评价链路比 deterministic generator 和 flow 更复杂。

因此当前建议是：

```text
Flow-Matching / Rectified Flow 作为主要研究路线；
Diffusion 作为强 baseline 或多模态场景下的扩展方案。
```

## 3. 从参数流形角度理解该任务

### 3.1 Adapter 参数不是普通欧氏向量

LoRA 参数通常写成：

```text
Delta W = B A
```

其中：

```text
A ∈ R^{r x in_dim}
B ∈ R^{out_dim x r}
```

表面上可以把所有 `A`、`B` 拼成一个大向量：

```text
phi = concat(vec(A_l), vec(B_l))_l
```

但这个坐标并不唯一。因为对任意可逆矩阵 `R`：

```text
B A = (B R)(R^{-1} A)
```

也就是说，不同的 `(A, B)` 可以对应同一个有效更新 `Delta W`。因此原始 LoRA 坐标存在 gauge redundancy。

真正影响 decoder 的不是 `A` 和 `B` 本身，而是：

```text
1. effective update space: DeltaW = B A
2. function space: Decoder_{W0 + DeltaW}(code) 的输出行为
```

因此分析和生成时应区分三层：

```text
LoRA coordinate space:      A, B
effective update space:     Delta W = B A
function space:             decoder_W0+DeltaW(code) 的重建函数
```

最终目标关心的是第三层，通常也至少应该分析第二层，而不是只分析第一层。

### 3.2 多厂家适配是共享 decoder 附近的流形运动

固定 base decoder 为 `W0`，对每个 domain 训练最优 adapter：

```text
phi_d^* = argmin_phi E_{x ~ D_d} loss(Decoder_{W0 + phi}(Encoder_d(x)), x)
```

所有 domain 的 adapter 集合为：

```text
M = {phi_d^* | d belongs to vendor/domain family}
```

这个集合很可能不是整个高维参数空间，而是一个低维、结构化的参数子流形。多厂家适配可以理解为：

```text
不同厂家 encoder 诱导不同 code distribution；
不同 code distribution 对应共享 decoder 附近不同方向的小规模补偿；
这些补偿点落在 decoder adapter 参数流形上。
```

也就是说，建议把任务表述为：

```text
用压缩码分布估计当前 encoder/domain 在共享 decoder 参数流形上的适配坐标
```

而不是简单地说：

```text
用压缩矩阵直接生成 decoder 参数
```

更推荐的概念链路是：

```text
C_support
  -> domain embedding z_d
  -> manifold coordinate alpha_d
  -> LoRA / DeltaW
  -> fixed decoder adaptation
```

对应的论文式贡献表述可以是：

```text
We model multi-vendor CSI feedback adaptation as conditional transport on a low-dimensional decoder adaptation manifold.
```

### 3.3 生成低维流形坐标可能优于直接生成完整 LoRA

如果 static LoRA 参数集合确实低维，可以先对有效更新做 PCA / SVD：

```text
DeltaW_d ≈ mean_DeltaW + Σ_i alpha_{d,i} basis_i
```

此时生成器只需要预测低维坐标：

```text
z_d -> alpha_d
```

再还原得到 adapter：

```text
alpha_d -> DeltaW_d -> LoRA / adapter update
```

这样比直接生成所有 `A/B` token 更稳：

- 参数维度更低。
- 避免部分 LoRA gauge redundancy。
- 更容易解释 domain 间差异。
- flow / diffusion 可以在低维 `alpha` 空间上做，而不是在完整参数空间上做。

推荐路线：

```text
static LoRA upper bound
-> effective DeltaW manifold analysis
-> low-dimensional coordinate generator
-> conditional flow-matching on manifold coordinates
-> diffusion baseline
```

## 4. 获得 LoRA 参数后如何分析流形性质

假设每个 domain `d`、每个训练 seed `s` 都训练得到一个 LoRA：

```text
phi_{d,s} = {A_{l,d,s}, B_{l,d,s}}_l
DeltaW_{l,d,s} = B_{l,d,s} A_{l,d,s}
```

建议主要分析 `DeltaW`，而不是只分析原始 `A/B`。

### 4.1 需要保存的实验信息

每个 adapter 样本应保存：

```text
domain_id / vendor_id
encoder_id
dataset_id
resolution: 32x32 or 64x64
compression ratio
random_seed
support codes: C_support ∈ R^{K x compressed_dim}
query codes: C_query
static LoRA: A, B
effective update: DeltaW = B A
downstream NMSE
training loss / validation loss
LoRA rank
LoRA inserted layers
```

然后将每个 adapter 向量化：

```text
v_{d,s} = concat(vec(DeltaW_1), vec(DeltaW_2), ...)
```

如果参数量太大，应分层分析：

```text
v_{l,d,s} = vec(DeltaW_{l,d,s})
```

为了避免大矩阵层支配距离，可以使用按层归一化：

```text
v_{d,s} = concat(vec(DeltaW_l) / (||W0_l||_F + eps))_l
```

这样比较的是相对扰动，而不是绝对参数规模。

### 4.2 判断是否低维

构造 adapter 矩阵：

```text
V ∈ R^{num_adapters x num_params}
```

对 `V` 中心化后做 PCA / SVD，观察累计解释方差：

```text
EVR(k) = top-k principal components explained variance
```

粗略判断标准：

```text
top 4  > 70%   很低维
top 8  > 80%   低维
top 16 > 90%   比较适合流形坐标生成
top 32 还不到 70%   参数变化比较复杂
```

但要做两种 PCA：

第一种：所有 seed 全部放入：

```text
{v_{d,s}}
```

它看总体变化，包括 domain 差异和训练 seed 差异。

第二种：先对每个 domain 取 seed 均值：

```text
bar_v_d = mean_s v_{d,s}
```

然后对 `{bar_v_d}` 做 PCA。它更专注于 domain / vendor 间的主变化。

如果 domain 均值上的 PCA 很低维，而包含所有 seed 的 PCA 不低维，说明训练随机性或 LoRA 坐标冗余可能较大。

### 4.3 判断 domain 间变化是否大于 seed 内部变化

计算每个 domain 的均值：

```text
bar_v_d = mean_s v_{d,s}
bar_v_all = mean_d bar_v_d
```

计算 domain 内部方差：

```text
within_var = mean_d mean_s ||v_{d,s} - bar_v_d||^2
```

计算 domain 间方差：

```text
between_var = mean_d ||bar_v_d - bar_v_all||^2
```

得到比值：

```text
ratio = within_var / between_var
```

解释：

```text
within_var << between_var
```

说明每个厂家 / domain 有稳定 adapter，不同 domain 之间确实存在系统性差异，适合 deterministic generator 或 flow。

```text
within_var ≈ between_var
```

说明同一个 domain 内不同 seed 的 adapter 差异已经和厂家间差异一样大，直接生成 LoRA 坐标会很难。此时应先检查 static LoRA 训练是否稳定，以及是否存在 LoRA 坐标冗余。

```text
within_var > between_var
```

说明训练噪声或参数等价性非常明显，当前 adapter 数据不适合直接作为生成目标。

### 4.4 判断是否平滑

平滑性不是只看 adapter 参数本身，而是看：

```text
code distribution space 和 adapter parameter space 是否有连续对应关系
```

先为每个 domain 构造 code distribution 表征：

```text
z_d = ConditionEncoder(C_support)
```

最简单可用统计量：

```text
z_d = concat(mean(C_d), std(C_d))
```

也可以加入协方差 sketch、低秩 PCA 特征，或者使用 DeepSets / Perceiver 得到 embedding。

然后计算两个距离矩阵：

```text
D_code[d1,d2] = distance(z_d1, z_d2)
D_lora[d1,d2] = distance(bar_v_d1, bar_v_d2)
```

再计算相关性：

```text
Spearman correlation(D_code, D_lora)
```

如果相关性高，说明：

```text
code 分布相近 -> 所需 decoder adapter 也相近
```

这就是参数流形平滑性的证据。

如果相关性低，说明仅靠 code distribution 可能无法决定 adapter，可能需要额外条件，例如：

```text
encoder architecture id
dataset/scenario id
resolution
compression ratio
SNR / channel statistics
少量原始 CSI 校准样本
```

### 4.5 插值实验

选择两个 domain 的 adapter 均值：

```text
bar_v_A, bar_v_B
```

做线性插值：

```text
v_alpha = (1 - alpha) bar_v_A + alpha bar_v_B
alpha ∈ [0, 1]
```

将 `v_alpha` 加到 decoder 上，在 domain A、domain B、混合数据或中间域数据上评估 NMSE。

可能结果：

```text
NMSE 随 alpha 平滑变化
```

说明 adapter 参数空间局部比较平滑，线性插值大致落在有效区域。

```text
两端 NMSE 好，中间 NMSE 崩掉
```

说明线性插值离开了有效流形，可能存在弯曲流形、多谷结构，或者不同 domain adapter 属于不同子流形。

```text
中间对混合数据更好
```

说明 adapter 具有可组合性，适合进一步研究流形坐标或 mixture adapter。

### 4.6 判断单峰还是多模态

多模态必须在“同一个 condition 下”判断。也就是固定：

```text
domain
encoder
dataset
support set C_support
query protocol
LoRA rank
inserted layers
training budget
```

只改变训练 seed，得到：

```text
{v_{d,1}, v_{d,2}, ..., v_{d,S}}
```

然后做：

```text
pairwise distance
PCA / UMAP 可视化
KMeans / GMM clustering
silhouette score
```

但更重要的是同时比较三种距离：

```text
parameter distance:
||DeltaW_i - DeltaW_j||

function distance:
E_c ||Decoder_{phi_i}(c) - Decoder_{phi_j}(c)||^2

performance distance:
|NMSE_i - NMSE_j|
```

三种典型情况：

#### 情况 A：单峰且稳定

```text
parameter distance 小
function distance 小
NMSE 接近
```

说明同一 condition 下基本收敛到同一个 adapter，适合 deterministic generator。

#### 情况 B：参数距离大，但函数等价

```text
parameter distance 大
function distance 小
NMSE 接近
```

这通常说明 LoRA 坐标存在等价解、gauge redundancy 或训练路径差异。此时不应直接拟合 `A/B`，而应：

```text
1. 拟合 DeltaW
2. 拟合低维 alpha 坐标
3. 使用 downstream NMSE / function loss 训练生成器
4. 对 LoRA 参数做规范化或对齐
```

#### 情况 C：真实多模态

```text
parameter distance 大
function distance 大
NMSE 都不错
```

这说明同一个 condition 下存在多个功能上不同但都有效的 adapter。此时 diffusion 或 flow 的分布建模更有价值。

特别是 diffusion 可以：

```text
sample 多个 adapter candidate
用少量 validation / calibration query 选择最佳 adapter
```

### 4.7 判断“相同 code 分布是否对应多个 adapter”

需要做 controlled experiment：

```text
固定同一个 domain
固定同一批 C_support
固定 query / train split
固定 LoRA rank 和插入层
只改变 random seed
```

训练多个 static LoRA：

```text
phi_1, phi_2, ..., phi_S
```

评估：

```text
1. 每个 adapter 的 NMSE 是否都好
2. 它们之间的 DeltaW 距离是否大
3. 它们在同一批 code 上的 decoder 输出是否不同
4. 它们是否在不同 query 子集上各有优势
```

如果多个 adapter 都好，但 decoder 输出非常接近：

```text
不是实质多模态，而是参数冗余 / 等价解
```

如果多个 adapter 都好，输出也明显不同，并且对不同 query 子集表现有差异：

```text
p(phi | C_support) 可能是多峰的
```

这种情况下，distributional generator 比单点 generator 更合理。

## 5. 不同厂家 decoder adapter 差异很大怎么办

不同厂家的 adapter 差异很大不一定是坏事。关键是分析这些差异是否有结构、是否能被 code distribution 解释。

### 5.1 差异大，但结构清晰

例如 PCA / UMAP 之后发现：

```text
vendor A, B, C 分成几个清楚簇
每个簇内部稳定
```

这说明不是混乱，而是存在多个厂家子流形。

此时推荐：

```text
condition encoder
-> vendor/domain embedding
-> mixture-of-experts generator
-> 每个 expert 负责一个 adapter 子流形
```

或者：

```text
先预测 cluster / vendor type
再在 cluster 内生成 LoRA
```

这种结构比强行用一个单峰 MLP 生成所有厂家 adapter 更合理。

### 5.2 差异大，且和 code 分布距离相关

如果：

```text
D_code 和 D_lora 高相关
```

说明 code distribution 确实携带了厂家差异。这是好事，意味着条件生成路线可行。

此时需要的是更强的条件编码器和生成器，例如：

```text
DeepSets / Perceiver condition encoder
+ Flow-Matching generator
```

或者：

```text
cluster-aware generator
+ local flow model
```

### 5.3 差异大，但和 code 分布无关

如果 adapter 差异很大，但 `D_code` 与 `D_lora` 低相关，说明仅靠 encoder 输出矩阵可能不足以决定 decoder adapter。

可能需要补充条件：

```text
encoder architecture id
encoder training recipe
dataset / scenario id
resolution
compression ratio
SNR / channel statistics
少量原始 CSI calibration 样本
少量 query 重建误差信号
```

否则生成器会倾向于学一个平均 adapter，导致所有 domain 都不够好。

### 5.4 差异大，且 seed 内部也大

如果同一个 domain 内不同 seed 的 adapter 差异也很大，说明 static LoRA 训练协议本身不稳定。

此时不应该急着上 diffusion / flow，而应先稳定 LoRA 训练：

```text
固定 base decoder
固定 LoRA rank
固定 LoRA 插入层
固定初始化策略
使用相同 support/query split
加强 weight decay
约束 LoRA norm
加入 DeltaW norm regularization
使用 early stopping
多 seed 报告均值和方差
```

否则生成模型会学习到训练噪声，而不是真实的厂家 adapter 流形。

## 6. 建议的诊断实验矩阵

建议按以下维度构建 adapter 数据集：

```text
domain d:
  多个厂家 / 数据集 / encoder / 场景

seed s:
  每个 domain 训练 5~10 个 LoRA

rank r:
  例如 4, 8, 16

support K:
  例如 8, 16, 32, 64

resolution:
  32x32 和 64x64 分开分析，必要时使用不同 head
```

需要输出的核心指标：

```text
PCA explained variance of DeltaW
within-domain variance
between-domain variance
within / between ratio
code-distance vs adapter-distance Spearman correlation
parameter-distance vs function-distance correlation
cluster silhouette score
static LoRA NMSE mean/std
linear interpolation NMSE curve
generated-to-static LoRA gap
sampling steps / latency
```

其中最关键的几个指标是：

```text
1. static LoRA upper bound
2. within / between variance ratio
3. code-distance vs adapter-distance correlation
4. function distance under same condition
```

这几个指标基本决定后续应使用 deterministic generator、flow-matching 还是 diffusion。

## 7. 模型选择判断规则

### 7.1 适合 deterministic generator 的情况

```text
PCA 显示低维
within_var 小
between_var 大
code-distance 和 adapter-distance 高相关
同一 domain 多 seed 基本单簇
function distance 小
```

推荐模型：

```text
C_support -> DeepSets / Perceiver -> z_d -> MLP -> alpha_d / DeltaW / LoRA
```

### 7.2 适合 Flow-Matching 的情况

```text
adapter 流形低维或中等维
domain 变化连续
不同厂家差异明显但有平滑结构
希望推理快
希望建模从 base distribution 到 adapter 的条件运输
```

推荐模型：

```text
C_support -> z_d
x0 -> flow_theta(x_t, t, z_d) -> adapter / alpha
```

更推荐先在低维 `alpha` 空间上做 flow：

```text
C_support -> z_d
flow over alpha
alpha -> DeltaW / LoRA
```

而不是直接在完整 `A/B` token 上做。

### 7.3 适合 Diffusion 的情况

```text
同一 condition 下存在多个功能不同但都有效的 adapter
seed 多模态明显
参数分布复杂
需要采样多个候选 adapter
有足够训练数据和计算预算
```

推荐策略：

```text
conditional diffusion over normalized adapter tokens
或 latent diffusion over low-dimensional alpha
```

如果直接复用 CCPG 风格的 DiT diffusion，需要注意：

```text
1. 不要只用随机 sample split，需要按 domain / encoder / vendor 做 OOD split
2. 不要只优化 A/B token MSE，要评估 DeltaW 和 downstream NMSE
3. condition matrix 的样本维不能随意加入有序 positional embedding
4. 需要比较 eps-prediction、v-prediction、x0-prediction
```

## 8. 推荐总体技术路线

当前最推荐的研究路线：

```text
Stage 0:
  训练 / 固定一个通用 base decoder W0

Stage 1:
  对每个 domain / vendor / encoder 训练 static LoRA
  得到 phi_{d,s}

Stage 2:
  将 LoRA 转成 DeltaW = B A
  做 PCA、within/between variance、距离相关性、插值实验、function distance

Stage 3:
  如果低维性明显，构造 adapter manifold coordinate alpha

Stage 4:
  训练 deterministic generator:
  C_support -> z_d -> alpha_d / DeltaW_d

Stage 5:
  如果 deterministic generator 和 static LoRA gap 仍明显，
  训练 conditional flow-matching

Stage 6:
  如果同一 condition 下存在真实多模态，
  再训练 diffusion baseline 或 diffusion candidate sampler
```

简化成一句话：

```text
先证明 decoder adapter 是否形成可学习的低维流形；
再决定是否需要 flow 或 diffusion；
不要一开始就在完整 LoRA A/B 坐标上直接做复杂生成。
```

## 9. 对当前项目的具体结论

针对 UniversalCSI 当前目标，比较稳妥的结论是：

```text
变化 encoder、固定 decoder、对 decoder 做 LoRA 适配是合理的。
```

但建议将研究对象从：

```text
用整个 encoder 输出矩阵直接生成 LoRA 参数
```

调整为：

```text
用 support code distribution 估计 domain embedding，
再在 decoder adaptation manifold 上预测低维坐标或生成 adapter。
```

优先级建议：

```text
1. static per-domain LoRA upper bound
2. DeltaW / function-space manifold diagnosis
3. DeepSets / Perceiver + deterministic generator
4. low-dimensional coordinate generator
5. conditional flow-matching
6. conditional diffusion baseline
```

如果后续实验发现：

```text
within_var 小、between_var 大、code-distance 与 adapter-distance 高相关
```

那么该方向非常适合写成“多厂家 CSI feedback 的 decoder adapter 流形学习”。

如果实验发现：

```text
同一 condition 下多个 seed 产生功能不同但都有效的 adapter
```

则可以进一步强调 diffusion / flow 的分布建模优势。

如果实验发现：

```text
adapter 差异大且无法由 code distribution 解释
```

则需要补充更多条件信息，或者重新定义 domain / support calibration 协议。

## 10. 最小可执行实验清单

为了尽快验证方向，建议先做以下最小实验：

```text
1. 选 3~5 个 domain / encoder / dataset setting
2. 每个 setting 训练 5 个 seed 的 static LoRA
3. 保存 A/B、DeltaW、support codes、query NMSE
4. 计算 DeltaW PCA explained variance
5. 计算 within/between variance ratio
6. 计算 code-distance vs adapter-distance Spearman correlation
7. 计算同 domain 不同 seed 的 function distance
8. 做两个 domain adapter 的线性插值 NMSE curve
```

判断标准：

```text
如果 static LoRA 显著优于 no-LoRA：
  说明 decoder adapter 有必要。

如果 within/between ratio 小：
  说明 domain adapter 稳定。

如果 PCA 低维：
  说明可以做 manifold coordinate generator。

如果 code-distance 和 adapter-distance 相关：
  说明 C_support 条件有效。

如果 parameter distance 大但 function distance 小：
  说明不要直接拟合 A/B。

如果同 condition 下 function distance 大且多个 adapter 都好：
  说明 diffusion / flow 的分布生成有价值。
```

这套分析完成后，就可以比较有把握地决定后续主线：

```text
deterministic generator
vs
flow-matching
vs
diffusion
```

