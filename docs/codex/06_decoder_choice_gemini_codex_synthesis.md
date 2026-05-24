# Decoder 选型分歧与 HybridDecoder 推荐方案

## 背景

本文件总结对 `docs/gemini/` 中 decoder 选型建议的复核结果，并结合 Codex 此前给出的分析，形成一个更可执行的折中方案。

用户的长期目标是：

```text
多厂家 UE 泛化提升
```

系统设定为：

```text
不同厂家 UE -> 不同 encoder
BS 端 -> 固定 base decoder
适配机制 -> 对 decoder 注入 LoRA
LoRA 参数 -> 由 encoder 输出的 compressed code 或其统计表示生成
生成方式 -> 后续可能使用 diffusion 或 flow-matching
```

因此 decoder 选型不是普通自编码器问题，而是一个 base model 设计问题：

```text
哪个 fixed base decoder 最适合作为多 encoder / 多厂家 LoRA 适配的底座？
```

## Gemini 的主要观点

`docs/gemini/04_future_research_multi_vendor_generalization.md` 中的核心建议是：

```text
BS 固定 decoder 必须选择 Transformer Decoder。
```

理由包括：

1. CNN decoder 有较强局部感受野归纳偏置。
2. 这种归纳偏置可能和非 CNN UE encoder 产生的 code 特征冲突。
3. Transformer 更接近通用函数近似器，表达能力更自由。
4. 当前优化后的 `TransNetDecoder` 是理想起点。
5. 后续可以将 Transformer 内部 `nn.Linear` 替换为 `LoRALinear`，支持动态 LoRA 注入。

Gemini 同时强调：

```text
code_adapter 在多厂家泛化任务中不应只是 optional trick，
而应成为统一语义投影器。
```

也就是：

```text
raw code from UE encoder
  -> code_adapter
  -> unified latent condition
  -> decoder / LoRA generator
```

## Codex 的主要观点

Codex 认同以下方向：

- 固定 BS base decoder 是合理的。
- `TransNetDecoder` 是强 baseline。
- `code_adapter` 对多厂家泛化非常重要。
- 应先验证 per-encoder static LoRA，再考虑 diffusion/flow-matching。
- 最终应以 LoRA 适配能力评价 decoder，而不是只看端到端 NMSE。

但 Codex 不认同“必须 Transformer Decoder”这个绝对判断。

核心理由是：

```text
CNN decoder 的归纳偏置作用在输出 CSI map 的空间结构上，
不一定等价于假设输入 code 来自 CNN encoder。
```

需要区分：

```text
CNN encoder bias: 影响 code 如何生成
CNN decoder bias: 影响重建图如何被局部修正
```

即使 UE encoder 是私有模型或 Transformer，BS decoder 的输出仍然是：

```text
(B, 2, 32, 32)
```

该输出空间具有共享的角延迟域结构。轻量 CNN residual refinement 可能提供稳定的重建先验，减少 LoRA 需要弥补的误差。

## 分歧本质

Gemini 更强调：

```text
输入 code 分布未知 -> decoder 应避免强手工偏置 -> 选 Transformer
```

Codex 更强调：

```text
输出 CSI 结构共享 -> decoder 可利用空间先验 -> Transformer 后接 CNN residual 可能更稳
```

这两个观点并不冲突。更合理的合成路线是：

```text
以 Transformer 作为 foundation decoder 主干，
以轻量 CNN residual head 作为可消融的输出 refinement。
```

也就是推荐：

```text
HybridDecoder = Transformer code mixer + CNN residual refinement
```

## 最推荐的 HybridDecoder 结构

### 总体结构

推荐结构：

```text
code: (B, code_dim)
  -> CodeAdapter
  -> Linear token projection
  -> Transformer decoder / token mixer
  -> reshape to CSI map
  -> CNN residual refinement
  -> output: (B, channel, nt, nc)
```

更具体：

```text
code
  -> LayerNorm(code_dim)
  -> Linear(code_dim, input_dim)
  -> reshape to tokens: (B, seq_len, d_model)
  -> TransformerDecoder / TransformerEncoder blocks
  -> reshape to coarse map: (B, channel, nt, nc)
  -> shallow CNN residual head
  -> out = coarse + residual
```

其中：

```text
input_dim = channel * nt * nc
seq_len = input_dim / d_model
code_dim = input_dim / cr
```

默认 COST2100 配置：

```text
channel = 2
nt = 32
nc = 32
input_dim = 2048
d_model = 64
seq_len = 32
cr = 4
code_dim = 512
```

### 推荐前向路径

推荐 forward：

```text
code
  -> code_adapter
  -> fc_decoder
  -> tokens
  -> transformer
  -> coarse
  -> residual_head(coarse)
  -> coarse + residual
```

伪代码：

```python
def forward(self, code):
    code = self.code_norm(code)
    tokens = self.fc_decoder(code)
    tokens = tokens.view(batch_size, seq_len, d_model)

    tokens = self.transformer(tokens, tokens)
    coarse = tokens.reshape(batch_size, channel, nt, nc)

    residual = self.refine(coarse)
    return coarse + residual
```

注意：最终不加 `sigmoid`、`hsigmoid`、`tanh` 或 `Identity`。

### CNN residual head 推荐结构

CNN head 不建议太重，初版保持轻量：

```text
Conv2d(channel, hidden, 3, padding=1)
LeakyReLU(0.3)
ResidualBlock(hidden)
ResidualBlock(hidden)
Conv2d(hidden, channel, 3, padding=1)
```

其中 `ResidualBlock` 可参考 CRNet/CLNet 风格：

```text
input
  -> ConvBN(hidden, hidden, 3)
  -> LeakyReLU
  -> ConvBN(hidden, hidden, 3)
  -> add identity
  -> LeakyReLU
```

也可以更轻：

```text
Conv2d(2, 16, 3)
LeakyReLU
Conv2d(16, 16, 3)
LeakyReLU
Conv2d(16, 2, 3)
```

推荐初始超参：

```text
refine_hidden = 16 或 32
refine_blocks = 2
activation = LeakyReLU(0.3)
normalization = BatchNorm2d 或不使用 normalization
```

如果 batch size 较小，`BatchNorm2d` 可能引入不稳定，可优先测试无 normalization 或 `GroupNorm`。

## 为什么不是纯 CNN decoder

纯 CNN residual decoder 可以作为 baseline，但不建议直接作为最终主线。

纯 CNN 方案：

```text
code
  -> LayerNorm
  -> Linear expand
  -> reshape
  -> CNN residual blocks
  -> out
```

优点：

- 简单稳定。
- 利用输出空间先验。
- 参数和训练成本较低。

缺点：

- code token 间全局关系建模较弱。
- 作为 foundation decoder 的表达能力可能不如 Transformer。
- 后续 LoRA 插入点不如 Transformer 线性层自然。

因此纯 CNN 更适合作为消融 baseline，而不是最推荐最终方案。

## 为什么不是纯 Transformer decoder

纯 Transformer decoder 是必须保留的 baseline。

优点：

- 表达能力强。
- 与 Gemini 的 foundation decoder 观点一致。
- LoRA 插入点清晰，尤其是 `fc_decoder`、attention projections、FFN。

风险：

- 对 encoder code 分布更敏感。
- 输出 map 缺少局部 refinement 先验。
- 如果训练数据有限，可能不如带 CNN head 的结构稳定。

因此不建议直接假设纯 Transformer 最优，应通过 LoRA 适配实验验证。

## LoRA 插入位置推荐

对于 HybridDecoder，LoRA 插入优先级如下。

### 1. fc_decoder / token projection

位置：

```text
code -> fc_decoder -> input_dim
```

这是最优先插入 LoRA 的位置。

原因：

- 不同 encoder/vendor 的主要差异首先体现为 code 分布不同。
- `fc_decoder` 是 code 到 decoder latent/token 空间的入口。
- LoRA 适配该层最直接。

推荐：

```text
LoRA rank: 4, 8, 16
alpha: 2 * rank 或 4 * rank
```

### 2. Transformer FFN

位置：

```text
Transformer layer linear1 / linear2
```

原因：

- FFN 控制 token feature 的非线性变换。
- 比 attention projection 更容易训练。
- 参数结构规则，适合后续生成。

### 3. Attention projections

位置：

```text
self_attn q_proj / k_proj / v_proj / out_proj
cross_attn q_proj / k_proj / v_proj / out_proj
```

原因：

- 能改变 token 交互方式。
- 对多 encoder code 语义差异可能有帮助。

风险：

- 生成参数更多。
- 训练不稳定性更高。

建议在 `fc_decoder + FFN` 有收益后再加入。

### 4. CNN residual head

初期不建议给 CNN head 加 LoRA。

原因：

- CNN head 应作为共享输出结构先验。
- 如果它也动态变化，会让系统难以分析。
- LoRA generator 输出维度会变大。

只有当实验显示不同 vendor 的输出残差形态差异很大时，再考虑对 CNN head 的最后一层 `Conv2d` 加轻量 LoRA/adapter。

## code_adapter 的推荐升级

当前 `code_adapter` 是：

```text
LayerNorm(code_dim) + Linear(code_dim, code_dim)
```

在多厂家 LoRA 生成任务中，建议将其升级为显式模块，并区分两个用途：

```text
decoder_input_adapter: 给 decoder 使用
lora_condition_adapter: 给 LoRA generator 使用
```

初版可以共享：

```text
raw code
  -> LayerNorm
  -> Linear
  -> adapted code
```

后续建议拆开：

```text
raw code -> decoder_adapter -> decoder
raw code or calibration set -> condition_adapter -> LoRA generator
```

### 单样本条件 vs 域条件

不建议一开始直接：

```text
single code -> generator -> LoRA
```

更稳的做法是：

```text
K 个 calibration codes
  -> set encoder / mean pooling
  -> vendor/domain embedding
  -> LoRA generator
```

简单 baseline：

```python
domain_embedding = mean(MLP(code_k), dim=K)
```

这样生成的 LoRA 更接近“厂家级适配”，而不是把每个样本的瞬时信息混进 decoder 参数。

## 推荐实验路线

### Phase 1：纯 decoder 选型

比较：

```text
A. TransNetDecoder
B. CNNResidualDecoder
C. HybridDecoder
```

训练方式：

```text
所有 encoder 轮换或混合训练
decoder 可训练
不使用 LoRA
```

记录：

```text
端到端 NMSE
训练稳定性
不同 encoder 的均衡性
参数量
推理开销
```

目标：

```text
筛掉明显不稳定或性能过差的 decoder。
```

### Phase 2：冻结 base decoder，训练 per-encoder static LoRA

对每个 decoder 候选：

```text
冻结 base decoder W0
每个 encoder/vendor 训练一套 LoRA
encoder 可固定或按实验设定训练
```

记录：

```text
zero-LoRA NMSE
static-LoRA NMSE
LoRA gain
LoRA rank vs NMSE
```

核心判断：

```text
哪个 decoder 最容易被少量 LoRA 适配？
```

这比端到端 NMSE 更重要。

### Phase 3：LoRA generator baseline

先不要直接上 diffusion/flow-matching。

先做：

```text
domain embedding -> MLP -> LoRA weights
```

训练目标：

```text
拟合 Phase 2 得到的 per-encoder optimal LoRA
或直接通过重建 loss 端到端训练 generator。
```

比较：

```text
static LoRA
MLP-generated LoRA
no LoRA
```

### Phase 4：Diffusion / Flow-Matching

只有当 MLP generator 已经显示收益，再考虑：

```text
domain embedding -> diffusion / flow-matching -> LoRA weights
```

适用场景：

- LoRA 参数存在多模态。
- 不同 calibration samples 对应多个可行 adapter。
- 希望生成分布而不是单点估计。

## 推荐消融实验

### Decoder 结构消融

```text
TransNetDecoder
TransNetDecoder + LayerNorm(code)
TransNetDecoder + CNN head
HybridDecoder full
CNNResidualDecoder
```

### CNN head 消融

```text
no CNN head
1 block
2 blocks
4 blocks
hidden=16
hidden=32
BatchNorm vs GroupNorm vs no norm
```

### LoRA 位置消融

```text
fc_decoder only
FFN only
fc_decoder + FFN
attention only
fc_decoder + FFN + attention
CNN head last conv
```

### 条件输入消融

```text
raw code
adapted code
mean pooled K calibration codes
vendor id embedding
adapted code + vendor id
```

## 推荐的当前实现顺序

建议不要一次性实现 diffusion/flow-matching。

更稳的当前开发顺序：

```text
1. 保留当前 TransNetDecoder 作为 baseline。
2. 实现 HybridDecoder，但默认 CNN head 很轻。
3. 增加 --decoder {transnet, hybrid, cnn_residual}。
4. 为 fc_decoder 添加 LoRA 注入点。
5. 做 per-encoder static LoRA。
6. 如果 static LoRA 有收益，再做 MLP LoRA generator。
7. 最后再考虑 diffusion/flow-matching。
```

## 最终建议

Gemini 的建议可以作为主线：

```text
Transformer foundation decoder + dynamic LoRA
```

但不应把“必须 Transformer”作为未经实验验证的结论。

Codex 最推荐的实际路线是：

```text
HybridDecoder:
  code_adapter
  -> Transformer foundation block
  -> CSI map reshape
  -> lightweight CNN residual refinement
  -> linear output
```

并通过以下指标决定是否保留 CNN head：

```text
冻结 base decoder 后的 LoRA 适配收益
未知 encoder/vendor 的 zero-shot / few-shot 表现
LoRA rank 与 NMSE 的效率曲线
训练稳定性
推理开销
```

如果实验显示 CNN residual head 没有提升，回退到纯 `TransNetDecoder`。

如果实验显示 CNN residual head 提升 LoRA 适配稳定性，则保留 hybrid 作为最终 base decoder。

核心原则：

```text
最终选型不由“Transformer 更通用”或“CNN 更适合 CSI”这种先验决定，
而由 fixed base decoder 在低秩可生成 adapter 下的跨厂家泛化能力决定。
```

