# 实验路线图与消融矩阵

## 总目标

验证以下系统是否可行：

```text
不同 UE encoder / vendor
  -> shared fixed BS base decoder
  -> generated LoRA adaptation
  -> robust CSI reconstruction
```

核心问题：

```text
哪个 base decoder 最适合作为低秩、可生成 adapter 的底座？
```

## Phase 0：数据和代码一致性检查

目标：

```text
确认数据范围、输出层、NMSE 坐标一致。
```

检查项：

```text
1. .pt 数据范围为 [-0.5, 0.5]
2. evaluator 不再执行 -0.5
3. decoder 最终直接 return out
4. CLI 中无 output_activation
5. scripts 中无 output_activation
6. 历史 [0,1] checkpoint 不直接混用
```

成功标准：

```text
python main.py --help 中无 output_activation
rg 非 exps 路径无 output_activation / HSigmoid / hsigmoid
小规模 evaluation 能正常输出 loss 和 NMSE
```

## Phase 1：Decoder 候选结构训练

候选：

```text
A. TransNetDecoder
B. CNNResidualDecoder
C. HybridDecoder
```

训练方式：

```text
encoder 可选 csinet / crnet / clnet / transnet
decoder 可训练
不使用 LoRA
使用相同数据范围 [-0.5, 0.5]
```

建议实验矩阵：

```text
encoder x decoder

csinet   x transnet
csinet   x cnn_residual
csinet   x hybrid

crnet    x transnet
crnet    x cnn_residual
crnet    x hybrid

clnet    x transnet
clnet    x cnn_residual
clnet    x hybrid

transnet x transnet
transnet x cnn_residual
transnet x hybrid
```

记录：

```text
train loss
val loss
test NMSE
参数量
FLOPs
训练时间
不同 seed 稳定性
```

成功标准：

```text
HybridDecoder 至少不明显弱于 TransNetDecoder；
CNN residual head 不导致训练不稳定；
不同 encoder 上性能差距不过度失衡。
```

## Phase 2：混合 encoder 训练 base decoder

目标：

```text
训练一个面向多 encoder 的 shared base decoder W0。
```

训练策略：

```text
batch 内混合 encoder
或 epoch 间轮换 encoder
或每个 encoder 单独 forward 后共享 decoder 更新
```

需要注意：

```text
如果不同 encoder 同时训练，encoder 本身也会适配 decoder；
这可能掩盖真实 vendor shift。
```

因此建议拆成两种设置：

### 设置 A：端到端混合训练

```text
encoder_i trainable
decoder trainable
```

用途：

```text
得到强 base decoder 和可用 baseline。
```

### 设置 B：encoder 固定或半固定

```text
encoder_i fixed or pretrained
decoder trainable
```

用途：

```text
更接近不同厂家私有 encoder 场景。
```

## Phase 3：冻结 base decoder，训练 static LoRA

这是关键阶段。

流程：

```text
1. 选择 Phase 1/2 中表现较好的 base decoder。
2. 冻结 base decoder W0。
3. 对每个 encoder/vendor 训练独立 LoRA_i。
4. 比较 no-LoRA 与 static-LoRA。
```

LoRA 插入配置：

```text
Config 1: fc_decoder only
Config 2: Transformer FFN only
Config 3: fc_decoder + FFN
Config 4: attention projections
Config 5: fc_decoder + FFN + attention
Config 6: optional CNN head last conv
```

推荐优先：

```text
fc_decoder only
```

然后：

```text
fc_decoder + FFN
```

记录：

```text
LoRA rank
LoRA alpha
可训练参数量
NMSE gain
是否接近 full fine-tuning
训练稳定性
```

成功标准：

```text
static LoRA 能显著恢复冻结 base decoder 的 NMSE 损失；
LoRA 参数量控制在约 1-5%；
不同 encoder 上 LoRA gain 稳定。
```

如果 static LoRA 无收益，不应继续训练 LoRA generator。

## Phase 4：Domain embedding 和 MLP LoRA generator

目标：

```text
用厂家级 domain embedding 生成 LoRA，而不是为每个 encoder 存一套 LoRA。
```

输入：

```text
K calibration codes per vendor
```

Domain embedding baseline：

```python
z_k = MLP(code_k)
z_domain = mean(z_k, dim=K)
```

Generator：

```text
z_domain -> MLP -> LoRA weights
```

训练方式可选：

### 方式 A：拟合 static LoRA

```text
target = Phase 3 trained LoRA weights
loss = MSE(predicted_lora, target_lora)
```

优点：

```text
训练简单，目标明确。
```

缺点：

```text
LoRA 权重有等价变换，直接 MSE 未必最优。
```

### 方式 B：重建 loss 端到端训练 generator

```text
predicted_lora = Generator(z_domain)
H_hat = Decoder_W0+predicted_lora(code)
loss = MSE(H_hat, H_gt)
```

优点：

```text
直接优化目标任务。
```

缺点：

```text
训练更复杂。
```

建议先做方式 A，再做方式 B。

成功标准：

```text
generated LoRA 明显优于 no-LoRA；
generated LoRA 接近 static LoRA 上限；
K 较小时仍稳定。
```

## Phase 5：Diffusion / Flow-Matching LoRA generator

只有在以下条件满足后再做：

```text
static LoRA 有收益；
MLP generator 有收益；
MLP generator 与 static LoRA 上限仍有明显差距；
或 LoRA 分布明显多模态。
```

候选：

```text
domain embedding -> diffusion -> LoRA weights
domain embedding -> flow matching -> LoRA weights
```

评价重点：

```text
unseen vendor
few-shot calibration
LoRA diversity
generated LoRA stability
inference latency
```

不要把 diffusion/flow-matching 作为第一版 generator。

## Decoder 消融矩阵

### 主结构消融

```text
TransNetDecoder
TransNetDecoder + code LayerNorm
CNNResidualDecoder
HybridDecoder without CNN head
HybridDecoder with CNN head
HybridDecoder with deeper CNN head
```

### Transformer 消融

```text
TransformerDecoderLayer(memory=tgt)
TransformerEncoderLayer token mixer
num_layers = 1 / 2 / 4
d_model = 32 / 64 / 128
dim_feedforward = 2*d_model / 4*d_model / 2048
nhead = 2 / 4
```

### CNN head 消融

```text
no CNN head
1 residual block
2 residual blocks
4 residual blocks
hidden = 16
hidden = 32
BatchNorm
GroupNorm
no normalization
```

### LoRA 消融

```text
rank = 2 / 4 / 8 / 16
alpha = rank / 2*rank / 4*rank
fc_decoder only
FFN only
attention only
fc_decoder + FFN
fc_decoder + FFN + attention
CNN head last conv
```

### 条件输入消融

```text
single raw code
single adapted code
mean pooled K raw codes
mean pooled K adapted codes
vendor id embedding
vendor id + domain embedding
K = 1 / 5 / 10 / 20
```

## 评价指标

基础指标：

```text
MSE loss
NMSE
```

多厂家泛化指标：

```text
seen vendor NMSE
unseen vendor NMSE
zero-shot NMSE
few-shot calibration NMSE
static LoRA upper bound
generated LoRA gap
```

效率指标：

```text
base decoder params
LoRA params
generator params
FLOPs
inference latency
calibration cost K
```

稳定性：

```text
seed 42
seed 2026
seed 3407
mean ± std
```

## 推荐近期执行顺序

最务实顺序：

```text
1. 确认当前 TransNetDecoder baseline 在 [-0.5,0.5] 上稳定。
2. 实现 HybridDecoder，但 CNN head 保持轻量。
3. 增加 --decoder {transnet, cnn_residual, hybrid}。
4. 跑 encoder x decoder 小规模矩阵。
5. 选择 TransNetDecoder 和 HybridDecoder 做 static LoRA。
6. 只在 fc_decoder 上先加 LoRA。
7. 验证 static LoRA 是否有收益。
8. 再做 domain embedding + MLP generator。
9. 最后考虑 diffusion / flow-matching。
```

## 决策标准

最终 decoder 不由先验决定，而由以下问题决定：

```text
冻结 base decoder 后，
少量 LoRA 是否能快速适配不同 encoder/vendor？
```

如果纯 Transformer 在该指标上最好：

```text
选择 TransNetDecoder。
```

如果 HybridDecoder 在该指标上更稳：

```text
选择 HybridDecoder。
```

如果 CNNResidualDecoder 端到端很好但 LoRA 适配差：

```text
不作为最终 base decoder。
```

最终成功标准：

```text
fixed base decoder + generated LoRA
在 unseen vendor / encoder 上显著优于 fixed base decoder only，
并接近 static per-vendor LoRA 上限。
```

