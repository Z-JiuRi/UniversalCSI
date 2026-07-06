# Decoder 全量参数 Flow Matching 生成实验设计

## 目标

本项目验证一个极限设定：

```text
输入:
  guide_codes: (N, 512)  # 指导码字集合
  theta_base:  随机初始化 decoder 参数
  theta_star:  目标 decoder 全量参数

输出:
  theta_gen:   Flow Matching 生成的 decoder 全量参数
```

当前第一版固定使用：

```text
theta_star checkpoint:
  exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth

decoder args:
  exps/COST2100/in/seed42/transnet_transnet/args.json

guide_codes:
  同一实验导出的 codewords，例如:
  exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt
```

不使用：

```text
target_code
raw CSI
adapter
LoRA
mapper
```

第一版目标不是证明泛化，而是确认：

- 全量 decoder 参数能否被 token 化、归一化、FM 生成、反归一化并还原。
- `(N,512)` 指导码字集合能否通过三种粗提取方式变成 `(512,512)` 条件。
- 三种条件注入方式能否指导参数 velocity 预测。
- 仅使用 MSE 类 loss 时，生成参数能否接近目标参数和目标 decoder 函数行为。

## 固定实验设定

### Decoder

第一版固定为：

```text
decoder = transnet
code_dim = 512
channel = 2
nt = 32
nc = 32
```

目标参数：

```text
theta_star = decoder state_dict from seed42/transnet_transnet/best_nmse.pth
```

初始参数：

```text
theta_base = 同架构 transnet decoder 随机初始化参数
```

注意：`theta_base` 必须保存到本地，推理和反归一化时必须加载同一个 `theta_base`，不能每次重新随机初始化。

### 训练监督

可用：

```text
theta_base
theta_star
guide_codes: (N,512)
D_star(guide_codes)  # 可由目标 decoder 预计算，用于函数 MSE
```

不可用：

```text
target_code
raw CSI
真实 NMSE
```

因此第一版只使用 MSE 类 loss：

```text
velocity_mse
endpoint_param_mse
function_mse          # optional, still MSE
fc_feature_mse        # optional, still MSE
```

## 总体流程

```text
guide_codes: (N,512)
        |
        v
Condition Extractor
  random / svd / set_query_transformer
        |
        v
condition_tokens: (512,512)
        |
        v
Condition Encoder
        |
        +-----------------------------+
                                      |
theta_base -> normalize/tokenize -----+
                                      |
theta_star -> normalize/tokenize --+  |
                                  |   |
                                  v   v
                            sample t, theta_t
                                  |
                                  v
                         Parameter FM
              film / cross_attention / hyper_lora
                                  |
                                  v
                           velocity_pred
                                  |
                                  v
                         theta_gen_norm
                                  |
                                  v
                    detokenize + denormalize
                                  |
                                  v
                           theta_gen
```

## 当前实现与模块划分

当前代码已经落在 `decoder_param_fm/` 子项目中：

```text
decoder_param_fm/
  README.md
  param_utils.py
  models.py
  train_param_fm.py
  sample_param_fm.py
  test_generated_nmse.py
  scripts/
    run_param_fm.sh
    run_param_fm_1ep.sh
    train_param_fm.sh
    sample_param_fm.sh
    test_generated_nmse.sh
    run_smoke_1ep.sh
  exps/
    ...
```

核心文件职责：

- `param_utils.py`：checkpoint 清洗、decoder 参数抽取、随机 `theta_base` 构造、per-tensor 归一化、token 化、反 token 化、反归一化、CSI/code 加载。
- `models.py`：三种条件粗提取、条件编码器、三种条件注入的参数 FM 网络、可学习 `tensor_id/layer_id/token_offset` meta embedding。
- `train_param_fm.py`：训练入口，保存 `args.json`、`run.log`、`artifacts/`、`checkpoints/`，学习率使用 warmup cosine。
- `sample_param_fm.py`：从 `theta_base` 出发做 ODE Euler 采样，生成 decoder 全量参数 checkpoint。
- `test_generated_nmse.py`：加载生成 decoder，用成对 code/CSI 计算全局 NMSE。

## 运行方式

正式实验脚本采用顶部集中 `参数=值` 的写法：

```bash
bash decoder_param_fm/scripts/run_param_fm.sh
```

1 epoch smoke 测试脚本同样采用顶部集中 `参数=值`：

```bash
bash decoder_param_fm/scripts/run_param_fm_1ep.sh
```

默认 1ep 配置：

```text
gpu=0
epochs=1
steps_per_epoch=1
condition_extract=random
condition_inject=film
param_norm=rms
token_size=512
hidden_dim=64
num_blocks=1
max_guide_codes=2048
max_samples=2048
```

训练输出目录遵循主项目风格：

```text
decoder_param_fm/exps/{run_name}/
  args.json
  run.log
  artifacts/
    theta_base.pt
    theta_star.pt
    param_meta.json
    norm_stats.pt
    decoder_args_resolved.json
  checkpoints/
    best_loss.pth
    last.pth
  generated/
    generated_decoder.pth
    nmse.json
```

当前已完成一次 GPU0 的 1ep smoke，日志显示：

```text
Using device: cuda:0
Prepared 40 tensors, 3236 parameter tokens, token_size=512
Epoch [1/1] loss=1.32367170 lr=0.0002
```

对应输出：

```text
decoder_param_fm/exps/smoke_1ep_random_film_rms_tok512_h64_seed42/generated/nmse.json
```

smoke NMSE 结果：

```text
num_samples = 2048
nmse_db = 33.48254430949862
```

该 smoke 只验证训练、采样、反归一化、加载 decoder、NMSE 测试全链路可运行；1 step 不代表生成质量。

## 参数抽取与保存

### 从 checkpoint 抽取 decoder 参数

目标 checkpoint 是完整 UniversalCSI 模型 checkpoint。需要抽取 decoder 部分：

```text
checkpoint["state_dict"]
  -> 过滤 key.startswith("decoder.")
  -> 去掉 "decoder." 前缀
  -> 与新建 decoder.state_dict() key 对齐
```

保存：

```text
theta_star.pt
```

### 随机初始化 theta_base

用同一个 args 构建 transnet decoder：

```text
decoder = build_decoder(
  decoder="transnet",
  cr=4,
  d_model=64,
  channel=2,
  nt=32,
  nc=32,
  dim_feedforward=...
)
```

保存随机初始化参数：

```text
theta_base.pt
base_init_seed
base_decoder_args.json
```

后续推理必须加载这份 `theta_base.pt`。

### 参数元信息

保存 `param_meta.json`：

```json
[
  {
    "name": "fc_decoder.weight",
    "shape": [2048, 512],
    "numel": 1048576,
    "token_start": 0,
    "token_end": 2048
  },
  {
    "name": "fc_decoder.bias",
    "shape": [2048],
    "numel": 2048,
    "token_start": 2048,
    "token_end": 2052
  }
]
```

## Tensor 级归一化

归一化按 tensor 存统计量，不按整层混合。

支持参数：

```text
--param_norm rms|zscore
```

统计量保存到：

```text
norm_stats.pt
```

### RMS 归一化

对每个 tensor：

```text
delta[name] = theta_star[name] - theta_base[name]
scale[name] = sqrt(mean(delta[name]^2)) + eps
```

归一化：

```text
theta_base_norm[name] = 0
theta_star_norm[name] = delta[name] / scale[name]
```

反归一化：

```text
theta_gen[name] = theta_base[name] + theta_gen_norm[name] * scale[name]
```

### Z-score 归一化

当前只针对这一个 decoder，因此 zscore 也对 delta 做：

```text
delta[name] = theta_star[name] - theta_base[name]
mean[name] = mean(delta[name])
std[name] = std(delta[name]) + eps
```

归一化：

```text
theta_base_norm[name] = 0
theta_star_norm[name] = (delta[name] - mean[name]) / std[name]
```

反归一化：

```text
delta_gen[name] = theta_gen_norm[name] * std[name] + mean[name]
theta_gen[name] = theta_base[name] + delta_gen[name]
```

### norm_stats 格式

```python
norm_stats = {
    "fc_decoder.weight": {
        "method": "rms",
        "shape": [2048, 512],
        "scale": tensor(...),
    },
    "decoder.layers.0.norm1.weight": {
        "method": "zscore",
        "shape": [64],
        "mean": tensor(...),
        "std": tensor(...),
    },
}
```

推理时必须加载同一份 `theta_base.pt` 和 `norm_stats.pt`。

## 参数 token 化

所有 decoder 参数按 tensor 展平后切成固定长度 token。

默认：

```text
param_token_size = 512
```

对每个 tensor：

```text
flat = theta_norm[name].flatten()
tokens = split(flat, size=512)
最后一个 token 不足 512 时补 0
```

需要保存 element mask：

```text
param_tokens[name]: (T_name, 512)
param_mask[name]:   (T_name, 512)
```

其中：

```text
mask = 1 表示真实参数元素
mask = 0 表示 padding
```

重要规则：

- padding 位置不参与 velocity MSE。
- padding 位置不参与 endpoint MSE。
- padding 位置不写回参数。
- cross-attention 中 padding token 可以作为 query 存在，但 loss 和 detokenize 必须使用 element mask。

合并所有 tensor：

```text
param_tokens: (T_total, 512)
param_mask:   (T_total, 512)
token_meta:   记录每个 token 属于哪个 tensor 和 offset
```

`token_meta` 不只用于 detokenize，也作为模型输入的一部分，用可学习 embedding 告诉参数生成器：

```text
这个 token 来自哪个 tensor、decoder 的哪一层、tensor 内第几个 token。
```

否则所有参数 token 都只是长度 512 的数值块，模型很难区分：

```text
fc_decoder.weight 的第 10 个 token
decoder.layers.0.self_attn.in_proj_weight 的第 10 个 token
decoder.norm.bias 的 token
```

这些位置的语义完全不同，必须显式编码。

## Token Meta Embedding

每个参数 token 需要构造结构化元信息：

```json
{
  "token_index": 0,
  "tensor_name": "fc_decoder.weight",
  "tensor_id": 0,
  "layer_id": 0,
  "token_offset": 0,
  "num_tokens_in_tensor": 2048
}
```

第一版只使用三个离散字段：

```text
tensor_embedding[tensor_id]
layer_embedding[layer_id]
token_offset_embedding[token_offset]
```

然后合成：

```text
meta_embedding =
    tensor_embedding
  + layer_embedding
  + token_offset_embedding
```

不在第一版加入：

```text
module_type
param_type
block_type
```

原因是 `tensor_id` 已经唯一对应具体 tensor，隐含了 weight/bias、attention/ffn/norm 等信息；`layer_id` 提供层级位置；`token_offset` 提供 tensor 内位置。三者已经覆盖第一版需要的位置信息，额外字段后续可以作为 ablation 加回。

最终参数 token 输入不再只是：

```text
param_token
```

而是：

```text
param_token + meta_embedding + time_embedding
```

或者更明确地：

```text
h = Linear(param_token)
  + MetaProjection(meta_embedding)
  + TimeProjection(time_embedding)
```

### Meta 字段定义

第一版支持以下字段。

#### tensor_id

每个 state_dict tensor 一个唯一 id：

```text
fc_decoder.weight -> 0
fc_decoder.bias -> 1
decoder.layers.0.self_attn.in_proj_weight -> 2
...
```

用途：

```text
区分具体 tensor。
```

#### layer_id

按 decoder 层级编号：

```text
fc_decoder: layer_id = 0
decoder.layers.0: layer_id = 1
decoder.layers.1: layer_id = 2
decoder.norm: layer_id = 3
```

用途：

```text
表达参数在 decoder 深度上的位置。
```

#### token_offset

tensor 内 token 的序号：

```text
0, 1, 2, ..., num_tokens_in_tensor - 1
```

如果某个 tensor token 很多，可以分桶：

```text
offset_bucket = floor(token_offset / bucket_size)
```

或使用正弦位置编码：

```text
sinusoidal(token_offset)
```

第一版使用：

```text
learned token_offset_embedding，最大长度覆盖最大 tensor token 数。
```

### 可选扩展字段

如果后续发现只用三类 meta 不够，可以再加入：

```text
module_type: fc_decoder / self_attn / cross_attn / ffn / norm
param_type:  weight / bias / norm_weight / norm_bias
block_type:  input_projection / attention / feedforward / normalization
```

这些字段第一版不实现，避免 meta embedding 过复杂。

### Meta Embedding 的使用位置

三种条件注入方式都应使用 token meta embedding。

#### film

```text
h = ParamTokenEncoder(param_token)
h = h + MetaProjection(meta_embedding)
h = h + TimeProjection(time_embedding)
h = FiLM(h, layer_condition)
velocity_token = VelocityHead(h)
```

#### cross_attention

```text
param_h = ParamTokenEncoder(param_token)
param_h = param_h + MetaProjection(meta_embedding)
param_h = param_h + TimeProjection(time_embedding)

param_h = CrossAttention(query=param_h, key=cond_h, value=cond_h)
velocity_token = VelocityHead(param_h)
```

#### hyper_lora

```text
h = ParamTokenEncoder(param_token)
h = h + MetaProjection(meta_embedding)
h = h + TimeProjection(time_embedding)

base_velocity = BaseVelocityHead(h)
lowrank_update = HyperLoRA(layer_condition, meta_embedding)
velocity_token = base_velocity + lowrank_update
```

### 为什么需要可学习 meta embedding

直接用 token 在全局 flatten 后的位置做固定位置编码也可以，但不够结构化。decoder 参数天然有层级语义：

```text
层位置
模块类型
参数类型
tensor 内偏移
```

可学习 meta embedding 能让模型自己学习这些语义对参数 velocity 的影响。

### 保存内容

`param_meta.json` 保存离散 id 和字符串映射：

```json
{
  "token_size": 512,
  "tokens": [
    {
      "global_token_id": 0,
      "tensor_name": "fc_decoder.weight",
      "tensor_id": 0,
      "layer_id": 0,
      "token_offset": 0,
      "valid_elements": 512
    }
  ]
}
```

模型 checkpoint 保存可学习 embedding 参数：

```text
tensor_embedding
layer_embedding
token_offset_embedding
```

## 条件粗提取

输入：

```text
guide_codes: (N, 512)
```

输出统一为：

```text
condition_tokens: (512, 512)
condition_mask:   (512,)  # 第一版通常全 1
```

暴露参数：

```text
--condition_extract random|svd|set_transformer
```

### 方式 1：random

直接随机采样 512 条：

```text
idx = randperm(N)[:512]
condition_tokens = guide_codes[idx]
```

训练时可以每个 iteration 重采样。

优点：

- 简单。
- 保留真实 code 样本。
- 有天然 augmentation。

缺点：

- 有采样噪声。
- 可能漏掉低频区域。

### 方式 2：svd

对全量 guide code 做样本维 SVD：

```text
Z = guide_codes                      # (N,512)
mu = mean(Z, dim=0)
Zc = Z - mu
Zc ≈ U[:, :512] S[:512] V[:512, :]
condition_tokens = S[:512, None] * V[:512, :]
```

输出：

```text
condition_tokens: (512,512)
```

建议额外把 `mu` 注入 Condition Encoder：

```text
condition_global_extra = mu
```

注意：

- SVD token 不是实际 code。
- 要处理符号翻转，例如令每个主方向绝对值最大的位置为正。
- SVD 可预计算并保存。

### 方式 3：set_transformer

你要求 SetTransformer 输入全量 `(N,512)`。不能用标准 full self-attention，因为复杂度是 `O(N^2)`。

实现上使用 query cross-attention：

```text
learned_queries: (512, d)
guide_embed:     MLP(guide_codes) -> (N, d)

condition_tokens = CrossAttention(
  query = learned_queries,
  key   = guide_embed,
  value = guide_embed
)
condition_tokens -> Linear(d, 512)
```

复杂度：

```text
O(512 * N)
```

而不是：

```text
O(N^2)
```

这满足“输入是全量 `(N,512)`”的要求，同时避免全量 self-attention 爆炸。

建议参数：

```text
set_d_model = 512
set_num_queries = 512
set_num_heads = 8
set_num_layers = 2
```

## Condition Encoder

无论粗提取方式是什么，统一输入：

```text
condition_tokens: (512,512)
condition_mask:   (512,)
```

基础编码：

```text
tokens = LayerNorm(condition_tokens)
tokens = TokenMLP(tokens)              # (512, d)
tokens = optional Transformer blocks   # (512, d)
```

全局条件：

```text
mean_pool = masked_mean(tokens)
std_pool  = masked_std(tokens)
max_pool  = masked_max(tokens)

global_condition = MLP([mean_pool, std_pool, max_pool])  # (d,)
```

分层条件：

```text
layer_condition[name] = LayerConditionHead[name](global_condition)
```

例如：

```text
fc_decoder
decoder.layers.0.self_attn
decoder.layers.0.multihead_attn
decoder.layers.0.ffn
decoder.layers.1.self_attn
decoder.layers.1.multihead_attn
decoder.layers.1.ffn
decoder.norm
```

## 条件注入到参数生成

暴露参数：

```text
--condition_inject film|cross_attention|hyper_lora
```

三种方式都要实现。

### 方式 1：film

每个参数 token 先经过 token state encoder：

```text
h = ParamTokenEncoder(param_token)
h = h + MetaProjection(meta_embedding)
h = h + TimeProjection(time_embedding)
```

然后由条件生成 FiLM 参数：

```text
gamma, beta = FiLMHead(layer_condition[name])
h = gamma * h + beta
velocity_token = VelocityHead(h)
```

特点：

- 最稳定。
- 参数开销小。
- 第一版默认推荐。

### 方式 2：cross_attention

参数 token 作为 query，condition token 作为 key/value：

```text
param_h = ParamTokenEncoder(param_tokens)
param_h = param_h + MetaProjection(meta_embeddings)
param_h = param_h + TimeProjection(time_embedding)
cond_h = ConditionTokenEncoder(condition_tokens)

param_h = CrossAttention(
  query = param_h,
  key   = cond_h,
  value = cond_h,
  key_padding_mask = condition_mask
)

velocity_tokens = VelocityHead(param_h)
```

你确认可以接受 cross-attention 参数和显存开销，因此实现时可以直接支持全量参数 token。

注意：

- param padding element 不参与 loss，但 param token 本身仍可参与 attention。
- condition_mask 要正确传入。

### 方式 3：hyper_lora

条件网络不直接生成全量 velocity，而是生成低秩调制：

```text
h = ParamTokenEncoder(param_token)
h = h + MetaProjection(meta_embedding)
h = h + TimeProjection(time_embedding)

base_velocity = BaseVelocityHead(h)

A, B, alpha = HyperLoRAHead(layer_condition[name])
lowrank_update = A @ B 或 token-wise lowrank projection

velocity = base_velocity + alpha * lowrank_update
```

对大矩阵参数，例如 `fc_decoder.weight`，可以按 token 或按 tensor 生成低秩 velocity。

特点：

- 参数效率较高。
- 对全量参数生成更稳。
- 但表达能力受 rank 限制。

建议默认：

```text
hyper_lora_rank = 16
```

## Parameter FM

训练路径：

```text
theta_0 = theta_base_norm_tokens
theta_1 = theta_star_norm_tokens
t ~ Uniform(t_eps, 1 - t_eps)
theta_t = (1 - t) * theta_0 + t * theta_1
v_target = theta_1 - theta_0
v_pred = ParamFM(theta_t, t, condition)
```

其中 `theta_*_tokens` shape：

```text
(T_total, 512)
```

## Loss

只使用 MSE 类 loss。

### velocity_mse

```text
velocity_mse = masked_mse(v_pred, v_target, param_mask)
```

### endpoint_param_mse

```text
theta_pred_1 = theta_t + (1 - t) * v_pred
endpoint_param_mse = masked_mse(theta_pred_1, theta_1, param_mask)
```

### function_mse 可选

虽然没有 CSI，但可以用目标 decoder 做函数蒸馏：

```text
probe_codes = sample(guide_codes, B)

D_pred = decoder(detokenize(denorm(theta_pred_1)))
D_star = target decoder

function_mse = MSE(D_pred(probe_codes), D_star(probe_codes))
```

这仍然是 MSE，不是重建监督。

### fc_feature_mse 可选

对 transnet decoder：

```text
fc_feature_mse = MSE(
  D_pred.fc_decoder(probe_codes),
  D_star.fc_decoder(probe_codes)
)
```

### 总损失

```text
loss = velocity_mse
     + lambda_endpoint * endpoint_param_mse
     + lambda_function * function_mse
     + lambda_fc * fc_feature_mse
```

默认：

```text
lambda_endpoint = 1.0
lambda_function = 0.0
lambda_fc = 0.0
```

如果只想做纯参数 FM，就保持 function/fc 为 0。

如果参数 MSE 下降但 decoder 行为不接近，再打开：

```text
lambda_function = 0.1
lambda_fc = 0.1
```

## 训练流程

1. 构建目标 decoder，加载 `theta_star`。
2. 用同架构随机初始化 decoder，保存 `theta_base`。
3. 按 tensor 计算 `norm_stats`，保存到本地。
4. 对 `theta_base/theta_star` 做归一化。
5. 按 tensor flatten 并切成 512 长度参数 token，生成 `param_mask/token_meta`。
6. 加载 `guide_codes: (N,512)`。
7. 根据 `--condition_extract` 得到 `condition_tokens: (512,512)`。
8. 根据 `--condition_inject` 训练 ParamFM。
9. 保存 checkpoint 和训练指标。

## 推理流程

1. 加载：

```text
theta_base.pt
norm_stats.pt
param_meta.json
ParamFM checkpoint
guide_codes.pt
```

2. 重新计算或加载 condition tokens：

```text
condition_tokens = CoarseExtract(guide_codes)
```

3. 从归一化参数起点开始：

```text
theta = theta_base_norm_tokens
```

4. ODE 采样：

```text
for step in range(ode_steps):
    t = step / ode_steps
    v = ParamFM(theta, t, condition)
    theta = theta + dt * v
```

5. detokenize：

```text
theta_gen_norm[name]
```

6. 反归一化：

```text
if norm == rms:
    theta_gen[name] = theta_base[name] + theta_gen_norm[name] * scale[name]

if norm == zscore:
    delta_gen[name] = theta_gen_norm[name] * std[name] + mean[name]
    theta_gen[name] = theta_base[name] + delta_gen[name]
```

7. 加载到 decoder：

```text
decoder.load_state_dict(theta_gen)
```

8. 保存生成参数：

```text
generated_decoder.pth
generated_decoder_state_dict.pt
```

## 评估指标

### 参数指标

```text
param_mse
param_rmse
param_cos_delta
per_tensor_mse
per_tensor_rmse
```

所有参数指标必须忽略 padding。

### 函数一致性指标

没有 CSI，因此只能评估目标 decoder 函数一致性：

```text
probe_codes = sample or heldout guide_codes

function_mse = MSE(D_gen(probe_codes), D_star(probe_codes))
fc_feature_mse = MSE(D_gen.fc_decoder(probe_codes), D_star.fc_decoder(probe_codes))
```

不能报告：

```text
true NMSE
CSI reconstruction MSE
```

除非后续加入 raw CSI。

## 必做对照

### normal condition

正常使用：

```text
condition_tokens = extract(guide_codes)
```

### zero condition

```text
condition_tokens = zeros(512,512)
```

如果 zero condition 和 normal condition 一样好，说明单任务下模型没有使用码字条件。

### random condition

用随机噪声代替：

```text
condition_tokens = randn(512,512)
```

### 三种粗提取对照

固定其他参数，只切换：

```text
--condition_extract random
--condition_extract svd
--condition_extract set_transformer
```

### 三种注入方式对照

固定其他参数，只切换：

```text
--condition_inject film
--condition_inject cross_attention
--condition_inject hyper_lora
```

### 两种归一化对照

固定其他参数，只切换：

```text
--param_norm rms
--param_norm zscore
```

## 默认配置

```text
condition_extract = random
condition_inject = film
param_norm = rms
param_token_size = 512
target_scope = all  # 固定全量 decoder 参数，不做接口
token_meta_embedding = learned
token_meta_fields = tensor_id, layer_id, token_offset

theta_base = random_init_saved
theta_star = exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth
guide_codes = exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt

loss:
  lambda_endpoint = 1.0
  lambda_function = 0.0
  lambda_fc = 0.0

fm:
  hidden_dim = 1024
  num_blocks = 4
  time_dim = 128
  ode_steps = 16
  ode_method = euler
```

## 实现注意事项

### 1. theta_base 必须固定

随机初始化的 base decoder 是生成路径起点。必须保存：

```text
theta_base.pt
base_init_seed
base_decoder_args.json
```

推理时不能重新初始化。

### 2. zscore 是 delta-zscore

当前只针对一个 decoder，所以 zscore 定义在 delta 上：

```text
delta = theta_star - theta_base
```

不是对多个 decoder checkpoint 做数据集统计。

### 3. SetTransformer 用 query cross-attention

输入是全量 `(N,512)`，但不能做 full self-attention。

使用：

```text
512 learned queries cross-attend to N guide code embeddings
```

### 4. padding mask 必须贯穿训练和推理

padding 位置不能参与：

```text
velocity_mse
endpoint_param_mse
detokenize writeback
```

### 5. token_meta 要进入模型

`param_meta.json` 不只是还原参数的索引表，还要提供可学习 embedding 的离散 id：

```text
tensor_id
layer_id
token_offset
```

训练和推理时都必须用同一份 `param_meta.json` 构造 meta embedding。否则模型无法知道每个 512 维参数 token 在 decoder 中的位置。

### 6. 单任务下 condition 可能被忽略

因为只有一个目标：

```text
guide_codes -> theta_star
```

模型可能只记住 `theta_star`。必须做 zero/random condition 对照。

### 7. 没有 CSI 不代表没有函数评估

可以比较：

```text
D_gen(code) vs D_star(code)
```

但这只是目标 decoder 蒸馏，不是真实重建指标。

## 里程碑

### Milestone 1：参数处理链路

- 抽取 `theta_star`。
- 随机初始化并保存 `theta_base`。
- 保存 `param_meta`。
- 保存 `norm_stats`。
- token 化和 detokenize 完全一致。
- 从 `param_meta` 构造可学习 meta embedding 所需 id。
- 反归一化后能还原目标参数。

### Milestone 2：条件粗提取

- random 输出 `(512,512)`。
- svd 输出 `(512,512)`。
- set_transformer 输出 `(512,512)`。

### Milestone 3：参数 FM

- film 可训练。
- cross_attention 可训练。
- hyper_lora 可训练。
- 三种注入方式都使用 token meta embedding。
- velocity/endpoint MSE 正确 mask padding。

### Milestone 4：采样与保存

- 从 `theta_base` ODE 采样。
- 反归一化为真实参数。
- 加载到 decoder。
- 保存生成 checkpoint。

### Milestone 5：对照实验

- normal / zero / random condition。
- random / svd / set_transformer。
- film / cross_attention / hyper_lora。
- rms / zscore。

## 结论

本项目第一版固定为：

```text
guide_codes: (N,512)
theta_base: 随机初始化 transnet decoder
theta_star: seed42/transnet_transnet best decoder
target_scope: 全量 decoder 参数
```

需要实现：

```text
3 种条件粗提取:
  random / svd / set_transformer

3 种条件注入:
  film / cross_attention / hyper_lora

2 种 tensor 级归一化:
  rms / zscore

参数 token 化:
  token_size=512
  padding mask
```

训练 loss 使用 MSE 类目标：

```text
velocity_mse
endpoint_param_mse
optional function_mse
optional fc_feature_mse
```

该实验主要验证全量参数 FM 生成链路是否可行。由于只有一个目标 decoder，必须通过 zero/random condition 对照判断模型是否真的使用了指导码字条件。
