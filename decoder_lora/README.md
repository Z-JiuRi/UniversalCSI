# decoder_lora 实验说明与崩溃配置总结

## 目标

`decoder_lora` 用于测试：

```text
source code -> closed-form affine/procrustes -> z0
z0 -> optional code adapter -> z1
z1 -> frozen seed42 decoder + LoRA -> CSI reconstruction
```

当前主要结构是：

```text
z1 = z0
   + gate_lr  * Up(Down(LN(z0)))
   + gate_mlp * W2 GELU(W1 LN(z0))
```

其中 `fc_decoder` 和 Transformer FFN 可以分别设置 LoRA rank/alpha。

## 崩溃判据

实验里观察到的“崩溃”不是单纯没有 `metrics.json`。有些实验只是进程被中断或还没跑完，日志尾部仍然正常。

真正的优化崩溃通常表现为：

```text
true_decoder_eval NMSE 从 -20~-26 dB 突然掉到 0 dB 附近
train_rec 从 1e-6 量级跳到 1e-4 量级
train_code 或 train_delta 异常变大
```

典型例子：

```text
train_code  ~= 24
train_delta ~= 24
decoder_nmse ~= -0.1 dB
```

这说明 `z1` 已经跑到 fixed decoder 无法识别的码字区域。

## 已观察到的高风险配置

### 1. `lr=1e-3 + code_adapter + lambda_code=0 + lambda_delta=0`

高风险。尤其是：

```text
code_adapter=gated_lr_mlp
code_lowrank_rank=128
code_mlp_hidden=512
lr=1e-3
lambda_code=0
lambda_delta=0
```

观察到的崩溃：

```text
seed2026_transnet_transnet:
20ep  -24.039 dB
80ep  -26.190 dB
120ep -0.086 dB

seed2026_clnet_transnet:
20ep  -19.058 dB
40ep  -20.677 dB
60ep  -0.042 dB
```

原因是 code adapter 可以直接改 decoder 输入。没有 code/delta 约束时，优化只看重建误差，`z1` 可能短期下降 train loss，但逐渐偏离 `z0` 和 teacher code，最后进入 decoder 不稳定区域。

### 2. `lr=1e-3 + code_adapter + 极弱 code/delta 约束`

例如：

```text
lambda_code=1e-5
lambda_delta=1e-5
```

这个约束量级有时不够压住 adapter。观察到 `seed2026_transnet_transnet` 可先提升到约 `-26.4 dB`，随后掉到接近 `0 dB`。

原因是 `lambda_code * MSE(z1, teacher_code)` 和 `lambda_delta * MSE(z1, z0)` 相对 `rec` 的梯度约束不足，不能阻止 residual 分支增益变大。

### 3. `lr=1e-3 + code_adapter + 强 code/delta 拉扯`

例如：

```text
lambda_code=1e-3
lambda_delta=1e-3
```

在跨架构 `clnet -> seed42 transnet decoder` 上观察到重建突然崩溃。这里不是 `z1` 完全无约束飞走，而是目标冲突：

```text
重建 CSI 好
z1 靠近 teacher code
z1 不偏离 affine 后的 z0
```

跨架构时 `z0` 和 teacher code 差距大，三个目标方向不一致。较大的学习率会让训练走到 decoder 不可用区域。

## 相对稳定的配置

### 1. 纯 LoRA，固定 `z0`

```text
code_adapter=none
lr=1e-3
```

目前看比较稳定。原因是输入码字 `z0` 固定，训练只修改 decoder 的低秩增量，不会让 decoder 输入分布飞走。

### 2. `code_adapter + lr=5e-4`

比 `lr=1e-3` 稳定。适合先跑跨架构实验。

### 3. `code_adapter + lambda_code=1e-3 + lambda_delta=1e-4`

在 `seed2026_transnet_transnet` 上可以稳定跑完 400 epoch：

```text
best NMSE ~= -26.886 dB
train_code ~= 0.01295
train_delta ~= 0.01011
```

但它不一定比纯 LoRA 更好，因为 code loss 会把 `z1` 拉向 teacher code，而最佳重建码字不一定严格等于 teacher code。

## 推荐调参原则

1. 纯 LoRA 可以优先使用 `lr=1e-3`。

2. 启用 code adapter 时，优先使用：

```text
lr=5e-4
eta_min=1e-4 或 2e-4
lambda_delta >= 1e-4
```

3. 如果使用 `lr=1e-3` 训练 code adapter，建议：

```text
lambda_code=1e-4 ~ 1e-3
lambda_delta=1e-4
```

不要让 `lambda_code=lambda_delta=0` 长训。

4. 跨架构实验不要同时把 `lambda_code` 和 `lambda_delta` 都设得很强。`lambda_code=1e-3, lambda_delta=1e-3` 容易造成目标拉扯。

5. 建议后续加入保护：

```text
gradient clipping: max_norm=1.0
adapter_lr 单独设置，小于 LoRA lr
delta_norm 正则或 z1 norm clamp
true_decoder_eval 比历史 best 差 5 dB 以上时 early stop
```

## 当前结论

`lr=1e-3` 本身不是问题。真正的问题是：

```text
高 lr + 可改变输入码字的 code adapter + 弱约束/冲突约束
```

纯 LoRA 因为不改变输入码字，稳定性明显更好。code adapter 对跨架构有价值，但需要更保守的学习率和更明确的码字幅度约束。
