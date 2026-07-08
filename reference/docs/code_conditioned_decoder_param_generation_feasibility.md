# 用码字指导生成 UniversalCSI Decoder 参数的可行性分析

## 1. 问题定义

目标是结合 `JiT` 和 `Recurrent-Parameter-Generation` 两个项目的思想，研究如下任务是否可行：

```text
输入:
  Z = {z_i}_{i=1..N}, z_i in R^code_dim
  其中 z_i 是 UniversalCSI encoder 对 CSI 样本 x_i 产生的码字

输出:
  theta_D
  即 UniversalCSI decoder 的参数

希望:
  D_{theta_D}(z_i) ≈ x_i
```

在当前 UniversalCSI 默认配置中：

```text
CSI shape = (2, 32, 32)
input_dim = 2048
cr = 4
code_dim = 512
Z shape ≈ (100000, 512)
```

这个问题不是“给单条码字生成单个 CSI”，而是“给一个实验/场景/encoder 产生的码字集合，生成与该码字分布匹配的 decoder 参数”。因此输入 `Z` 应被视为一个无序集合或经验分布，而不是普通序列。

## 2. 两个参考项目能提供什么

### 2.1 JiT 的参考价值

JiT 是像素空间 diffusion/flow 风格的图像生成模型。它的核心启发不是直接拿来生成 decoder 参数，而是：

- 用连续时间 `t` 表示从噪声到目标的生成路径。
- 模型学习速度场或去噪方向。
- 条件信息通过 embedding、FiLM/adaLN 等方式调制 denoiser。
- 采样阶段从噪声出发，经过 ODE step 得到目标。

对应到 decoder 参数生成，可以写成：

```text
theta_0 ~ N(0, I)
theta_t = (1 - t) * theta_0 + t * theta_D
model(theta_t, t, condition=Z) -> d theta / dt
```

这与当前本地已有的 `decoder_param_fm/`、`decoder_generalization_fm/` 的 Flow Matching 路线一致。

### 2.2 RPG 的参考价值

RPG 的对象是神经网络参数本身。它把 checkpoint 的 state_dict 切成参数 token，再用 recurrent backbone 和一维 diffusion 生成参数。它对本问题的参考价值更直接：

```text
decoder state_dict
  -> 按 tensor/layer/token 切分
  -> 标准化
  -> 参数 token 序列
  -> diffusion / flow matching 生成
  -> 反标准化并还原 state_dict
```

但 RPG 原始设定多以 checkpoint index、任务条件或类别条件为输入；本任务需要把 `Z = (N, 512)` 这种大规模码字集合编码成条件。因此还需要一个 set encoder。

## 3. 与 UniversalCSI 的接口匹配

UniversalCSI 的 decoder 接口固定为：

```text
decoder(code): (B, code_dim) -> (B, channel, nt, nc)
```

当前支持的 decoder 包括：

- `transnet`：`Linear(code_dim, input_dim)` + 2 层 TransformerDecoder。
- `cnn_residual`：`LayerNorm(code_dim)` + `Linear(code_dim, input_dim)` + CNN refinement。
- `hybrid`：semantic projector + token projection + TransformerEncoder + CNN refinement。

第一阶段最适合固定 `transnet`，原因是：

- 参数结构最清晰。
- 当前已有 codeword 和 checkpoint 实验最多。
- decoder 参数规模可控。
- `fc_decoder.weight` 直接决定码字到 CSI 初始 token 的全局投影，是最关键的参数。

以 `channel=2, nt=32, nc=32, cr=4, d_model=64` 为例，`transnet` decoder 的主要参数包括：

```text
fc_decoder.weight: (2048, 512)
fc_decoder.bias:   (2048,)
2 层 TransformerDecoder 的 attention / FFN / norm 参数
decoder.norm.weight/bias
```

这类参数可以自然 token 化，适合 RPG 式参数生成。

## 4. 推荐建模方案

### 4.1 总体结构

推荐结构如下：

```text
全量码字集合 Z: (N, 512)
        |
        v
Set Condition Encoder
  random / SVD / set_transformer
        |
        v
condition tokens + global condition
        |
        v
Parameter Flow / Diffusion
        |
        v
decoder parameter tokens
        |
        v
detokenize + denormalize
        |
        v
UniversalCSI decoder theta_D
```

这个结构正好融合两个项目：

- JiT 提供“条件速度场生成”的训练和采样范式。
- RPG 提供“checkpoint 参数 token 化并生成”的参数建模范式。
- UniversalCSI 提供真实的 `code -> CSI` 函数评估闭环。

### 4.2 条件编码

码字集合 `Z` 是无序集合，不能简单按行当作序列喂给 Mamba/LSTM。合理方式包括：

1. `random`：固定或随机抽取 K 条码字作为条件 token。
2. `svd`：对码字矩阵中心化后取主方向和奇异值，得到低维统计摘要。
3. `set_transformer`：learnable queries 对全量码字做 cross-attention，学习 K 个代表性条件 token。

从已有结果看，`set_transformer` 更有潜力，因为它能从全量码字分布中学习任务相关统计；`random` 简单但不稳定；`svd` 保留二阶主方向，但可能丢失 decoder 需要的非线性分布信息。

### 4.3 参数生成

推荐不直接 flatten 全部参数成一个长向量，而是按 tensor 和 token 切分：

```text
theta_D state_dict
  -> tensor name
  -> layer id / tensor id / token offset
  -> token vector
```

模型输入每个参数 token 时应包含：

- 当前 noisy/interpolated 参数 token `theta_t`。
- 时间步 `t`。
- `tensor_id`、`layer_id`、`token_offset` 等 meta embedding。
- 码字集合编码得到的 condition token/global condition。

条件注入方式可从简单到复杂：

- `film`：用 global condition 生成 scale/shift，稳定且参数量较小。
- `cross_attention`：参数 token 直接 attend 到 condition tokens，表达能力更强但更难训。
- `hyper_lora`：由条件生成低秩修正，适合约束生成自由度。

## 5. 本地已有证据

当前仓库中已经存在两个高度相关的子项目：

- `decoder_param_fm/`：单对或多对 `(guide_codes, decoder checkpoint)` 的全量 decoder 参数 Flow Matching。
- `decoder_generalization_fm/`：按实验目录组织数据，研究“全量 encoder codewords -> transnet decoder 全部参数”的泛化。

这说明该方向已经不只是概念方案，而是有本地实现基础。

### 5.1 单目标重建证据

`decoder_param_fm/reports/generated_param_mse_ep1000_ep2000/generated_param_mse.md` 中，单目标 `seed42/transnet_transnet` 的参数生成结果显示：

| 实验 | Param Global MSE | CSI NMSE |
| --- | ---: | ---: |
| set_transformer + film + zscore, 2000 epoch | `3.24e-08` | `2.96 dB` |
| set_transformer + film + zscore, 1000 epoch | `7.08e-08` | `7.02 dB` |

参数 MSE 已经非常低，但 CSI NMSE 仍为正值，说明：

- 参数空间 MSE 小不等于 decoder 函数行为好。
- 小 tensor、norm/bias、decoder 输出层等敏感参数可能对 CSI 重建有放大效应。
- 单纯监督参数 endpoint 不足以保证通信指标。

### 5.2 CSI 生成质量证据

`decoder_param_fm/reports/csi_mse_analysis.md` 中，多种条件/注入/归一化组合在测试集上有如下最好结果：

```text
set_transformer + film + zscore:
  Agg NMSE = -4.12 dB
  Agg MSE  = 0.000175
```

这说明用码字指导生成 decoder 后，生成的 decoder 已经具备非随机的 CSI 重建能力。更重要的是，z-score 归一化显著优于 RMS，set_transformer 条件优于简单 random 条件。

但该结果距离正常训练的 UniversalCSI decoder 还有明显差距。固定 teacher decoder 在已有 mapper 报告中可达到约 `-29 dB` 级别，而生成 decoder 的最好结果仍只有 `-4 dB` 左右。因此目前证据支持“方向可行”，不支持“已接近可用上限”。

## 6. 可行性判断

### 6.1 单场景、单 encoder、单 decoder 架构

结论：**可行，适合作为第一阶段验证。**

设定：

```text
encoder = transnet
decoder = transnet
seed / scene 固定
输入 = train_code.pt
目标 = 同实验 best_nmse.pth 的 decoder 参数
```

该设定下，模型更像是在学习“码字集合条件下重建一个目标 decoder 参数”。只要训练足够，参数 MSE 可以很低。它适合验证：

- 参数 token 化和还原是否正确。
- 码字集合条件是否能被模型使用。
- Flow/Diffusion 是否能生成合法 decoder state_dict。
- 生成 decoder 是否能跑通 `decoder(code)` 并输出有限 NMSE。

但这个阶段不能证明泛化，因为训练和采样目标基本是同一个 decoder。

### 6.2 跨 seed 泛化

结论：**有条件可行，是第二阶段核心验证。**

设定：

```text
train:
  seed42, seed2026, ... 的 transnet_transnet 实验
test:
  held-out seed 的 transnet_transnet 实验
```

关键问题是：码字集合 `Z` 是否包含足够信息来确定对应 decoder 参数。理论上，同一个 encoder/decoder 架构在不同 seed 下可能学到等价但参数不同的解；这些解在函数空间相近，但参数空间不唯一。因此跨 seed 生成时，不应只追求参数 MSE，更应追求：

```text
D_generated(Z_test) ≈ X_test
```

也就是说，评估必须以 CSI NMSE 为主，参数 MSE 只能作为辅助。

### 6.3 跨 encoder 架构泛化

结论：**难度明显更高，暂时只能作为探索。**

不同 encoder 产生的码字分布可能差异很大，且 decoder 对码字坐标系非常敏感。对于同一个 CSI 数据集，`transnet`、`crnet`、`clnet` 等 encoder 的 code space 可能不是简单线性变换关系。用一个生成器从它们的码字集合直接生成对应 decoder 参数，需要模型同时学会：

- 识别码字分布属于哪类 encoder。
- 估计 code space 的坐标系和尺度。
- 生成与该坐标系匹配的 decoder 入口投影。
- 保持后续 Transformer/CNN 重建逻辑稳定。

如果训练样本数量只有少数 checkpoint，这个任务很容易过拟合。

### 6.4 直接生成全量 decoder 参数 vs 生成 adapter/LoRA

全量参数生成表达能力最大，但搜索空间也最大。对于 `transnet` decoder，`fc_decoder.weight` 就有 100 万级参数；全量生成时少量偏差可能显著影响输出。

更稳妥的中间路线是：

```text
固定一个强 decoder D_base
用码字集合 Z 生成 adapter / LoRA / affine correction 参数
D_adapted = D_base + Delta_theta(Z)
```

这样可以把生成空间限制在较小的增量参数上，减少无效 decoder 的概率。RPG 的参数生成思想同样适用于 LoRA/adapter 参数，而且数据需求会小得多。

## 7. 主要技术风险

### 7.1 码字集合到 decoder 参数不是一一映射

同一组 CSI 和码字可能存在多个功能等价的 decoder 参数。参数空间有置换、缩放、归一化等不唯一性。直接用参数 MSE 学习一个目标 checkpoint，可能学到的是 checkpoint 的偶然坐标，而不是函数本质。

缓解方式：

- 以 decoder 输出 NMSE 作为主评估。
- 加入 function loss：`MSE(D_generated(z_i), D_target(z_i))`。
- 加入 CSI loss：`MSE(D_generated(z_i), x_i)`。
- 优先生成低维 adapter 或关键子模块，而非全量参数。

### 7.2 参数 MSE 与 NMSE 不完全一致

已有结果已经显示，参数 MSE 可以达到 `1e-8` 量级，但 CSI NMSE 仍然较差。原因包括：

- 小 tensor 如 norm/bias 的相对误差会被后续网络放大。
- `fc_decoder.weight` 的高奇异方向会放大 code residual。
- Transformer 层对 norm 和 attention bias 较敏感。
- 不同 tensor 的函数敏感度不同，统一 MSE 权重不合理。

缓解方式：

- 按 tensor 敏感度加权参数 loss。
- 对 norm/bias 单独加权或单独建模。
- 在训练中加入 decoder forward 的函数级 loss。
- 对生成参数做短步数 decoder-only finetune。

### 7.3 条件集合太大

典型 `train_code.pt` 可能是 `(100000, 512)`，直接 attention 成本高。随机采样 K 条码字可能丢失尾部样本或分布细节。

缓解方式：

- set_transformer 使用 learnable queries cross-attention。
- SVD/PCA summary 与随机样本拼接。
- 加入均值、方差、分位数、协方差低秩特征。
- 分层采样：按 code norm、PCA 坐标或聚类中心采样。

### 7.4 训练样本数不足

RPG 类方法通常需要大量 checkpoint 数据。UniversalCSI 目前如果只有几十个实验 checkpoint，则全量参数生成器很容易记忆训练样本。

缓解方式：

- 系统化生成 checkpoint 数据集：多 seed、多 encoder、多 cr、多 decoder。
- 先只做同架构跨 seed，减少变化源。
- 使用 frozen base decoder + 小 adapter，降低目标维度。
- 数据增强：decoder 参数插值、SWA/EMA checkpoint、不同 epoch checkpoint。

## 8. 建议实验路线

### 阶段 1：单目标闭环

目标：验证完整链路正确。

```text
输入: seed42/transnet_transnet/train_code.pt
目标: seed42/transnet_transnet/best_nmse.pth decoder
模型: set_transformer + film + zscore + FM
评估:
  param MSE
  D_generated(train_code) vs train CSI NMSE
  D_generated(test_code) vs test CSI NMSE
```

通过标准：

- 生成参数可完整 load 到 `TransNetDecoder`。
- forward 无 NaN/Inf。
- NMSE 明显优于随机 decoder。

### 阶段 2：跨 seed held-out

目标：验证条件码字是否能指导泛化。

```text
train:
  多个 seed 的 transnet_transnet
test:
  held-out seed 的 transnet_transnet
```

评估重点：

- held-out seed 上的 generated decoder NMSE。
- 与最近邻训练 decoder 的 NMSE 对比。
- 与直接平均参数、随机初始化、固定 teacher decoder 的 baseline 对比。

如果生成器只比随机好但不如最近邻，说明它主要记忆训练集；如果超过最近邻，才说明码字条件真正提供了泛化信息。

### 阶段 3：生成 adapter 而非全量 decoder

目标：降低输出维度，提高稳定性。

```text
固定 D_base
生成 Delta_theta 或 adapter 参数
```

可选对象：

- `fc_decoder` 的低秩修正。
- code affine adapter。
- decoder norm/bias calibration。
- CNN refinement head。

这个阶段更可能得到接近可用的结果，因为 decoder 主体能力由 `D_base` 保证，生成器只负责适配 code space。

### 阶段 4：跨 encoder/跨场景

目标：验证真正的场景泛化。

此阶段应在前面三个阶段通过后再做。需要严格 held-out：

- held-out seed。
- held-out encoder。
- held-out scenario。
- held-out cr。

## 9. 推荐指标

不要只看参数 MSE。建议每次实验至少记录：

```text
参数层面:
  global param MSE
  per-tensor MSE / NMSE
  norm/bias tensor MSE
  fc_decoder.weight MSE

函数层面:
  MSE(D_gen(z), D_target(z))
  MSE(D_gen(z), x)
  global NMSE
  per-sample NMSE p50/p95/p99

泛化层面:
  train split NMSE
  held-out seed NMSE
  held-out encoder NMSE
  nearest-neighbor decoder baseline
  fixed teacher decoder baseline
```

其中最重要的是 held-out 条件下的 CSI NMSE。参数 MSE 只说明是否还原了某个 checkpoint，不说明 decoder 是否对码字集合真正有效。

## 10. 总体结论

用码字集合指导生成 UniversalCSI decoder 参数在工程上是可行的，当前仓库已经具备关键组件：

- UniversalCSI decoder 参数结构清楚。
- codeword 导出路径存在。
- `decoder_param_fm/` 已实现码字条件参数 Flow Matching。
- `decoder_generalization_fm/` 已实现按实验目录组织的泛化训练框架。
- JiT/RPG 分别提供了条件生成和参数 token 化的成熟范式。

但从现有结果看，这条路线目前仍处于研究验证阶段。它已经能生成“非随机、可运行、有一定重建能力”的 decoder，但距离正常训练 decoder 的 NMSE 还有较大差距。主要瓶颈不是能不能生成参数，而是生成参数是否在函数空间上与目标 decoder 等价。

最务实的判断是：

- **单目标参数重建：可行。**
- **同架构跨 seed 泛化：值得重点推进，但必须以 held-out NMSE 验证。**
- **跨 encoder/跨场景全量 decoder 生成：风险较高，需要更多 checkpoint 数据。**
- **生成 adapter/LoRA/关键子模块参数：比直接生成全量 decoder 更可能先达到可用效果。**

因此推荐路线不是立刻追求“一步生成完整 UniversalCSI decoder”，而是先用 `transnet` decoder 做跨 seed 小闭环，再转向“固定 base decoder + 码字条件 adapter/低秩修正”。这样更符合现有证据，也更容易把 NMSE 推到接近可用区间。
