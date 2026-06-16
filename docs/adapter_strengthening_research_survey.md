# Code Adapter 增强方向调研：面向 CSI 反馈码字空间迁移

日期：2026-06-15

本文目标是为当前 UniversalCSI 中“冻结 encoder + 冻结 decoder + 中间 code adapter”的效果不足问题提供外部论文和技术路线支撑。结论先行：

> 现有 `CodeAdapter(code) -> code'` 太像“后验单点翻译器”。它只在最终压缩码字上做一次残差 MLP 变换，但跨 seed、跨场景或跨架构的 CSI autoencoder 更像是两个私有码字语言之间的协议不兼容。外部工作支持三条更强路线：  
> 1. 从只改中间 code，升级到 decoder-side / multi-layer / token-level adaptation；  
> 2. 从单样本 adapter，升级到 domain/support-set 条件 adapter 或 LoRA；  
> 3. 从后验补丁，升级到训练期显式约束 shared code space。

## 1. 当前任务的关键问题

UniversalCSI 当前固定接口是：

```text
CSI x: (B, channel, nt, nc)
  -> encoder
  -> code: (B, code_dim)
  -> optional CodeAdapter
  -> decoder
  -> reconstructed CSI
```

默认 COST2100 `in`、`channel=2, nt=32, nc=32, cr=4` 时：

```text
input_dim = 2048
code_dim = 512
```

当前 `CodeAdapter` 是：

```text
code' = code + scale * MLP(LayerNorm(code))
```

其中 MLP hidden dim 是 `4 * code_dim`，最后一层零初始化。这个设计有两个优点：训练稳定、不会一开始破坏原 code。但它也有明显上限：

- 它只观察单个样本的最终 code，看不到目标 domain 的整体分布；
- 它只改 decoder 输入，不改 decoder 内部对 code 的解释方式；
- 它假设两个 code space 之间存在一个统一、平滑、样本无关的映射；
- 它没有训练期 shared latent 约束，只能在两个模型都已经形成私有协议后再补救。

本地实验已经说明这个上限很真实：

| 方法 | 起点 NMSE | 最好 NMSE | 现象 |
|---|---:|---:|---|
| 跨 seed 直接重组 `transnet_hybrid` | `+20 ~ +31 dB` | 不可用 | decoder 基本读不懂另一个 seed 的 code |
| code adapter，无 teacher | `+23.854 dB` | `-20.646 dB` | 能救，但离同 seed baseline `-28 dB` 约差 `7~8 dB` |
| code adapter，code-only | `+23.854 dB` | `-21.059 dB` | 当前 adapter 上限仍明显不足 |
| LoRA 只改 `HybridDecoder.token_projection` | `+28.634 dB` | `-8.043 dB` | 只改一个入口线性层远远不够 |
| 只训 `TransNetDecoder.fc_decoder` | `+15.461 dB` | `-21.627 dB` | 比入口 LoRA 强，但仍有天花板 |
| 固定 decoder 重新训练 encoder | 不适用 | `-28.61 ~ -28.66 dB` | decoder 可以复用，关键是 encoder 必须训练进目标 decoder 可读的 code space |

因此，后续研究重点不应只是“把 MLP 做大”，而应围绕以下问题设计：

```text
如何让 encoder 输出进入 decoder 可读的公共码字语言？
如何用少量参数改 decoder 的解释规则？
如何利用一个 domain/session 的一组 support code 生成适配参数？
```

## 2. CSI Feedback 相关工作

### 2.1 CsiNet：CSI 反馈 autoencoder 范式

论文：[Deep Learning for Massive MIMO CSI Feedback](https://arxiv.org/abs/1712.08919)

原理：

- 把 CSI 反馈看成 autoencoder 压缩重建问题；
- UE 侧 encoder 输出低维 codeword；
- BS 侧 decoder 从 codeword 重建 CSI；
- 端到端优化 MSE/NMSE。

对当前任务的支撑：

- CsiNet 本身证明了“encoder + decoder 联合训练”可以学习高效 CSI code；
- 但也隐含了一个问题：codeword 语义由 encoder 和 decoder 共同定义，并不天然可解释、可交换；
- 你现在遇到的跨 seed 崩溃，正是 autoencoder latent space 私有化的典型结果。

可能效果：

- 如果只追求单模型 NMSE，继续端到端训练有效；
- 如果追求跨 encoder/decoder 组合，必须额外约束 latent/code space。

### 2.2 CsiNet-LSTM：利用时序相关性，而不是只看单样本

论文：[Deep Learning-based CSI Feedback Approach for Time-varying Massive MIMO Channels](https://arxiv.org/abs/1807.11673)

原理：

- 在 CsiNet 基础上引入 LSTM；
- 利用时间变化信道中的相关性；
- 不把每个 CSI 样本完全独立处理。

对当前任务的支撑：

- 当前 adapter 是 sample-wise 的，只看一个 code；
- 如果场景/domain/session 内 code 有稳定统计结构，那么 adapter 应该读取一组 calibration/support code，而不是只读单样本；
- 这支持“domain-level adapter / support-set conditioned LoRA”路线。

可能效果：

- 对同一场景、同一路径统计、同一 session 的迁移更有帮助；
- 对完全无关场景可能收益有限。

### 2.3 CRNet：多分辨率特征，说明 decoder 侧结构很关键

论文：[Multi-resolution CSI Feedback with deep learning in Massive MIMO System](https://arxiv.org/abs/1910.14322)

原理：

- 用多分辨率卷积分支提取 CSI 特征；
- 在相近复杂度下优于 CsiNet；
- 强调 CSI 重建不是简单全连接映射，而需要结构化 spatial/frequency 特征恢复。

对当前任务的支撑：

- 如果 decoder 内部重建路径才是关键，那么只在 code 入口加 MLP 很可能不足；
- 更合理的是对 decoder 的 `fc_decoder`、token projection、FFN、attention projection、CNN refinement head 做多层适配。

可能效果：

- multi-layer decoder adapter 或 LoRA 可能明显强于单个 code adapter；
- 尤其适合 `hybrid` decoder，因为入口 token projection 之后还有 Transformer/CNN 重建逻辑。

### 2.4 CsiNet+ / 多码率 CSI：反馈码率变化需要结构化共享

论文：[Convolutional Neural Network based Multiple-Rate Compressive Sensing for Massive MIMO CSI Feedback](https://arxiv.org/abs/1906.06007)

原理：

- 提出 CsiNet+ 和多码率反馈机制；
- 关注不同 compression rate 下的网络复用和存储开销；
- 通过共享结构支持多个反馈率。

对当前任务的支撑：

- 多码率本质上也是“不同 code distribution/维度设置下复用 decoder”；
- 它说明如果一开始按共享结构设计，模型可以更好适配不同反馈设定；
- 这支持训练期引入 shared code regularization，而不是事后强行翻译。

可能效果：

- 对 `cr=4/8/16` 多压缩率统一 decoder 有参考价值；
- 可以设计 shared decoder + rate/domain adapter。

### 2.5 CLNet：复值输入和注意力说明 CSI 有强物理结构

论文：[CLNet: Complex Input Lightweight Neural Network designed for Massive MIMO CSI Feedback](https://arxiv.org/abs/2102.07507)

原理：

- 用伪复值输入层处理实部/虚部；
- 用 attention 增强关键 CSI 区域；
- 在精度和复杂度之间做权衡。

对当前任务的支撑：

- 当前 code adapter 不知道实部/虚部、角延迟结构、稀疏路径等物理含义；
- 如果要增强 adapter，可以把 code reshape/tokenize，或把 encoder 中间特征、path/cluster 统计作为条件；
- 纯 MLP code 翻译缺少 CSI inductive bias。

可能效果：

- 在 WAIRD、DeepMIMO、COST2100 跨场景迁移中，加入物理结构统计可能比扩大 MLP 更稳。

### 2.6 STNet / Transformer CSI：token 级交互值得适配

论文：[A Spatially Separable Attention Mechanism for massive MIMO CSI Feedback](https://arxiv.org/abs/2208.03369)

原理：

- 用轻量 Transformer 处理 CSI feedback；
- 通过空间可分离注意力降低复杂度；
- 说明 Transformer 对 CSI feedback 有竞争力，但 encoder 侧复杂度需要控制。

对当前任务的支撑：

- 对 `TransNet` / `hybrid` 类 decoder，code 进入 token projection 后才展开成重建 token；
- 只适配输入 code，可能错过 token 级关系；
- 可以尝试 token adapter、prefix token、decoder FFN LoRA、attention LoRA。

可能效果：

- 对 `hybrid` decoder，multi-layer token adaptation 预计强于单层 `token_projection` LoRA；
- 对 `transnet` decoder，`fc_decoder` + Transformer block adapter 可能比只训 `fc_decoder` 更强。

### 2.7 CSI feedback 综述：泛化、在线训练、标准化是核心难点

论文：[Overview of Deep Learning-based CSI Feedback in Massive MIMO Systems](https://arxiv.org/abs/2206.14383)

原理：

- 系统梳理 DL-based CSI feedback；
- 讨论 architecture、bit-level feedback、joint communication design；
- 明确指出 generalization、online training、complexity、standardization 是实际部署难点。

对当前任务的支撑：

- 你的问题不只是模型小技巧，而是 CSI feedback 部署中的泛化/标准化问题；
- 多厂家、多场景、多 seed 的 decoder 复用，需要“标准化 code language”或轻量在线适配机制。

可能效果：

- 把当前课题表述为“CSI feedback code space standardization / decoder-side adaptation”更容易成立。

### 2.8 UniversalNet：先做输入标准化，减轻模型迁移压力

论文：[Generalizing Deep Learning-Based CSI Feedback in Massive MIMO via ID-Photo-Inspired Preprocessing](https://arxiv.org/abs/2409.13494)

原理：

- 通过 ID-photo-inspired preprocessing 标准化不同环境下的 CSI 输入格式；
- 在 sparse domain 做轻量对齐；
- 尽量不改已有 CSI feedback 网络权重。

对当前任务的支撑：

- 如果不同数据集/场景的 CSI 输入分布没有对齐，后面的 code adapter 会背负过重任务；
- 在 encoder 前或 code 前加入 domain normalization / sparsity alignment 可能比只在 code 后修补更有效。

可能效果：

- 对跨 COST2100/WAIRD/DeepMIMO 的泛化更重要；
- 对同数据集跨 seed 的私有协议问题帮助有限，但可作为多场景统一框架的前置模块。

### 2.9 EG-CsiNet：用物理解释做环境泛化

论文：[Generalizable Learning for Massive MIMO CSI Feedback in Unseen Environments](https://arxiv.org/abs/2512.22840)

相关早期版本：[Enhancing Environment Generalizability for Deep Learning-Based CSI Feedback](https://arxiv.org/abs/2507.06833)

原理：

- 显式建模不同环境之间的 distribution shift；
- 把 shift 拆成多径结构变化和单路径响应变化；
- 用多 cluster decoupling 和 fine-grained alignment 增强泛化；
- 报告中提到相对 SOTA 可降低超过 `3 dB` 的 generalization error。

对当前任务的支撑：

- 这直接支持“不要只让 adapter 自己学全部 domain shift”；
- 可以把 support set 的 code/CSI 做 cluster/path 统计，再生成 adapter 或 LoRA；
- teacher code 直接 MSE 对齐可能太粗，物理分解后的 alignment 更合理。

可能效果：

- 对跨环境迁移比跨 seed 更有针对性；
- 可作为 WAIRD/DeepMIMO 多场景实验的重要理论支撑。

### 2.10 Fed-PELAD：个性化 encoder + LoRA-adapted shared decoder

论文：[Fed-PELAD: Communication-Efficient Federated Learning for Massive MIMO CSI Feedback with Personalized Encoders and a LoRA-Adapted Shared Decoder](https://arxiv.org/abs/2510.25181)

原理：

- UE 侧保留 personalized encoders；
- BS 侧维护 shared decoder；
- 用 LoRA 适配 shared decoder；
- 通过交替冻结和学习率比例校准增强 LoRA 聚合稳定性；
- 摘要报告在异构条件下相比常规方法降低 uplink communication cost 到 `42.97%`，并带来 `1.2 dB` CSI feedback accuracy gain。

对当前任务的支撑：

- 这是最接近当前目标的通信论文之一：不同 encoder / 异构数据 + shared decoder + LoRA；
- 它支持 decoder-side LoRA，而不是只做 code adapter；
- 它也支持“个性化 encoder 不必完全统一，但 shared decoder 需要可适配”。

可能效果：

- 对多厂家 UE encoder + BS shared decoder 的叙事非常强；
- 可以作为后续方案的直接通信领域支撑：`personalized encoder + shared decoder + per-domain LoRA`。

### 2.11 One-sided CSI feedback：实际部署中 encoder/decoder 联合维护很难

论文：[Deep Learning for CSI Feedback: One-Sided Model and Joint Multi-Module Learning Perspectives](https://arxiv.org/abs/2405.05522)

原理：

- 指出主流 two-sided DL CSI feedback 需要 UE 侧和 BS 侧模型强耦合；
- 这种耦合会带来跨 vendor 协作、模型维护和责任划分问题；
- 讨论 one-sided model 和 joint multi-module learning。

对当前任务的支撑：

- 你当前做的“冻结某侧，只适配另一侧或中间码字”正是 two-sided 强耦合的工程问题；
- 论文从系统部署角度支持“解耦 UE encoder 和 BS decoder”的研究动机；
- 但本地实验说明，简单 code adapter 还不足以完成这种解耦。

可能效果：

- 可作为论文/汇报中的动机部分：为什么要研究可插拔 encoder/decoder、shared decoder、adapter。

## 3. 通用 Adapter / PEFT 技术

### 3.1 Houlsby Adapter：每层插入瓶颈 adapter

论文：[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)

原理：

- 冻结预训练模型；
- 在 Transformer 层中插入小型 bottleneck adapter；
- 每个任务只训练少量新增参数；
- 摘要中报告在 GLUE 上接近 full fine-tuning，新增约 `3.6%` 参数。

对当前任务的支撑：

- 当前 UniversalCSI 的 adapter 只在 encoder 和 decoder 之间插一次；
- Houlsby adapter 的成功经验是“多层小 adapter”，不是“单点大 adapter”；
- 如果 decoder 内部多层都形成了私有 code 解释规则，入口 adapter 很难完全修正。

建议实验：

```text
decoder token_projection adapter
+ each Transformer FFN adapter
+ optional attention output adapter
+ optional CNN refinement adapter
```

可能效果：

- 参数量仍可控；
- 比单点 code adapter 更接近 full fine-tuning；
- 需要避免过拟合，可先只加 decoder-side。

### 3.2 AdapterHub / Pfeiffer Adapter：adapter 组合和可插拔生态

论文：[AdapterHub: A Framework for Adapting Transformers](https://arxiv.org/abs/2007.07779)

原理：

- 把 adapter 作为可共享、可插拔模块；
- 支持不同任务/语言动态插入；
- 强调 adapter 的工程可复用性。

对当前任务的支撑：

- UniversalCSI 可以把 adapter 组织成：

```text
domain adapter: COST2100-in / COST2100-out / WAIRD / DeepMIMO
encoder adapter: csinet / transnet / crnet / vendor-A
rate adapter: cr=4 / cr=8 / cr=16
```

- 不同 adapter 可以组合，而不是只训练一个全局 code adapter。

可能效果：

- 对多场景、多 encoder 实验管理更清晰；
- 适合作为“BS 侧共享 decoder + 插拔式适配模块”的工程路线。

### 3.3 AdapterFusion：先学多个 adapter，再组合

论文：[AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://arxiv.org/abs/2005.00247)

原理：

- 第一阶段为不同任务训练 task-specific adapters；
- 第二阶段冻结这些 adapter，学习如何融合它们；
- 避免 catastrophic forgetting 和多任务数据配比困难。

对当前任务的支撑：

- 可以先为不同 domain/encoder 训练 adapter：

```text
adapter_COST2100_in
adapter_WAIRD
adapter_DeepMIMO
adapter_encoder_csinet
adapter_encoder_transnet
```

- 再学习一个 gate/fusion 模块根据 support code 组合已有 adapter。

可能效果：

- 当目标场景接近已有几个场景的组合时，可能优于单 adapter；
- 对完全新场景仍需要 support set 或少量校准样本。

### 3.4 LoRA：低秩权重更新，适合 decoder-side 小参数适配

论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

原理：

- 冻结原权重 `W`；
- 训练低秩增量 `ΔW = B A`；
- 推理时可合并到原权重，无额外延迟；
- 摘要中报告可把 GPT-3 类模型的可训练参数减少 `10000x`，训练显存约降 `3x`。

对当前任务的支撑：

- CSI decoder 的关键线性层非常适合 LoRA；
- 但本地只改 `HybridDecoder.token_projection` 效果差，说明 LoRA 的位置和覆盖层数不够；
- LoRA 应该优先放在 decoder 能解释 code 的关键路径，而不是只放一个入口层。

建议实验优先级：

```text
1. TransNetDecoder.fc_decoder LoRA
2. HybridDecoder.token_projection + decoder Transformer FFN LoRA
3. attention q/k/v/out LoRA
4. CNN refinement head 1x1/3x3 conv LoRA 或 adapter
5. encoder-side LoRA 作为辅助，不作为主线
```

可能效果：

- 对错配起点 `+15 ~ +28 dB`，单层 LoRA 不够；
- 多层 decoder LoRA 有机会接近 `fc_decoder` 训练，甚至超过 `-21 dB`；
- 若仍离 `-28 dB` 很远，说明必须引入训练期 shared code 约束。

### 3.5 AdaLoRA：不要平均分配 rank，应按层重要性分配

论文：[AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)

原理：

- 普通 LoRA 给每层相同 rank；
- AdaLoRA 根据重要性动态分配参数预算；
- 用 SVD 形式参数化增量，并裁剪不重要奇异值；
- 摘要报告低预算场景下优于固定预算 LoRA。

对当前任务的支撑：

- CSI decoder 中不同层对 code space 适配的重要性不同；
- 入口 `token_projection` 可能不是唯一瓶颈；
- 可以先全层低 rank 训练，再根据 delta norm、梯度、NMSE ablation 找关键层。

可能效果：

- 在参数预算固定时比统一 rank 更稳；
- 可解释哪些 decoder 层最需要适配。

### 3.6 QLoRA：部署和显存优化，不是解决 adapter 弱的主因

论文：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

原理：

- 冻结 4-bit 量化基座；
- 通过 LoRA 反传训练；
- 摘要报告 65B 模型可在单张 48GB GPU 上微调，并保留接近 16-bit fine-tuning 的效果。

对当前任务的支撑：

- 如果未来 BS decoder 很大，QLoRA 可降低训练/部署成本；
- 但当前 adapter 弱主要不是显存问题，而是适配位置和目标不对。

可能效果：

- 对当前 UniversalCSI 小模型不是最高优先级；
- 可作为工程部署补充。

### 3.7 IA3：学习通道/激活缩放，适合轻量 domain calibration

论文：[Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)

原理：

- IA3 通过学习向量缩放激活；
- 参数量比 adapter/LoRA 更小；
- 摘要中介绍其在 few-shot PEFT 中效果强、成本低。

对当前任务的支撑：

- CSI domain shift 可能有一部分是通道/频域能量尺度变化；
- 对 decoder hidden tokens 加 learned scale/gate 可能比全 MLP 更稳；
- 可作为 `LayerNorm affine + channel gate + LoRA` 的低成本组合。

可能效果：

- 对幅度分布变化、domain normalization 有用；
- 对跨 seed 私有协议的旋转/置换/非线性变换可能不够。

### 3.8 ReFT / LoReFT：直接干预 representation，而不只改权重

论文：[ReFT: Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592)

原理：

- 冻结模型；
- 学习对隐藏表示的低秩线性子空间干预；
- 摘要报告 LoReFT 在多类任务上比 LoRA 更参数高效。

对当前任务的支撑：

- 你的 `CodeAdapter` 本质上就是 representation intervention，但只作用在最外层 code；
- ReFT 的启发是：应在 decoder 内部关键 hidden states 上做低秩干预；
- 对 `hybrid` decoder，可在 token projection 后、每个 Transformer block 后、refinement 前做 intervention。

可能效果：

- 比 weight-side LoRA 更直接修正“decoder 正在读到什么表示”；
- 可以和 code adapter 并行，形成 multi-point representation adaptation。

### 3.9 Prefix / Prompt Tuning：给 decoder 加可学习上下文 token

论文：

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602)

原理：

- 冻结主模型；
- 训练少量连续 prompt/prefix tokens；
- 让模型在每层或输入端 attend 到这些虚拟 token；
- Prefix-Tuning 摘要报告训练约 `0.1%` 参数即可接近 full-data fine-tuning，并在低数据场景有优势。

对当前任务的支撑：

- 对 Transformer decoder，可以给每个 domain/encoder 一组 learnable prefix tokens；
- prefix 表示“这个 code 来自哪个 encoder/domain，该如何解释”；
- 比单点 code adapter 更适合 token-based decoder。

可能效果：

- 对 `hybrid` / `transnet` decoder 有较强可试性；
- 对纯 CNN decoder 不如 LoRA/adapter 直接。

### 3.10 Ladder Side-Tuning：旁路网络可能比插层 adapter 更省显存

论文：[LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](https://arxiv.org/abs/2206.06522)

原理：

- 冻结大模型；
- 训练小 side network；
- 通过 ladder connections 读取 backbone 中间激活；
- 摘要报告相近参数量下比 Adapter/LoRA 更省训练显存，并在低显存下精度更好。

对当前任务的支撑：

- 可以保留 frozen decoder 主路径，同时加一个轻量 residual reconstruction side branch；
- side branch 输入可以是 code、decoder tokens、中间 feature；
- 最终输出：

```text
H_hat = decoder_frozen(code') + side_refiner(code, tokens, H_hat)
```

可能效果：

- 对当前 `-20 ~ -21 dB` 的残差错误可能有效；
- 如果 frozen decoder 主路径已经严重误读 code，side branch 需要足够强，否则只是修边。

## 4. Latent Space / Model Stitching / 对齐理论

### 4.1 Model Stitching：两个网络表示相似，也需要 stitching layer

论文：[Understanding image representations by measuring their equivariance and equivalence](https://arxiv.org/abs/1411.5908)

原理：

- 通过在两个网络中间插入 transformation/stitching layer，测试表示是否等价；
- 如果一个简单 stitching layer 能连接两个表示，说明它们捕获了相近信息。

对当前任务的支撑：

- 当前 code adapter 正是 CSI autoencoder 的 stitching layer；
- adapter 能把 `+23 dB` 拉到 `-21 dB`，说明 code 中有可恢复信息；
- 但离 `-28 dB` 远，说明两个 code space 不是简单等价，或 decoder 需要内部协同适配。

可能效果：

- 可以系统评估 linear / affine / MLP / residual / invertible adapter 的 stitching 能力；
- 如果更强 stitching 仍到不了 baseline，说明需要训练期约束。

### 4.2 Stitchable Neural Networks：分层 stitching 比单点 stitching 强

论文：[Stitchable Neural Networks](https://arxiv.org/abs/2302.06586)

原理：

- 把同一模型族的不同规模 anchor 网络按层切开重组；
- 学习 stitching layers 连接不同 anchor 的中间激活；
- 用少量训练即可得到多种 accuracy/efficiency trade-off。

对当前任务的支撑：

- 不必只在 encoder/decoder 边界 stitch；
- 可以在 decoder 内部多个位置 stitch：

```text
code space
token space
Transformer block hidden space
CNN feature space
output residual space
```

可能效果：

- 多点 stitching 更可能逼近 full fine-tuning；
- 也能定位具体失配发生在哪一层。

### 4.3 Functional Latent Alignment：只对齐 code MSE 可能不是正确目标

论文：[Model Stitching by Functional Latent Alignment](https://arxiv.org/abs/2505.20142)

原理：

- 重新审视 model stitching；
- 认为 stitching 不应只匹配某层表示本身，而应考虑功能性 latent alignment；
- 借鉴知识蒸馏思想，让 stitched 表示在后续功能上对齐。

对当前任务的支撑：

- 当前 teacher code MSE 可能伤害重建，说明“靠近 teacher code”不一定等于“decoder 可用”；
- 更合理的 teacher 目标是：

```text
decoder_target(adapter(code_source)) 的中间 token / 输出功能
```

而不是只做：

```text
adapter(code_source) ≈ teacher_code
```

可能效果：

- 用 decoder feature distillation、Jacobian/gradient matching、output residual loss，可能比 raw code MSE 更稳；
- 可解释为什么固定 `lambda=0.1` teacher code 会明显退化。

### 4.4 Git Re-Basin：独立训练模型有置换/对称性，不对齐就不能平均或重组

论文：[Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836)

原理：

- 独立训练神经网络可能功能相近，但隐藏单元存在置换对称；
- 需要先对齐 hidden units，才能在权重空间合并或连接；
- 论文提出通过 permutation alignment 使模型进入相近 basin。

对当前任务的支撑：

- 跨 seed encoder/decoder 崩溃可能包含 hidden/code 维度置换、旋转、缩放、符号翻转；
- 单个 MLP 可以学一部分，但未必能让 decoder 内部所有层同步对齐；
- 先做 code space linear/procrustes/CCA 诊断很重要。

建议诊断：

```text
1. 收集同一 x 的 code_A, code_B
2. 做 mean/std 对齐
3. 做 Orthogonal Procrustes
4. 做 full linear regression
5. 做 CCA/SVCCA 相似性
6. 比较 decoder_B(T(code_A)) 的 NMSE
```

可能效果：

- 如果线性/正交对齐已接近 adapter 上限，说明主要是全局几何错位；
- 如果线性完全不行而 MLP 有效，说明非线性私有协议更重；
- 如果任何 code 对齐都不行，说明 decoder 内部也要适配。

### 4.5 SimCLR / Contrastive Learning：训练期统一表示比事后对齐更稳

论文：[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

原理：

- 通过对比学习让不同增强视图的表示接近；
- projection head 对表示质量很关键；
- 学到的表示可用于下游任务。

对当前任务的支撑：

- 可以把同一个 CSI 样本经过不同 encoder seed / architecture 的 code 作为 positive pair；
- 训练 shared latent projector，使不同 encoder 输出进入公共空间；
- decoder 只读 shared latent，而不是每个 encoder 的私有 code。

可能效果：

- 对多 encoder、多 seed 统一 code space 可能比后验 adapter 更强；
- 需要训练时保留多个 encoder 或离线导出 code pairs。

## 5. Support Set / Hypernetwork / 生成式 Adapter

### 5.1 Deep Sets：domain 条件应该是集合，不是固定顺序矩阵

论文：[Deep Sets](https://arxiv.org/abs/1703.06114)

原理：

- 对集合输入，模型应满足 permutation invariance；
- 典型形式是：

```text
f({x_1, ..., x_K}) = rho(sum_i phi(x_i))
```

对当前任务的支撑：

- 一个 domain/session 的 support code 没有天然顺序；
- 不应该 flatten `(K, code_dim)` 后喂 MLP；
- 应该用 Deep Sets / pooling 得到 domain embedding。

可能效果：

- 对小 K calibration 样本稳；
- 实现简单，可作为第一版 domain-conditioned adapter。

### 5.2 Set Transformer：support code 之间的关系也重要

论文：[Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825)

原理：

- 用 attention 建模集合元素之间的相互关系；
- 通过 inducing points 降低复杂度；
- 适合 multiple instance、point cloud、few-shot 等集合任务。

对当前任务的支撑：

- CSI support set 不是简单均值，可能包含多个 path/cluster/UE 子分布；
- Set Transformer 可以从 `K` 个 code 中提取 domain-level latent；
- 再用该 latent 生成 adapter/LoRA/gate。

可能效果：

- 比 Deep Sets 更强；
- 当 K 较小且 domain 内有多模态结构时更有优势。

### 5.3 Perceiver：把长 code 矩阵压缩成固定 latent tokens

论文：[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)

原理：

- 用少量 latent queries cross-attend 大规模输入；
- 把高维/长序列输入压缩进固定 latent bottleneck；
- 摘要强调可扩展到数十万输入元素。

对当前任务的支撑：

- 如果 support set 是 `(K, code_dim)`，甚至 `(K, tokens, d_model)`，Perceiver 比 flatten 更合理；
- 它可以输出固定长度 domain tokens，供 LoRA generator 或 prefix generator 使用。

可能效果：

- 适合后续“整体矩阵条件生成 LoRA”；
- 第一版可用小 Perceiver，不必上 diffusion。

### 5.4 HyperNetworks：从 domain embedding 生成 adapter/LoRA 权重

论文：[HyperNetworks](https://arxiv.org/abs/1609.09106)

原理：

- 一个网络生成另一个网络的权重；
- 可看作跨层/跨任务的 relaxed weight sharing；
- 摘要中展示了在 RNN/CNN 中生成非共享权重的可行性。

对当前任务的支撑：

- 用 support code 得到 domain embedding；
- hypernetwork 生成：

```text
CodeAdapter 参数
decoder LoRA A/B
IA3 gate
prefix tokens
```

- 这样 adapter 不再是所有 domain 一个固定函数，而是按 domain/session 变化。

可能效果：

- 对多场景、多厂家、多 encoder 更强；
- 训练难度比 static adapter 高，应先做 static per-domain LoRA 上限实验。

## 6. Domain Adaptation / 分布对齐

### 6.1 DANN：表示应该让 domain classifier 分不出来

论文：[Domain-Adversarial Neural Networks](https://arxiv.org/abs/1412.4446)

原理：

- 学到的表示既要完成主任务，又要让 domain classifier 难以判断来源；
- 通过 adversarial objective 让 source/target 表示域不变。

对当前任务的支撑：

- 如果目标是 shared code space，可以在训练期加入 domain adversarial loss：

```text
code -> reconstruction
code -> domain classifier with gradient reversal
```

- 让不同 seed/domain/encoder 的 code 更难区分。

可能效果：

- 对跨环境泛化有帮助；
- 对 CSI 中有物理差异的 domain，要小心过度抹掉有用 domain 信息。

### 6.2 Deep CORAL：对齐二阶统计，适合 code distribution 诊断

论文：[Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719)

原理：

- 对齐 source/target feature 的 covariance；
- 用简单的二阶统计 loss 缩小 domain gap。

对当前任务的支撑：

- 可以先测不同 encoder code 的 mean/cov 差异；
- 在 adapter 或 encoder 训练中加入 mean/cov alignment；
- 比 raw teacher code MSE 更弱、更稳，不强迫每个样本完全等于 teacher code。

可能效果：

- 可能缓解 fixed teacher code lambda 过强导致的退化；
- 对非线性协议不够，但适合作为低风险正则。

### 6.3 MMD / DAN：分布级对齐比逐样本 code MSE 更温和

论文：[Learning Transferable Features with Deep Adaptation Networks](https://arxiv.org/abs/1502.02791)

原理：

- 用 Maximum Mean Discrepancy 度量 source/target 表示分布差异；
- 在深层特征上加入分布对齐损失。

对当前任务的支撑：

- teacher code MSE 要求 `adapter(code_A_i) ≈ code_B_i`；
- MMD 只要求整体分布接近，不强迫样本级一一对应；
- 对不同 seed 的 code language，分布级对齐可能更稳。

可能效果：

- 可以作为 adapter 训练的 auxiliary loss；
- 单独用 MMD 不保证 decoder 可读，必须配合 reconstruction loss。

## 7. 对当前 UniversalCSI 最有价值的路线排序

### 路线 A：Multi-layer decoder-side adapter / LoRA

优先级：最高。

原因：

- 本地实验显示只训 `fc_decoder` 比只改 `token_projection` LoRA 强；
- 外部 adapter/LoRA 工作普遍不是只在一个边界插模块，而是在多层关键位置插模块；
- CSI decoder 是解释 code 的主体，错配时只改 code 入口不够。

建议实现：

```text
--adapt_decoder_layers fc_decoder,token_projection,ffn,attn,refiner
--adapter_type lora|bottleneck|ia3|reft
--lora_rank 8/16/32/64
--adapter_hidden_ratio 0.25/0.5/1.0
```

建议实验：

| 实验 | 目的 |
|---|---|
| 只训 `fc_decoder` | 保留当前 strong baseline |
| `fc_decoder` LoRA | 看低秩是否足够 |
| `fc_decoder + FFN` LoRA | 看 decoder 内部适配收益 |
| `token_projection + FFN + attention` LoRA | 对 hybrid decoder 做完整 token path 适配 |
| bottleneck adapter vs LoRA vs IA3 | 比较不同 PEFT 形式 |

预期：

- 有希望超过当前 `-21.6 dB`；
- 若仍明显低于 `-28 dB`，说明后验 adapter 路线存在根本上限。

### 路线 B：Functional distillation，而不是 raw teacher code MSE

优先级：最高。

原因：

- 本地 fixed `lambda=0.1` teacher code 明显伤害；
- model stitching / functional latent alignment 支持“功能对齐比表示逐点 MSE 更重要”。

建议 loss：

```text
L = L_recon(H_hat, H)
  + alpha * L_decoder_feature(student_tokens, teacher_tokens)
  + beta  * L_output_distill(H_hat_student, H_hat_teacher)
  + gamma * L_distribution_align(code_student, code_teacher)
```

其中 feature 可以取：

```text
decoder token_projection output
Transformer block output
fc_decoder output token
refinement head input
```

预期：

- 比直接 teacher code MSE 稳；
- 能解释“teacher code 强约束为什么伤害重建”。

### 路线 C：Support-set conditioned domain adapter / LoRA generator

优先级：高。

原因：

- 单样本 adapter 不知道当前 domain 的整体 code 分布；
- CSI 场景迁移通常是 domain/session 级变化；
- Deep Sets / Set Transformer / Perceiver / HyperNetworks 都支持集合条件生成参数。

第一版建议：

```text
support codes C_s: (K, code_dim)
  -> DeepSets/SetTransformer
  -> domain embedding z_d
  -> generate IA3 gates or LoRA scales
  -> decoder adaptation for query codes
```

先不要直接生成大矩阵，先生成小参数：

```text
LayerNorm affine
IA3 gate
LoRA scaling
LoRA B only
prefix tokens
adapter routing weights
```

预期：

- 对跨场景/跨 vendor 更有潜力；
- 对同数据集跨 seed 也可用，但需要多个 seed/domain 训练生成器。

### 路线 D：训练期 shared code space 约束

优先级：高，但需要重新训练。

原因：

- 固定 decoder 重新训练 encoder 能回到 `-28.6 dB`，说明问题可在训练期解决；
- 后验 adapter 只能补救已形成的私有协议；
- shared latent 约束能从源头减少私有化。

建议方式：

```text
multi-encoder shared decoder training
code contrastive loss
code covariance alignment
domain adversarial loss
code whitening / normalization
teacher decoder functional distillation
```

预期：

- 长期最可能接近 baseline；
- 工程成本高于后验 adapter。

### 路线 E：Code space 诊断与线性对齐基线

优先级：高，作为所有方案前置诊断。

建议做：

```text
mean/std alignment
Orthogonal Procrustes
full linear regression
low-rank linear map
CCA/SVCCA
MMD/CORAL distance
nearest-neighbor code retrieval
decoder sensitivity/Jacobian
```

目的：

- 判断当前失配主要是缩放、旋转、置换、低秩偏移、非线性变换，还是 decoder 内部语义错配；
- 避免盲目堆 adapter。

预期：

- 如果线性映射能接近 MLP adapter，说明先做规范化和正交对齐即可；
- 如果线性远差于 MLP，说明非线性 adapter 有意义；
- 如果 MLP 也上不去，说明要 decoder-side 或训练期约束。

## 8. 论文与技术手段总表

| 类别 | 论文/技术 | 链接 | 可迁移点 | 对当前任务的价值 |
|---|---|---|---|---|
| CSI feedback | CsiNet | https://arxiv.org/abs/1712.08919 | autoencoder CSI 压缩重建 | 说明 code 是联合训练私有协议 |
| CSI feedback | CsiNet-LSTM | https://arxiv.org/abs/1807.11673 | 利用时序/样本相关 | 支持 support-set/domain 级适配 |
| CSI feedback | CRNet | https://arxiv.org/abs/1910.14322 | 多分辨率重建 | 支持 decoder-side 多层适配 |
| CSI feedback | CsiNet+ / multiple-rate | https://arxiv.org/abs/1906.06007 | 多码率共享结构 | 支持 shared decoder / rate adapter |
| CSI feedback | CLNet | https://arxiv.org/abs/2102.07507 | 复值输入、attention | 支持物理结构/注意力条件 |
| CSI feedback | STNet | https://arxiv.org/abs/2208.03369 | 轻量 Transformer | 支持 token-level adapter |
| CSI feedback | CSI feedback overview | https://arxiv.org/abs/2206.14383 | 泛化、复杂度、标准化 | 支持课题动机 |
| CSI feedback | UniversalNet | https://arxiv.org/abs/2409.13494 | 输入标准化、稀疏域对齐 | 支持 preprocessing/domain normalization |
| CSI feedback | EG-CsiNet | https://arxiv.org/abs/2512.22840 | 物理分解、环境泛化 | 支持 physics-aware alignment |
| CSI feedback | Fed-PELAD | https://arxiv.org/abs/2510.25181 | personalized encoder + LoRA shared decoder | 与当前方向高度相关 |
| CSI feedback | One-sided CSI feedback | https://arxiv.org/abs/2405.05522 | UE/BS 解耦部署问题 | 支持可插拔 encoder/decoder 动机 |
| Adapter | Houlsby Adapter | https://arxiv.org/abs/1902.00751 | 多层 bottleneck adapter | 反证单点 adapter 不够 |
| Adapter | AdapterHub | https://arxiv.org/abs/2007.07779 | adapter 可插拔生态 | 支持 domain/encoder adapter 管理 |
| Adapter | AdapterFusion | https://arxiv.org/abs/2005.00247 | 多 adapter 融合 | 支持多 domain adapter fusion |
| PEFT | LoRA | https://arxiv.org/abs/2106.09685 | 低秩权重增量 | 支持 decoder-side LoRA |
| PEFT | AdaLoRA | https://arxiv.org/abs/2303.10512 | 自适应 rank 分配 | 支持按 decoder 层重要性分配参数 |
| PEFT | QLoRA | https://arxiv.org/abs/2305.14314 | 量化 + LoRA | 部署/显存优化 |
| PEFT | IA3 / T-Few | https://arxiv.org/abs/2205.05638 | 激活缩放 | 支持轻量 domain gate |
| PEFT | ReFT / LoReFT | https://arxiv.org/abs/2404.03592 | representation intervention | 支持 decoder hidden state adapter |
| PEFT | Prefix-Tuning | https://arxiv.org/abs/2101.00190 | 虚拟 prefix token | 支持 decoder prompt/prefix |
| PEFT | P-Tuning v2 | https://arxiv.org/abs/2110.07602 | 深层 prompt | 支持多层 token prompt |
| PEFT | Ladder Side-Tuning | https://arxiv.org/abs/2206.06522 | 旁路网络 | 支持 frozen decoder + residual side branch |
| Latent 对齐 | Model stitching | https://arxiv.org/abs/1411.5908 | stitching layer 测表示等价 | 直接解释 code adapter |
| Latent 对齐 | Stitchable Neural Networks | https://arxiv.org/abs/2302.06586 | 分层 stitching | 支持多点 adapter |
| Latent 对齐 | Functional Latent Alignment | https://arxiv.org/abs/2505.20142 | 功能性 latent 对齐 | 支持 feature/output distillation |
| Latent 对齐 | Git Re-Basin | https://arxiv.org/abs/2209.04836 | permutation symmetry alignment | 支持 code 置换/旋转诊断 |
| 表示学习 | SimCLR | https://arxiv.org/abs/2002.05709 | 对比学习统一表示 | 支持 shared code contrastive loss |
| 集合条件 | Deep Sets | https://arxiv.org/abs/1703.06114 | permutation-invariant set encoder | 支持 support code -> domain embedding |
| 集合条件 | Set Transformer | https://arxiv.org/abs/1810.00825 | attention set encoder | 支持多模态 support set |
| 集合条件 | Perceiver | https://arxiv.org/abs/2103.03206 | cross-attention latent bottleneck | 支持长 code 矩阵条件压缩 |
| 权重生成 | HyperNetworks | https://arxiv.org/abs/1609.09106 | 生成网络权重 | 支持 LoRA/adapter generator |
| Domain adaptation | DANN | https://arxiv.org/abs/1412.4446 | domain-invariant representation | 支持 shared code training |
| Domain adaptation | Deep CORAL | https://arxiv.org/abs/1607.01719 | covariance alignment | 支持 code 分布对齐 |
| Domain adaptation | DAN / MMD | https://arxiv.org/abs/1502.02791 | MMD 分布对齐 | 替代过强 teacher code MSE |

## 9. 推荐下一步实验路线

### 9.1 先做诊断，不直接堆新模块

建议新增 `scripts/analyze_code_space_alignment.py`：

```text
输入：
  checkpoint_A encoder
  checkpoint_B encoder/decoder
  train/test data

输出：
  code mean/std
  covariance spectrum
  CKA/CCA similarity
  Procrustes alignment NMSE
  linear regression alignment NMSE
  MLP adapter alignment NMSE
```

判断标准：

| 结果 | 解释 | 下一步 |
|---|---|---|
| Procrustes 明显有效 | 主要是旋转/符号/置换问题 | 加 orthogonal/linear adapter、code normalization |
| full linear 有效但 orthogonal 不够 | 主要是仿射/缩放/低秩混合 | linear + residual adapter |
| MLP 有效但 linear 不行 | 非线性协议 | bottleneck/Invertible adapter |
| MLP 也卡在 `-21 dB` | decoder 内部解释规则不兼容 | decoder-side multi-layer adapter/LoRA |

### 9.2 做 decoder-side PEFT 上限

实验矩阵：

```text
baseline: frozen encoder + frozen decoder + no adapter
current:  code_adapter
control:  train fc_decoder full
new-1:    fc_decoder LoRA
new-2:    fc_decoder + Transformer FFN LoRA
new-3:    fc_decoder + FFN + attention LoRA
new-4:    hidden ReFT / IA3 gates
new-5:    side refiner
```

指标：

```text
Best NMSE
Final NMSE
trainable params
delta/base norm ratio
support/query gap
是否接近同 seed baseline -28 dB
```

### 9.3 重写 teacher code 目标

不建议继续只扫：

```text
MSE(adapter(code_A), code_B)
```

建议改为：

```text
reconstruction loss
+ decoder feature distillation
+ distribution alignment
+ weak code regularization
```

其中 code regularization 不应过强，优先从：

```text
lambda = 1e-4, 5e-4, 1e-3
```

开始，而不是 `0.1`。

### 9.4 做 static per-domain LoRA 上限，再做 LoRA generator

顺序：

```text
1. 每个目标 domain/encoder pair 训练一套 static decoder LoRA
2. 验证 static LoRA 是否显著超过 code_adapter
3. 如果 static LoRA 有收益，再训练 support-set -> LoRA generator
4. generator 先生成 gate/scale/prefix，再生成完整 A/B
```

如果 static LoRA 本身无收益，不建议直接上 hypernetwork/diffusion 生成 LoRA。

### 9.5 训练期 shared code space

长期路线：

```text
多个 encoder seeds / architectures
  -> shared projection head
  -> shared decoder
  -> reconstruction loss
  -> contrastive / CORAL / MMD / adversarial alignment
```

目标不是让所有 code 完全一样，而是让 decoder 读到的 shared latent 稳定。

## 10. 可写进论文或汇报的核心论点

1. CSI feedback autoencoder 的 codeword 不是天然标准码字，而是 encoder-decoder 联合训练形成的私有协议。
2. 跨 seed 重组从 `-28 dB` baseline 崩溃到正 NMSE，说明私有协议问题即使在同数据集、同结构下也存在。
3. 单点 code adapter 能把崩溃模型救到 `-20 ~ -21 dB`，说明源 code 中保留了足够信息，但 decoder 可读性没有完全恢复。
4. teacher code MSE 过强会伤害重建，说明“靠近另一个 encoder 的 raw code”不等价于“对目标 decoder 功能可读”。
5. 外部 PEFT 工作支持多层、小参数、decoder-side adaptation；外部 stitching 工作支持分层对齐和功能对齐；外部 CSI 泛化工作支持物理分布对齐与 shared decoder 适配。
6. 因此，下一步不应只扩大中间 MLP，而应做 decoder-side multi-layer PEFT、support-set conditioned adapter、functional distillation 和训练期 shared code space。

## 11. 最推荐的短期可落地组合

如果只选一个最现实的短期方案：

```text
decoder-side multi-layer LoRA / adapter
+ functional distillation
+ code distribution alignment
```

具体为：

```text
冻结 encoder_A
冻结 decoder_B base weights
在 decoder_B 的 fc/token/FFN 层插 LoRA 或 bottleneck adapter
训练 recon loss
弱加入 teacher decoder feature distill
弱加入 CORAL/MMD code distribution alignment
```

它比当前 code adapter 更强的原因：

- 直接改 decoder 如何解释 code；
- 不强迫 source code 逐点等于 teacher code；
- 多层适配覆盖 decoder 的真实重建路径；
- 参数量仍比 full fine-tuning 小，符合 BS 侧共享 decoder 的部署叙事。

如果这个组合仍不能接近 `-28 dB`，就可以有力地证明：

```text
后验轻量适配存在上限，必须在训练期学习 shared code space。
```

