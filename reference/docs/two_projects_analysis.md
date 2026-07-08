# JiT 与 Recurrent-Parameter-Generation 项目分析

本文基于当前仓库中的两个独立项目目录进行代码层分析：

- `JiT/`：Just image Transformer，用于像素空间图像扩散生成。
- `Recurrent-Parameter-Generation/`：RPG，Recurrent Diffusion for Large-Scale Parameter Generation，用于生成神经网络参数。

这两个项目都使用 PyTorch 和扩散式训练目标，但它们处理的对象完全不同：JiT 生成图像像素，RPG 生成模型权重。对 CSI 反馈自编码任务而言，JiT 更接近“高维张量去噪重建”的生成建模范式，RPG 更接近“模型参数或适配器参数生成”的元学习/参数生成范式。

## 一、项目定位对比

| 维度 | JiT | Recurrent-Parameter-Generation |
| --- | --- | --- |
| 核心目标 | 在像素空间训练类别条件图像扩散模型 | 从 checkpoint 数据集中学习生成完整神经网络参数 |
| 生成对象 | RGB 图像张量 `(B, 3, H, W)` | 模型参数 token 序列 `(B, L, dim_per_token)` |
| 主要模型 | Transformer denoiser | Mamba/Transformer/LSTM/GMLP + 一维扩散 denoiser |
| 条件信息 | ImageNet 类别标签和时间步 | permutation state、任务条件、类别条件等 |
| 训练数据 | ImageFolder 格式 ImageNet | 一组已训练 checkpoint |
| 评估方式 | 采样图像，计算 FID 和 Inception Score | 生成 checkpoint，再加载到下游模型上测试精度/IoU 等 |
| 工程入口 | `main_jit.py` | `workspace/*.py` 配置脚本和 `workspace/evaluate/*.py` |

## 二、JiT 项目分析

### 1. 代码结构

JiT 的主路径较集中：

- `main_jit.py`：参数解析、分布式初始化、数据加载、模型构建、优化器、checkpoint resume、训练和评估调度。
- `denoiser.py`：扩散训练目标和采样过程封装。
- `model_jit.py`：JiT Transformer 主体，包括 patch embedding、时间/类别嵌入、RoPE、adaLN 调制和输出层。
- `engine_jit.py`：单 epoch 训练、图像采样、FID/IS 评估。
- `util/`：分布式、学习率、位置编码、RMSNorm 等工具函数。

整体设计比较自包含：`main_jit.py` 只依赖 ImageNet 数据目录和 CLI 参数即可启动训练或生成评估。

### 2. 数据流

训练数据通过 `torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'))` 读取，图像先经过中心裁剪、随机水平翻转和 `PILToTensor()`，进入训练循环后转为 `float32`，并归一化到 `[-1, 1]`。

核心训练路径如下：

```text
ImageNet image x, label y
  -> normalize to [-1, 1]
  -> sample t ~ sigmoid(N(P_mean, P_std))
  -> sample noise e
  -> noisy state z = t * x + (1 - t) * e
  -> JiT(z, t, dropped_label)
  -> predict x_pred
  -> convert to velocity v_pred
  -> MSE(v_pred, v)
```

这里的扩散形式不是传统 DDPM 离散时间噪声预测，而是更接近连续时间插值/ODE 的速度预测。`Denoiser.forward()` 直接构造 `z` 和目标速度 `v = (x - z) / (1 - t)`，模型输出 `x_pred` 后再换算成 `v_pred` 计算 L2 loss。

### 3. 模型结构

`model_jit.py` 中的 `JiT` 是图像到图像的 Transformer denoiser。主要组件包括：

- `BottleneckPatchEmbed`：先用 patch-size stride 卷积把图像切成 patch，再用 `1x1` 卷积映射到 Transformer hidden size。
- `TimestepEmbedder`：将连续时间步 `t` 映射为 hidden embedding。
- `LabelEmbedder`：类别条件 embedding，并预留 `num_classes` 作为 classifier-free guidance 的 unconditional label。
- `JiTBlock`：RMSNorm + attention + SwiGLU FFN，并通过时间/类别条件生成 adaLN 的 shift、scale、gate。
- `VisionRotaryEmbeddingFast`：对 attention 的 query/key 加二维 RoPE。
- `FinalLayer`：把 token 输出映射回 patch 像素，再 `unpatchify` 成图像。

模型族通过工厂字典注册：

```text
JiT-B/16, JiT-B/32
JiT-L/16, JiT-L/32
JiT-H/16, JiT-H/32
```

其中 B/L/H 主要改变 depth、hidden size 和 head 数，`/16` 或 `/32` 改变 patch size。

### 4. 采样与评估

`Denoiser.generate()` 从高斯噪声开始，沿 `t=0 -> 1` 用 Euler 或 Heun ODE step 更新图像。每一步会分别执行 conditional 和 unconditional forward，再按 CFG 系数融合：

```text
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

`engine_jit.evaluate()` 会临时切换到 EMA 参数，按类别均匀生成指定数量图像，保存为 PNG，再用 `torch_fidelity` 计算 FID 和 Inception Score。当前代码内置了 256 和 512 分辨率的 ImageNet FID statistics 文件。

### 5. 工程特点与风险

JiT 的优点是路径清晰，训练和生成逻辑集中，模型接口简单。它适合研究像素空间高分辨率扩散生成，也适合迁移到其他 `(C, H, W)` 张量重建问题。

主要风险和注意点：

- 训练成本很高，README 示例面向 8 张 H200 GPU。
- 代码大量默认 CUDA，包括自定义 attention 中直接 `.cuda()`，CPU 或非 CUDA 后端适配成本较高。
- `torch.compile` 用在 block 和 final layer 上，可能受 PyTorch 版本、显卡架构和动态 shape 影响。
- 数据加载默认 ImageFolder/ImageNet 结构，迁移到 CSI 数据需要重写 dataset 和归一化方式。
- 评估强绑定图像生成指标，不能直接用于 CSI 的 NMSE。

## 三、Recurrent-Parameter-Generation 项目分析

### 1. 代码结构

RPG 的工程组织比 JiT 更分散，核心模块包括：

- `model/__init__.py`：组合 recurrent sequence model 和 diffusion loss，定义 `MambaDiffusion`、`TransformerDiffusion` 等模型。
- `model/mamba.py`、`model/transformer.py`、`model/lstm.py`、`model/gatemlp.py`：不同序列建模 backbone。
- `model/diffusion.py`：DDPM/DDIM 训练器与采样器，以及条件一维扩散损失。
- `model/denoiser.py`：一维 CNN/UNet denoiser。
- `dataset/dataset.py`：checkpoint 到 token 序列的预处理和反变换。
- `dataset/register.py`：不同下游 checkpoint 数据集的注册。
- `workspace/**/*.py`：具体实验配置、训练循环、生成脚本和评估脚本。

RPG 的使用方式不是一个统一 CLI，而是运行某个 workspace 配置脚本。每个配置脚本定义 dataset、模型、优化器、训练循环、tag、checkpoint 路径和生成测试命令。

### 2. 数据流

RPG 不直接学习图像或样本，而是学习一组已经训练好的模型 checkpoint。`BaseDataset` 会读取 checkpoint 目录中的 `.pth` 文件，并把参数字典转换为 token 序列：

```text
checkpoint state_dict
  -> 跳过非浮点参数、标量、num_batches_tracked
  -> 对每层参数计算结构、均值、方差
  -> running_var 做 log 变换
  -> 标准化参数
  -> 按 dim_per_token 切分/填充成 token
  -> 得到 (sequence_length, dim_per_token)
```

反向生成时，`save_params()` 会把生成的 token 通过 `postprocess()` 还原成原始 state_dict 结构，再保存为可加载 checkpoint。

token 化策略由 `granularity` 控制：

- `0`：直接 flatten 后整体切 token。
- `1`：按层切分。
- `2`：按输出维度更细粒度切分。

位置编码由 `pe_granularity` 控制，可不用位置编码、使用 1D 位置编码，或使用按层和层内 token 位置构造的 2D 位置编码。

### 3. 模型结构

RPG 的主模型可以拆成两级：

```text
condition / permutation_state / positional_embedding
  -> recurrent sequence model
  -> 每个参数 token 的条件向量 c
  -> conditional 1D diffusion
  -> 生成参数 token
  -> 反变换为 checkpoint
```

`ModelDiffusion` 是核心组合类：

- `to_condition`：把外部条件映射到 `d_model`。
- `to_permutation_state`：为不同 checkpoint index 学一个 embedding，训练时输入 dataset 返回的 index。
- `self.model`：Mamba、Transformer、LSTM 或 GMLP，用位置编码和条件生成整段 token 条件。
- `DiffusionLoss`：对真实参数 token 执行扩散训练。

以 `MambaDiffusion` 为例，`MambaModel` 会把固定或可训练位置编码 `pe` 与 condition 相加，然后送入若干层 `Mamba2`，输出形状约为：

```text
(batch, sequence_length, d_model)
```

之后 `DiffusionLoss` 使用 `ConditionalUNet` 作为一维 denoiser，对每个参数 token 学习噪声预测。`diffusion_batch` 用于限制一次扩散训练或采样的 token 数，避免显存随参数规模爆炸。

### 4. 训练、生成与评估

以 `workspace/example/cifar10_resnet18.py` 为例，训练流程如下：

```text
Cifar10_ResNet18 checkpoint dataset
  -> BaseDataset(dim_per_token=8192)
  -> MambaDiffusion(sequence_length, positional_embedding)
  -> AdamW + CosineAnnealingLR
  -> loss = model(output_shape=param.shape, x_0=param, permutation_state=index)
  -> 定期保存 RPG 模型 checkpoint
  -> 定期生成下游模型 checkpoint 并调用 test_command
```

生成脚本 `workspace/evaluate/generate.py` 会加载某个 workspace 配置和 RPG checkpoint，调用 `model(sample=True)` 生成参数，再由 dataset 的 `save_params()` 保存为下游模型 checkpoint，可选择立即执行测试命令。

评估脚本 `workspace/evaluate/evaluate.py` 更偏分析用途：它会加载原始 checkpoint、生成 checkpoint 和加噪 checkpoint，运行下游任务测试，比较准确率和错误样本集合 IoU，用于判断生成模型是否只是复制已有 checkpoint，还是产生了有性能且有差异的参数。

### 5. 工程特点与风险

RPG 的优点是任务抽象很有扩展性：只要能把目标模型参数保存为 state_dict，并提供 test.py，就可以把该任务接入参数生成流程。它对大规模参数的处理方式也比较明确：按 token 切分参数，用 recurrent backbone 建模 token 间关系，再用局部一维扩散生成每个 token。

主要风险和注意点：

- 工程入口分散，很多实验配置是 Python 脚本而不是统一 CLI，复现实验前需要先读对应 workspace 文件。
- 依赖 `mamba-ssm`，安装和 CUDA 编译环境要求较高。
- 数据集不是原始样本，而是大量已训练 checkpoint；准备数据成本高。
- `BaseDataset` 会缓存结构信息，如果 checkpoint 集合变化，需要注意 cache 是否过期。
- 生成参数的可用性依赖后处理、归一化统计、BN running_var 变换和下游模型加载是否完全匹配。
- `test_command` 通过字符串调用外部命令，跨环境迁移时要检查路径、conda 环境和 GPU 配置。

## 四、两个项目的共同点

虽然生成对象不同，但两者有几个共同思想：

1. 都把生成过程表述为从噪声到目标的去噪/反扩散过程。
2. 都用条件信息控制生成结果：JiT 使用类别标签，RPG 使用任务条件、checkpoint/permutation embedding 或类别条件。
3. 都强调采样阶段和训练阶段的接口分离：训练计算 loss，采样从噪声递推生成目标。
4. 都用较重的深度模型作为 denoiser 或条件生成器，需要较强 GPU 环境。
5. 都不是即插即用到 CSI 任务：JiT 的输入和评估是图像，RPG 的输入和输出是 checkpoint。

## 五、对 CSI 反馈自编码任务的参考价值

### 1. JiT 可借鉴的方向

CSI 稀疏角延迟域矩阵本质上也是多通道二维张量，默认形状类似 `(2, 32, 32)`。因此 JiT 中以下设计有迁移参考价值：

- 用 Transformer 直接处理二维 patch token。
- 用时间步 embedding 和条件调制控制重建/生成过程。
- 用 velocity 或 denoising objective 学习从噪声到 CSI 的映射。
- 用 EMA 参数做采样评估。

如果要迁移到 CSI，需要改动的部分包括：

- 将输入通道从 3 改为 CSI 的 `channel`，默认 2。
- 将图像归一化和 ImageFolder 数据加载替换为 `.pt` CSI DataLoader。
- 将输出评价从 FID/IS 改为 MSE/NMSE。
- 小尺寸 CSI 下 patch size、token 数和 hidden size 要重新设计，不能直接套 ImageNet 配置。
- 如果目标仍是压缩反馈，需要加入 encoder/codeword 约束，而不只是无条件重建 CSI。

### 2. RPG 可借鉴的方向

RPG 对 CSI 任务的直接迁移价值不在“重建 CSI 样本”，而在“生成或适配模型参数”。可能方向包括：

- 为不同场景、不同 seed、不同压缩率的 CSI autoencoder 生成 decoder 或整模型参数。
- 学习从场景条件到模型参数的映射，例如 indoor/outdoor、SNR、用户分布或天线配置。
- 生成轻量 adapter 参数，用于快速适配新信道分布。
- 分析多个训练好的 CSI checkpoint 之间的参数空间结构。

但这要求先构建 checkpoint 数据集，而不是 CSI 样本数据集。对于当前 UniversalCSI 的常规训练路径，RPG 不是直接替代 encoder-decoder 的方法，更像是上层的“模型生成器”。

## 六、迁移优先级建议

如果目标是改进当前 CSI 自编码重建模型，优先参考 JiT，而不是 RPG。原因是 JiT 的建模对象仍是样本张量，和 CSI 输入输出形态更接近。可以先做一个小规模原型：

```text
CSI x
  -> 加噪/采样 t
  -> Transformer denoiser
  -> 重建 x 或预测 velocity
  -> NMSE 评估
```

如果目标是跨数据集、跨场景或跨压缩率快速生成模型参数，再考虑 RPG。它需要先准备大量 UniversalCSI checkpoint，并定义统一的参数 token 化、结构缓存和测试命令。

工程上建议分两步走：

1. 先把 JiT 的 denoising/Transformer block 思路改造成 CSI 小模型，验证 NMSE 是否有收益。
2. 当已有足够多 CSI checkpoint 后，再评估 RPG 是否能用于 decoder 初始化、adapter 生成或场景条件参数生成。

## 七、结论

JiT 是一个结构清晰的像素空间扩散生成项目，适合作为“二维张量去噪建模”的参考。它和 CSI 数据形态更接近，但需要重写数据、评价指标和压缩接口。

RPG 是一个面向模型权重生成的参数扩散项目，工程复杂度更高，依赖 checkpoint 数据集和下游测试闭环。它不适合作为 CSI 样本重建模型的直接替代，但对“为不同 CSI 场景生成模型参数”很有参考价值。

对于当前仓库的 CSI 反馈自编码目标，更现实的路线是先借鉴 JiT 的 denoising Transformer 结构做样本级重建实验；RPG 则作为后续模型参数生成或快速场景适配方向储备。
