# UniversalCSI

UniversalCSI 是一个用于 CSI 反馈压缩与重建的 PyTorch 实验框架。项目从 TransNet
自编码重建流程扩展而来，把模型拆成可组合的 encoder、可选 code adapter 和
decoder，并补充了冻结模块训练、teacher code 对齐、LoRA 微调、只训练
`fc_decoder`、codeword 导出与分析脚本。

核心数据流如下：

```text
CSI input
  -> selectable encoder
  -> optional code adapter
  -> selectable decoder
  -> reconstructed CSI
```

默认任务使用预处理后的稀疏角延迟域 CSI 张量 `(2, 32, 32)`。其中 `2` 是实部和
虚部通道，`32 x 32` 是天线维和延迟/频域维。通过 `--channel`、`--nt`、`--nc`
可以切换输入形状。

## 主要功能

- 支持 14 种 encoder：`csinet`、`cnn`、`cbam_cnn`、`crnet`、`clnet`、
  `transnet`、`resnet`、`dscnn`、`convnext`、`mlp_mixer`、`attention_cnn`、
  `swin`、`mlp_ae`、`sparse_resnet`。
- 支持 3 种 decoder：`transnet`、`cnn_residual`、`hybrid`。
- 支持任意 encoder 与 decoder 组合，只要满足统一码字接口。
- 支持完整模型训练、checkpoint 评估、resume 训练。
- 支持只加载并冻结 encoder 或 decoder，用于跨 seed、跨结构迁移实验。
- 支持残差 MLP `CodeAdapter`，用于对齐冻结 encoder 和冻结 decoder 之间的码字空间。
- 支持 teacher code loss，可使用固定权重、可学习权重或 code-only 训练。
- 支持 `HybridDecoder.token_projection` 的 LoRA 微调，并记录 LoRA 权重指标。
- 支持仅训练 `TransNetDecoder.fc_decoder`。
- 支持训练或评估结束后导出 train split 的 encoder codewords。
- 支持 codeword 分布、跨模型对齐、LoRA 条件分析和综合报告脚本。

## 项目结构

```text
main.py                         训练/评估入口
dataloader/dataloader.py        .pt 数据加载与 DataLoader 构建
models/UniversalCSI.py          通用模型工厂和 encoder/decoder 组合
models/encoders/                各类 CSI encoder
models/decoders/                各类 CSI decoder
models/lora.py                  LoRA Linear、参数冻结和指标收集
utils/parser.py                 CLI 参数定义
utils/init.py                   设备初始化、模型初始化、checkpoint 加载
utils/solver.py                 训练、验证、测试、codeword 导出
utils/statics.py                MSE/NMSE 统计
scripts/train.sh                基础训练脚本
scripts/test.sh                 评估脚本
scripts/train_frozen_decoder.sh 冻结 decoder 训练 encoder
scripts/train_adapter.sh        冻结 encoder/decoder 训练 code adapter
scripts/train_lora.sh           LoRA 微调
scripts/train_fc_decoder.sh     只训练 TransNet fc_decoder
scripts/run_all_analysis.sh     codeword 分析流水线
```

实验输出默认写入：

```text
exps/{exp_name}/
├── args.json
├── run.log
├── checkpoints/
│   └── best_nmse.pth
├── tensorboard/
└── codewords/
    └── train_code.pt
```

## 环境安装

项目环境由 `env.yaml` 定义，当前环境名是 `torch`：

```bash
conda env create -f env.yaml
conda activate torch
```

主要依赖包括 Python 3.11、PyTorch 2.5.1、CUDA 12.4、TensorBoard、Matplotlib 和
`thop`。

## 数据格式

`main.py` 需要显式传入训练、验证和测试数据：

```text
--train_path path/to/train.pt
--val_path   path/to/val.pt
--test_path  path/to/test.pt
```

每个 `.pt` 文件应保存一个 PyTorch Tensor。推荐形状为：

```text
(N, channel, nt, nc)
```

默认是：

```text
(N, 2, 32, 32)
```

如果数据是二维展平张量，`MyDataLoader` 会按当前 `--channel --nt --nc` reshape：

```text
(N, channel * nt * nc) -> (N, channel, nt, nc)
```

加载后数据会转换为 `float32`。当前 DataLoader 会同时返回样本索引：

```text
(sparse_gt, indices)
```

普通训练只使用 `sparse_gt`；teacher code 对齐训练会用 `indices` 从
`--teacher_code` 中取对应样本的目标码字。

## 维度约定

模型统一使用以下维度：

```text
input_dim = channel * nt * nc
code_dim = input_dim // cr
```

`--cr` 是压缩率分母。默认 `channel=2, nt=32, nc=32, cr=4` 时：

```text
input_dim = 2048
code_dim = 512
compression ratio = 1/4
```

所有 encoder 的接口：

```text
input: (B, channel, nt, nc)
code:  (B, code_dim)
```

所有 decoder 的接口：

```text
code:   (B, code_dim)
output: (B, channel, nt, nc)
```

注意事项：

- `input_dim` 应能被 `cr` 整除。
- `transnet` encoder/decoder 和 `hybrid` decoder 需要 `input_dim` 能按 `d_model`
  切成 token 序列。
- `mlp_mixer`、`swin` 等 patch/window 结构还要求 `nt`、`nc` 与内部 patch/window
  配置兼容。

## 模型组件

### Encoder

可通过 `--encoder` 选择：

```text
csinet         浅层 CsiNet 风格卷积 + 全连接瓶颈
cnn            标准 CNN 下采样编码器
cbam_cnn       带 CBAM 通道/空间注意力的 CNN
crnet          多分支卷积 CRNet 风格编码器
clnet          带空间注意力和通道注意力的轻量编码器
transnet       TransformerEncoder 编码器
resnet         残差 CNN 编码器
dscnn          深度可分离卷积编码器
convnext       ConvNeXt 风格编码器
mlp_mixer      Patch token + MLP-Mixer 编码器
attention_cnn  SE + SpatialGate 注意力 CNN
swin           Swin 风格窗口注意力编码器
mlp_ae         纯 MLP 自编码器编码端
sparse_resnet  稀疏变换 + ResNet 编码器
```

### Decoder

可通过 `--decoder` 选择：

```text
transnet      Linear 扩展 + TransformerDecoder
cnn_residual  Linear 粗重建 + CNN 残差精修
hybrid        Transformer token 重建 + CNN 残差精修
```

`--hidden` 和 `--num_blocks` 控制 CNN 精修头的通道数和残差块数量，主要影响
`cnn_residual` 与 `hybrid`。

### CodeAdapter

`--code_adapter` 会在 encoder 和 decoder 之间插入残差 MLP：

```text
code -> LayerNorm -> Linear -> GELU -> Linear -> scale -> residual add
```

最后一层初始为 0，因此初始行为接近 identity。典型用法是加载一个冻结 encoder 和
一个冻结 decoder，只训练 adapter 来对齐两者码字空间。

### LoRA

当前 LoRA 只支持：

```text
--decoder hybrid --lora_component token_projection
```

启用后会把 `HybridDecoder.token_projection` 包装为 `LoRALinear`，冻结基础模型参数，
只训练 LoRA 的 A/B 矩阵。可调参数：

```text
--lora_rank 8
--lora_alpha 16
```

训练日志会记录 LoRA 的权重范数、delta/base ratio、最大绝对值等指标。

## 基础训练

直接运行 `main.py`：

```bash
python main.py \
  --exp_name COST2100/in/seed42/transnet_hybrid/base \
  --train_path ./COST2100/in_train.pt \
  --val_path ./COST2100/in_val.pt \
  --test_path ./COST2100/in_test.pt \
  --epochs 400 \
  --batch_size 200 \
  --workers 0 \
  --cr 4 \
  --encoder transnet \
  --decoder hybrid \
  --nt 32 \
  --nc 32 \
  --d_model 64 \
  --dim_feedforward 2048 \
  --scheduler cosine \
  --lr_init 2e-4 \
  --weight_decay 1e-3 \
  --seed 42 \
  --gpu 0
```

也可以用脚本，通过环境变量覆盖默认值：

```bash
seed=42 gpu=0 encoder=transnet decoder=hybrid batch_size=200 bash scripts/train.sh
```

`scripts/run.sh` 会批量启动多个 encoder/decoder 组合实验。该脚本包含当前机器的
本地数据路径和 GPU 编号，换环境前需要修改或用环境变量覆盖。

## 评估与 codeword 导出

评估完整 checkpoint：

```bash
python main.py \
  --exp_name COST2100/in/test/seed42/transnet_hybrid \
  --train_path ./COST2100/in_train.pt \
  --val_path ./COST2100/in_val.pt \
  --test_path ./COST2100/in_test.pt \
  --epochs 1 \
  --batch_size 200 \
  --workers 0 \
  --cr 4 \
  --encoder transnet \
  --decoder hybrid \
  --pretrained exps/COST2100/in/seed42/transnet_hybrid/base/checkpoints/best_nmse.pth \
  --evaluate \
  --gpu 0
```

脚本方式：

```bash
seed=42 gpu=0 encoder=transnet decoder=hybrid bash scripts/test.sh
```

评估模式除了输出 test loss 和 NMSE，还会把 train split 的 encoder 输出保存到：

```text
exps/{exp_name}/codewords/train_code.pt
```

训练模式结束后同样会导出 train codewords。这个文件可作为 adapter、fc decoder 或
codeword 分析脚本的输入。

## Checkpoint 加载与冻结训练

### 完整加载

```text
--pretrained path/to/best_nmse.pth
```

用于完整模型评估或在完整模型权重上继续训练。

### 只加载 encoder

```text
--pretrained_encoder path/to/checkpoint.pth
```

只复制 checkpoint 中的 `encoder.*` 参数，并冻结 encoder。

### 只加载 decoder

```text
--pretrained_decoder path/to/checkpoint.pth
```

只复制 checkpoint 中的 `decoder.*` 参数，并冻结 decoder。

如果传入 `--pretrained_encoder` 或 `--pretrained_decoder`，`--pretrained` 会被忽略。
部分加载时，当前模型的 encoder/decoder 结构和 checkpoint 中对应部分必须一致。

冻结预训练 decoder、训练新 encoder 的脚本：

```bash
encoder=transnet decoder=hybrid seed=2026 decoder_seed=42 gpu=0 \
  bash scripts/train_frozen_decoder.sh
```

## Adapter 训练

Adapter 训练用于连接一个冻结 encoder 和一个冻结 decoder：

```bash
encoder=transnet decoder=hybrid \
encoder_seed=3407 decoder_seed=42 seed=3407 gpu=0 \
bash scripts/train_adapter.sh
```

脚本会传入：

```text
--code_adapter
--pretrained_encoder ...
--pretrained_decoder ...
```

如果提供 `--teacher_code`，还会加入 encoder code 对齐损失。teacher code 文件应为
训练集顺序对齐的二维张量：

```text
(num_train_samples, code_dim)
```

常见设置：

```bash
# 重建损失 + 固定权重 code loss
code_loss_lambda=0.1 bash scripts/train_adapter.sh

# 只优化 code loss
code_loss_only=true bash scripts/train_adapter.sh
```

`--code_loss_only` 和 `--code_loss_lambda` 都要求存在 `--teacher_code`。
如果希望使用主程序的可学习 lambda，需要直接运行 `main.py` 并传入
`--teacher_code`，同时不要传 `--code_loss_lambda`；`scripts/train_adapter.sh`
当前默认会补上 `code_loss_lambda=0.1`。

## 只训练 TransNet fc_decoder

该模式用于冻结 encoder 和 decoder 其他部分，只训练 `TransNetDecoder.fc_decoder`：

```bash
encoder=transnet decoder=transnet \
encoder_seed=3407 decoder_seed=42 seed=3407 gpu=0 \
bash scripts/train_fc_decoder.sh
```

限制：

- 必须使用 `--decoder transnet`。
- 不能同时启用 `--code_adapter`。
- 可选配 `--teacher_code`、`--code_loss_lambda`、`--code_loss_only`。
- 如果启用 teacher code，代码会比较当前 `fc_decoder(source_code)` 与冻结 teacher
  `fc_decoder(teacher_code)` 的 token 输出。

## LoRA 训练

LoRA 当前只支持 hybrid decoder 的 `token_projection`：

```bash
lora_component=token_projection \
encoder=transnet decoder=hybrid \
lora_rank=64 lora_alpha=128 \
pretrained=exps/COST2100/in/seed42/transnet_hybrid/base/checkpoints/best_nmse.pth \
gpu=0 seed=2026 \
bash scripts/train_lora.sh
```

也可以分别加载 encoder 和 decoder：

```bash
pretrained_encoder=path/to/encoder_checkpoint.pth \
pretrained_decoder=path/to/decoder_checkpoint.pth \
bash scripts/train_lora.sh
```

LoRA 模式会在训练开始前先做一次测试，随后每个 epoch 记录 test loss、NMSE 和 LoRA
指标。

## Resume、调度器与日志

使用 `--resume path/to/checkpoint.pth` 可从保存的训练状态恢复，包括模型、优化器、
调度器、best NMSE 和可学习 code loss lambda。

学习率调度器：

```text
--scheduler const   固定学习率
--scheduler cosine  warmup + cosine annealing
```

优化器使用 AdamW，并将 bias、LayerNorm/BatchNorm 等一维参数放入 no weight decay
组。日志写入 `exps/{exp_name}/run.log`，TensorBoard 写入
`exps/{exp_name}/tensorboard/`。

## 指标

训练损失是重建 MSE，或在迁移训练中叠加 code loss：

```text
loss = recon_mse + lambda * code_mse
```

`--code_loss_only` 时：

```text
loss = code_mse
```

测试 NMSE 在全测试集上累加误差能量和信号能量后计算：

```text
NMSE = 10 * log10(sum(|sparse_gt - sparse_pred|^2) / sum(|sparse_gt|^2))
```

当前实现不对 `sparse_gt` 和 `sparse_pred` 做 `-0.5` 去中心化。

## Codeword 分析

`scripts/run_all_analysis.sh` 按顺序运行以下分析：

```text
analyze_codewords.py             基础 per-split codeword 摘要
deep_analyze_train_codewords.py  train codeword 深度统计
summarize_codeword_analysis.py   汇总分析报告
comprehensive_lora_analysis.py   LoRA 条件分析
enhanced_lora_analysis.py        按 decoder 增强分析
consolidated_analysis.py         综合报告和图表
```

默认输出到：

```text
exps/seed42/COST2100/in/codeword_analysis/
```

这些脚本通常依赖已有实验目录、`training_results.csv` 和 `codewords/train_code.pt`。
运行前需要确认路径与当前实验组织一致。

## 常用脚本速查

```bash
# 基础训练
seed=42 gpu=0 encoder=transnet decoder=hybrid bash scripts/train.sh

# 评估 checkpoint
seed=42 gpu=0 encoder=transnet decoder=hybrid bash scripts/test.sh

# 冻结 decoder 训练 encoder
encoder=transnet decoder=hybrid decoder_seed=42 seed=2026 gpu=0 \
  bash scripts/train_frozen_decoder.sh

# 训练 code adapter
encoder=transnet decoder=hybrid encoder_seed=3407 decoder_seed=42 seed=3407 gpu=0 \
  bash scripts/train_adapter.sh

# 只训练 TransNet fc_decoder
encoder=transnet decoder=transnet encoder_seed=3407 decoder_seed=42 seed=3407 gpu=0 \
  bash scripts/train_fc_decoder.sh

# Hybrid decoder LoRA
encoder=transnet decoder=hybrid lora_rank=64 lora_alpha=128 gpu=0 \
  bash scripts/train_lora.sh

# codeword 分析流水线
bash scripts/run_all_analysis.sh
```

## 开发注意事项

- 新增 CLI 参数应在 `utils/parser.py` 中维护，并同步更新 README 与相关脚本。
- 新增 encoder 需要在 `models/encoders/__init__.py` 和
  `models/UniversalCSI.py::build_encoder()` 注册，同时加入 `--encoder choices`。
- 新增 decoder 需要在 `models/decoders/__init__.py` 和
  `models/UniversalCSI.py::build_decoder()` 注册，同时加入 `--decoder choices`。
- 模型必须保持统一接口，避免在模块内部直接依赖具体数据路径或实验目录。
- 不要提交 `exps/`、数据集、checkpoint、TensorBoard 文件和本地临时分析结果。
- 脚本中存在当前机器的绝对路径，移植时优先通过环境变量覆盖。

## 结果报告建议

报告实验结果时至少包含：

```text
数据集/场景：COST2100 in/out 或 WAIRD 等
数据路径：train/val/test .pt
模型：encoder + decoder + 是否 code_adapter/LoRA/fc_decoder
压缩率：cr
输入维度：channel, nt, nc
训练设置：epochs, batch_size, lr, scheduler, weight_decay
随机种子：seed
checkpoint：pretrained/pretrained_encoder/pretrained_decoder/resume
指标：test loss, NMSE
硬件：GPU/CPU 信息
```
