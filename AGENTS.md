# 仓库贡献指南

## 项目结构与模块组织

本仓库是面向 CSI 反馈自编码重建任务的 PyTorch 实验框架。核心入口是 `main.py`，负责解析参数、创建实验目录、构建 DataLoader、初始化 `UniversalCSI` 模型，并执行端到端训练、评估、checkpoint 保存和 codeword 导出。

当前代码只保留统一自编码训练路径：

```text
CSI input -> encoder -> decoder -> reconstructed CSI
```

主要目录如下：

- `models/UniversalCSI.py`：总模型工厂，组合任意 encoder 和 decoder。
- `models/encoders/`：CSI 压缩编码器实现，包括 `csinet`、`cnn`、`cbam_cnn`、`crnet`、`clnet`、`transnet`、`resnet`、`dscnn`、`convnext`、`mlp_mixer`、`attention_cnn`、`swin`、`mlp_ae`、`sparse_resnet`。
- `models/decoders/`：CSI 重建解码器实现，包括 `transnet`、`cnn_residual`、`hybrid`。
- `dataloader/dataloader.py`：读取 `.pt` CSI 张量并构建 train/val/test DataLoader。
- `utils/`：参数解析、设备初始化、模型加载、训练/评估循环、日志、调度器和 NMSE 统计。
- `scripts/`：端到端训练和整模型评估脚本。
- `docs/`：补充说明文档。

不要提交数据集、checkpoint、TensorBoard 事件、实验日志、临时分析产物或本地绝对路径配置。实验输出默认写入 `./exps/{exp_name}/`，该目录通常只作为本地运行结果。

## 数据流与维度说明

整体任务是将预处理后的 CSI 稀疏角延迟域矩阵压缩成反馈码字，再重建为原始 CSI 张量。默认单样本维度为 `(2, 32, 32)`，其中 `2` 表示实部和虚部两个通道，`nt` 和 `nc` 分别表示天线维与延迟/频域维。实际维度由 `--channel`、`--nt`、`--nc` 控制：

```text
input_dim = channel * nt * nc
code_dim = input_dim // cr
```

默认 `channel=2, nt=32, nc=32, cr=4` 时，`input_dim=2048`，`code_dim=512`。`cr` 是压缩率分母，表示压缩率 `1/cr`。相关实现假定 `input_dim` 可被 `cr` 整除；Transformer/Mixer/Swin 等 token 化结构还会对 `d_model`、patch 或 window 尺寸有额外整除要求，应以对应模块实现为准。

数据从 `--train_path`、`--val_path`、`--test_path` 指向的 `.pt` 文件进入 `MyDataLoader`：

- `.pt` 内容会用 `torch.load(..., weights_only=True)` 读取并转换为 `float32`。
- 数据可以是 `(N, channel, nt, nc)`，也可以是二维展平张量，二维数据会 reshape 为 `(N, channel, nt, nc)`。
- DataLoader 仍会返回 `(sparse_gt, indices)`，但当前训练和评估只使用 `sparse_gt`。

训练和评估由 `utils/solver.py` 驱动：

- 训练：`model(sparse_gt)` 输出 `sparse_pred`，用 `nn.MSELoss()` 直接比较 `(B, channel, nt, nc)` 的重建结果与输入。
- 评估：`Tester` 计算 MSE loss，并由 `utils/statics.py::evaluator()` 累加全测试集的误差能量和信号能量，再输出全局 NMSE。

```text
NMSE = 10 * log10(sum(|error|^2) / sum(|gt|^2))
```

当前 `evaluator()` 不再做 `-0.5` 去中心化；如果改动指标逻辑，应同步更新 README 和实验结果说明。

## 模型组合与训练模式

`UniversalCSIModel` 的固定接口如下：

```text
CSI input: (B, channel, nt, nc)
  -> encoder
  -> decoder
  -> reconstructed CSI: (B, channel, nt, nc)
```

所有 encoder 输出 `(B, code_dim)`，所有 decoder 接收 `(B, code_dim)`。新增模型时应保持这个接口，不要把数据集、损失函数或脚本里的特殊逻辑写进模型内部。

支持的主要模式：

- 从头联合训练：使用 `--encoder` 和 `--decoder` 选择结构。
- 整模型评估：`--evaluate --pretrained checkpoint.pth`。
- resume 训练：`--resume checkpoint.pth`。

不再支持 partial encoder/decoder loading、frozen decoder training、CodeAdapter、teacher code loss、code-only training、LoRA、只训练 `decoder.fc_decoder`。

## 构建、运行与验证命令

- `conda env create -f env.yaml`：创建 Conda 环境，当前环境名为 `torch`。
- `conda activate torch`：激活环境。
- `bash scripts/train.sh`：用环境变量覆盖参数后启动基础训练。
- `bash scripts/test.sh`：加载整模型 checkpoint 后评估。

直接运行 `main.py` 的基础训练示例：

```bash
python main.py \
  --exp_name COST2100/in/seed42/transnet_hybrid \
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
  --scheduler cosine \
  --lr_init 2e-4 \
  --seed 42 \
  --gpu 0
```

当前仓库没有自动化测试套件。修改模型、数据加载、损失、checkpoint 加载或指标逻辑后，至少做一次小规模运行检查，例如 `--epochs 1 --workers 0 --batch_size 4` 或使用已知 checkpoint 运行 `--evaluate`。

## 代码风格与命名规范

使用与现有 PyTorch 代码一致的风格。缩进使用 4 个空格；函数、变量、模块名使用 `snake_case`；类名使用 `PascalCase`。CLI 参数统一维护在 `utils/parser.py`。

新增 encoder/decoder 时需要：

- 在对应目录新增实现文件。
- 在 `models/encoders/__init__.py` 或 `models/decoders/__init__.py` 导出类。
- 在 `build_encoder()` 或 `build_decoder()` 注册名称。
- 在 `utils/parser.py` 的 `choices` 中加入 CLI 名称。
- 保持 `(B, channel, nt, nc) -> (B, code_dim)` 或 `(B, code_dim) -> (B, channel, nt, nc)` 接口。

## Commit 与 Pull Request 规范

每个 commit 聚焦单一改动，提交信息简短直接，注意用中文，并说明受影响组件，例如 `Update README for base training` 或 `Fix NMSE calculation`。

Pull Request 应包含改动摘要、验证命令和关键结果。实验相关 PR 需要注明数据集/场景、encoder、decoder、压缩率、epoch 数、seed、checkpoint 和 NMSE 变化。不要把数据集、checkpoint 或大体积分析产物加入提交，日志和分析报告等产物应当提交。

## 安全与配置建议

不要在可复用 Python 代码中硬编码本地绝对路径。路径应通过 `--train_path`、`--val_path`、`--test_path`、`--pretrained`、`--resume` 等 CLI 参数传入。脚本中的本地默认路径仅作为当前机器的实验便利配置，移植到其他环境前应改成环境变量覆盖。

评估和训练时，需要确认 `channel/nt/nc/cr`、encoder、decoder 与 checkpoint 完全匹配，否则参数形状或码字维度会不一致。
