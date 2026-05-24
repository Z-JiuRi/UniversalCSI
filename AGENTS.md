# 仓库贡献指南

## 项目结构与模块组织

本仓库是 TransNet 在 CSI 反馈任务上的 PyTorch 实现。核心入口是 `main.py`，
负责解析参数、构建数据加载器、初始化模型，并执行训练或评估。模型代码位于
`models/`，主要文件是 `models/TransNet.py`。数据加载逻辑位于
`dataloader/`。训练、评估、日志、调度器、参数解析和初始化工具位于 `utils/`。
根目录下的 `scripts.sh` 是示例启动脚本，`env.yaml` 定义 Conda 环境。

请不要将生成产物提交到版本控制。README 推荐将 COST2100 数据和实验输出放在
相邻目录中，例如 `COST2100/` 和 `Experiments/checkpoints/`。

## 数据流与维度说明

整体任务是对预处理后的 COST2100 CSI 稀疏角延迟域矩阵做自编码重建。代码中默认
单个样本的稀疏 CSI 维度是 `(2, 32, 32)`，其中 `2` 表示实部和虚部两个通道，
`32` 是天线维，另一个 `32` 是截取后的延迟/频域维。实际维度由 CLI 参数
`--channel`、`--nt` 和 `--nc` 控制，展平后的特征数为
`channel * nt * nc`，默认是 `2 * 32 * 32 = 2048`。

数据从 `--train_path`、`--val_path` 和 `--test_path` 指向的 `.pt` 文件进入
`dataloader/cost2100.py`：

- 训练集、验证集和测试集分别读取三个显式传入的 `.pt` 文件，例如
  `./COST2100/in_train.pt`、`./COST2100/in_val.pt` 和 `./COST2100/in_test.pt`。
  三者都会被转换为 `float32`，并按 `(N, channel, nt, nc)` 校验或 reshape。
- DataLoader 输出训练/验证 batch 时是单元素 tuple：`(sparse_gt,)`，其中
  `sparse_gt` 维度是 `(B, 2, 32, 32)`。测试 batch 也是单元素 tuple：
  `(sparse_gt,)`，维度为 `(B, 2, 32, 32)`。

训练和评估的数据流由 `utils/solver.py` 驱动：

- `Trainer._iteration()` 接收 `(sparse_gt,)`，将 `sparse_gt` 放到目标设备后调用
  `model(sparse_gt)` 得到 `sparse_pred`，维度保持 `(B, 2, 32, 32)`。
  损失函数是 `nn.MSELoss()`，直接比较 `sparse_pred` 与 `sparse_gt`。
- `Tester._iteration()` 接收 `(sparse_gt,)`。模型输入 `sparse_gt` 并输出
  `(B, 2, 32, 32)` 的重建结果；MSE loss 在稀疏域计算，
  `utils/statics.py::evaluator()` 计算 `NMSE`。

`models/TransNet.py` 中的 `transnet(reduction=args.cr, d_model=args.d_model)` 创建
Transformer 自编码器。`args.cr` 在代码中传给 `reduction`，表示压缩率分母；
例如默认维度下 `--cr 4` 对应压缩率 `1/4`，压缩后的码字长度是
`(channel * nt * nc) / 4 = 512`。

模型前向传播的主要维度变化如下：

- 输入 `src`: `(B, 2, 32, 32)`。
- `input_dim = channel * nt * nc`。
- `feature_shape = (input_dim // d_model, d_model)`。默认 `d_model=64`，所以
  `feature_shape=(32, 64)`。要求 `input_dim % d_model == 0`。
- `src.view(-1, input_dim // d_model, d_model)` 将输入展为 Transformer 序列：
  默认是 `(B, 32, 64)`，其中序列长度为 `32`，每个 token 的特征维为 `64`。
- Encoder 输入和输出维度都保持 `(B, 32, 64)`。每层自注意力使用 `nhead=2`，
  单头维度是 `d_model / nhead = 32`。
- Encoder 输出展平成 `(B, input_dim)`，经过
  `fc_encoder: Linear(input_dim, input_dim/cr)` 得到压缩码字 `(B, input_dim/cr)`。
  默认 `cr=4` 时是 `(B, 512)`。
- `fc_decoder: Linear(input_dim/cr, input_dim)` 将码字恢复为 `(B, input_dim)`，再
  reshape 回 `(B, input_dim // d_model, d_model)` 作为 Decoder 输入。
- Decoder 输出仍为 `(B, 32, 64)`，最终 `view(-1, 2, 32, 32)` 得到
  `sparse_pred: (B, 2, 32, 32)`。

`evaluator()` 的指标路径如下：

- 计算 `NMSE` 时，`sparse_gt` 和 `sparse_pred` 先减去 `0.5` 做去中心化，然后在
  稀疏域 `(B, 2, 32, 32)` 上按样本计算
  `10 * log10(sum(|error|^2) / sum(|gt|^2))` 后取平均。

## 构建、测试与开发命令

- `conda env create -f env.yaml`：创建项目 Conda 环境。
- `conda activate transnet`：运行脚本前激活环境。
- `python main.py --exp_name exp_1 --train_path ./COST2100/in_train.pt --val_path ./COST2100/in_val.pt --test_path ./COST2100/in_test.pt --epochs 400 --batch_size 200 --workers 0 --cr 4 --nt 32 --nc 32 --scheduler const --gpu 0`：从头开始训练，输出写入 `./exps/exp_1/`。
- `python main.py --exp_name eval_4_in --train_path ./COST2100/in_train.pt --val_path ./COST2100/in_val.pt --test_path ./COST2100/in_test.pt --pretrained ./checkpoints/4_in.pth --evaluate --batch_size 200 --workers 0 --cr 4 --nt 32 --nc 32 --cpu`：使用 checkpoint 进行评估，输出写入 `./exps/eval_4_in/`。
- `sh scripts.sh`：调整路径后运行示例训练脚本。

## 代码风格与命名规范

使用兼容 Python 3.8 的语法，并遵循现有 PyTorch 代码风格。缩进使用 4 个空格。
函数、变量和模块使用 `snake_case`；类名使用 `PascalCase`，例如 `Trainer`、
`Tester` 和模型类。CLI 参数统一维护在 `utils/parser.py`。当前仓库没有提交
formatter 或 linter 配置，因此应保持与周围代码一致，避免大范围无关格式化。

## 测试指南

当前仓库没有自动化测试套件。修改后请先进行小规模运行检查。改动模型、数据加载
或指标代码时，应使用已知 checkpoint 做评估。涉及训练流程的改动，可运行
`--epochs 1`，确认 loss 和 NMSE 日志正常输出。报告结果时请注明数据路径、
checkpoint、场景、压缩率、随机种子和硬件信息。

## Commit 与 Pull Request 规范

近期历史使用简短直接的提交信息，例如 `Update README.md`。每个 commit 应聚焦
单一改动，并说明受影响组件，例如 `Update scheduler warmup logic` 或
`Fix COST2100 loader shape handling`。

Pull Request 应包含改动摘要、验证命令，以及 NMSE 等结果变化。实验相关改动需
注明场景（`in` 或 `out`）、压缩率、epoch 数和使用的 checkpoint。除非明确要求，
不要提交数据集、日志或模型 checkpoint。

## 安全与配置建议

不要在可复用代码中硬编码本地绝对路径。路径应通过 `--train_path`、`--val_path`、
`--test_path`、`--pretrained` 和 `--resume` 等 CLI 参数传入。实验日志、
TensorBoard 文件和 checkpoint 会写入 `./exps/{exp_name}/`；下载的数据集和
checkpoint 应保留在版本控制之外。
评估时请确认压缩率与 checkpoint 匹配。
