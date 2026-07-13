# Affine Residual Adapter 码字对齐实验

本目录用于训练两个 UniversalCSI 实验之间的码字映射器。默认目标是把 source encoder 产生的 codeword 映射到 target encoder 的 codeword 空间，然后接 frozen target decoder，在真实 CSI 上评估重建 NMSE。

默认数据流：

```text
source_code
  -> 闭式 affine 对齐：z0 = source @ W + b
  -> 恒等初始化 residual MLP：z = z0 + f(z0)
  -> frozen target decoder
  -> reconstructed CSI
```

`W,b` 只用 train split 的 source/target codeword 拟合，val/test 只用于评估。默认 `W,b` 作为 buffer 冻结，只训练 residual MLP。`use_final_norm` 默认关闭，因为实验中 final LayerNorm 会破坏 affine 学到的尺度和偏置。

## 文件说明

- `models.py`：mapper 结构定义，包括 `affine_residual_mlp`、`affine_linear`、`direct_mlp`。
- `train_adapter.py`：训练、评估、checkpoint、mapped code 导出入口。
- `scripts/train_adapter.sh`：参数化单次训练脚本。
- `scripts/run_adapter.sh`：默认示例运行脚本。

## 输出内容

默认输出到：

```text
adapter/exps/<mapper_type>/seed<source_seed>/<source_encoder>/code<lambda_code>_rec<lambda_recon>_lr<lr>_ep<epochs>
```

例如 `source_seed=1014, source_encoder=transnet, mapper_type=affine_residual_mlp` 时默认目录是：

```text
adapter/exps/affine_residual_mlp/seed1014/transnet/code1.0_rec0.0_lr5e-4_ep400
```

每次运行会写入：

- `run.log`：训练日志。
- `tensorboard/`：TensorBoard event 文件，记录配置、数据规模、参数量、affine 统计、train/val/test 指标、best 指标。
- `args.json`：本次运行参数。
- `affine_alignment.pt`：离线拟合得到的 `W,b`。
- `history.json`：逐 epoch 指标。
- `metrics.json`：最佳指标汇总。
- `checkpoints/best_code_mse.pth`：按 val code MSE 选择的最佳 mapper。
- `checkpoints/best_decoder_nmse.pth`：按 val 真实 CSI decoder NMSE 选择的最佳 mapper。
- `checkpoints/last.pth`：最后一个 epoch。
- `codewords/train_mapped_code.pt`、`val_mapped_code.pt`、`test_mapped_code.pt`：导出的映射后码字。

## 快速运行

默认示例实验是 `seed1014/transnet_transnet -> seed1024/transnet_transnet`：

```bash
bash adapter/scripts/run_adapter.sh
```

直接调用训练脚本：

```bash
bash adapter/scripts/train_adapter.sh
```

`train_adapter.sh` 会在后台启动训练，并把终端输出重定向到 `/dev/null`。训练日志请看实验目录下的 `run.log`。

覆盖参数示例：

```bash
gpu=1 epochs=200 lambda_recon=1000 bash adapter/scripts/run_adapter.sh
mapper_type=affine_linear epochs=20 bash adapter/scripts/run_adapter.sh
mapper_type=direct_mlp epochs=100 bash adapter/scripts/run_adapter.sh
```

Smoke test：

```bash
max_train_samples=64 max_eval_samples=32 epochs=1 batch_size=16 cpu=1 \
exp_dir=/tmp/adapter_smoke bash adapter/scripts/train_adapter.sh
```

## `train_adapter.sh` 参数说明

下面所有参数都可以通过环境变量覆盖，例如：

```bash
source_exp=... target_exp=... gpu=1 epochs=200 bash adapter/scripts/train_adapter.sh
```

### 实验输入

`source_exp`

source 实验目录。脚本默认从该目录读取：

```text
codewords/train_code.pt
codewords/val_code.pt
codewords/test_code.pt
```

默认值：

```text
exps/COST2100/in/base/seed${source_seed}/${source_encoder}_${source_decoder}
```

`source_seed`

source 实验 seed。会影响默认 `source_exp` 和默认 `exp_dir`。

默认值：

```text
1014
```

`source_encoder`

source 实验 encoder 名称。会影响默认 `source_exp` 和默认 `exp_dir` 中的架构层级。

默认值：

```text
transnet
```

`source_decoder`

source 实验 decoder 名称。只影响默认 `source_exp` 的实验目录名。

默认值：

```text
transnet
```

`target_exp`

target 实验目录。脚本默认从该目录读取 target 的三份 codeword。离线 affine 拟合和 code loss 都以这些 target codeword 为监督。

默认值：

```text
exps/COST2100/in/base/seed${target_seed}/${target_encoder}_${target_decoder}
```

`target_seed`

target 实验 seed。会影响默认 `target_exp`。

默认值：

```text
1024
```

`target_encoder`

target 实验 encoder 名称。会影响默认 `target_exp`。

默认值：

```text
transnet
```

`target_decoder`

target 实验 decoder 名称。会影响默认 `target_exp`、默认 `decoder` 和默认 decoder checkpoint 路径。

默认值：

```text
transnet
```

`target_decoder_exp`

用于加载 frozen target decoder 的实验目录。默认等于 `target_exp`。脚本读取：

```text
checkpoints/best_nmse.pth
args.json
```

如果你想把 code 对齐到一个 target codeword，但使用另一个 decoder 评估，可以单独覆盖这个参数。

`decoder_checkpoint`

frozen target decoder checkpoint 路径。

默认值：

```text
${target_decoder_exp}/checkpoints/best_nmse.pth
```

`decoder_args_json`

frozen target decoder 对应的 `args.json`。

默认值：

```text
如果 ${target_decoder_exp}/args.json 存在则使用它，否则不传入
```

`source_train_code`、`source_val_code`、`source_test_code`

显式指定 source 三个 split 的 codeword 路径。默认留空，表示使用：

```text
${source_exp}/codewords/{train,val,test}_code.pt
```

`target_train_code`、`target_val_code`、`target_test_code`

显式指定 target 三个 split 的 codeword 路径。默认留空，表示使用：

```text
${target_exp}/codewords/{train,val,test}_code.pt
```

### CSI 数据路径

`train_csi`

训练集真实 CSI 路径，用于 `lambda_recon > 0` 时的重建 loss，也用于训练集相关导出索引对齐。

默认值：

```text
/nfs5/zxd/Huawei/datasets/COST2100/in_train.pt
```

`val_csi`

验证集真实 CSI 路径，用于 val decoder NMSE 评估和选择 `best_decoder_nmse.pth`。

默认值：

```text
/nfs5/zxd/Huawei/datasets/COST2100/in_val.pt
```

`test_csi`

测试集真实 CSI 路径，只用于最终/周期性 test 评估，不参与 affine 拟合和训练。

默认值：

```text
/nfs5/zxd/Huawei/datasets/COST2100/in_test.pt
```

### Mapper 结构

`mapper_type`

选择码字映射器结构：

- `affine_residual_mlp`：默认方案。先闭式 affine，再 residual MLP 非线性修正。
- `affine_linear`：先闭式 affine，再恒等初始化的无激活线性层堆叠。主要用于 baseline。
- `direct_mlp`：不使用 affine 初始化，直接用 MLP 从 source code 学 target code。主要用于 baseline。

默认值：

```text
affine_residual_mlp
```

`hidden_dim`

residual MLP 或 direct MLP 的隐藏层维度。

默认值：

```text
1024
```

`num_blocks`

对 `affine_residual_mlp` 表示 residual block 数量；对 `affine_linear` 表示 identity linear 层数；对 `direct_mlp` 表示 MLP 层数。

默认值：

```text
4
```

`dropout`

MLP 中 dropout 概率。

默认值：

```text
0.0
```

`residual_scale`

residual 分支缩放系数，只对 `affine_residual_mlp` 有意义：

```text
z = z0 + residual_scale * f(z0)
```

默认值：

```text
0.1
```

`use_final_norm`

是否在 residual blocks 之后加最终 `LayerNorm`。`1` 表示开启，`0` 表示关闭。默认关闭，因为之前实验中 final norm 会明显破坏 affine 对齐后的尺度。

默认值：

```text
0
```

`no_block_norm`

是否关闭 residual block 内部的 `LayerNorm`。`1` 表示关闭 block norm，`0` 表示保留 block norm。

默认值：

```text
0
```

`train_affine`

是否把离线拟合得到的 `W,b` 作为可训练参数。`0` 表示冻结为 buffer，`1` 表示参与训练。默认冻结，避免破坏闭式 affine 初始解。

默认值：

```text
0
```

`align_ridge`

闭式 affine 拟合的 ridge 正则强度。拟合形式是：

```text
min ||source @ W + b - target||^2 + align_ridge * ||W||^2
```

默认值：

```text
1.0
```

### Loss 与训练

`lambda_code`

code space MSE loss 权重：

```text
MSE(mapped_code, target_code)
```

默认值：

```text
1.0
```

`lambda_recon`

真实 CSI 重建 loss 权重：

```text
MSE(target_decoder(mapped_code), csi_gt)
```

默认值是 `0.0`，表示只用 code loss 训练。如果希望直接针对真实 CSI NMSE 微调，可以设为正数。注意 code MSE 通常是 `1e-3` 量级，recon MSE 通常是 `1e-6` 量级；如果和 `lambda_code=1.0` 同时使用，`lambda_recon` 往往需要设到几百到几千才会有明显权重，例如 `500`、`1000` 或 `2000`。

注意：即使 `lambda_recon=0.0`，训练日志中的 `train_recon_mse` 也会正常计算；它只是不会参与反向传播和参数更新。

`epochs`

训练 epoch 数。

默认值：

```text
100
```

`batch_size`

训练和评估 batch size。

默认值：

```text
4096
```

`workers`

DataLoader worker 数。当前数据已是 `.pt` 张量，通常 `0` 就够。

默认值：

```text
0
```

`lr`

AdamW 学习率。

默认值：

```text
1e-3
```

`weight_decay`

AdamW weight decay。bias 和一维参数不会加 weight decay。

默认值：

```text
1e-4
```

`scheduler`

学习率调度器：

- `cosine`：warmup cosine。
- `const`：固定学习率。

默认值：

```text
cosine
```

`eta_min`

cosine scheduler 的最低学习率。

默认值：

```text
1e-5
```

`eval_every`

每隔多少个 epoch 在 val/test 上评估 code 指标和真实 CSI decoder NMSE。

默认值：

```text
1
```

### 导出与采样

`export_codewords`

训练结束后是否导出 mapped train/val/test codeword。`1` 表示导出，`0` 表示不导出。

默认值：

```text
1
```

`max_train_samples`

训练集最多使用多少样本。`0` 表示使用全部 train codeword。主要用于 smoke test 或快速调参。

默认值：

```text
0
```

`max_eval_samples`

val/test 最多使用多少样本。`0` 表示使用全部 val/test。主要用于 smoke test 或快速评估。

默认值：

```text
0
```

### 复现与设备

`seed`

随机种子。

默认值：

```text
2026
```

`gpu`

使用的 GPU id。仅在 `cpu=0` 时生效。

默认值：

```text
2
```

`cpu`

是否强制 CPU 运行。`1` 表示使用 CPU，`0` 表示使用 GPU。

默认值：

```text
0
```

`python_bin`

启动训练时使用的 Python 命令。默认要求当前 shell 已经能找到 `python`，例如先执行 `conda activate torch`。如果不想激活环境，可以显式指定解释器路径。

默认值：

```text
python
```

### Decoder fallback 参数

正常情况下这些参数从 `decoder_args_json` 读取，不需要手动设置。只有在没有可用 `args.json`，或者你明确要覆盖 decoder 构建配置时才需要调整。

`channel`

CSI 通道数。

默认值：

```text
2
```

`nt`

天线维。

默认值：

```text
32
```

`nc`

延迟/频域维。

默认值：

```text
32
```

`decoder`

decoder 名称。默认跟随 `target_decoder`。

默认值：

```text
transnet
```

`cr`

压缩率分母。

默认值：

```text
4
```

`d_model`

Transformer 类 decoder 的模型维度。

默认值：

```text
64
```

`dim_feedforward`

Transformer feedforward 隐藏维度。

默认值：

```text
2048
```

`hidden`

部分 decoder 使用的隐藏通道数。

默认值：

```text
16
```

`decoder_num_blocks`

部分 decoder 使用的 block 数。

默认值：

```text
2
```

### 输出路径

`exp_dir`

实验输出目录。如果不显式设置，脚本会按 mapper、source seed、source encoder、loss、学习率和 epoch 自动生成：

```text
adapter/exps/${mapper_type}/seed${source_seed}/${source_encoder}/code${lambda_code}_rec${lambda_recon}_lr${lr}_ep${epochs}
```

如果需要自定义默认目录的 seed、架构层级或最后一级名称，可以覆盖 `exp_seed`、`exp_arch`、`exp_name`。例如：

```bash
exp_arch=transnet_to_seed1024 exp_name=rec1000_lr2e-4 bash adapter/scripts/train_adapter.sh
```

## 推荐设置

根据 `seed1024 -> seed1014` 的实测，推荐优先使用：

```bash
mapper_type=affine_residual_mlp \
use_final_norm=0 \
train_affine=0 \
residual_scale=0.1 \
lambda_code=1.0 \
lambda_recon=0.0 \
bash adapter/scripts/train_adapter.sh
```

如果 code loss 已经收敛但真实 CSI NMSE 仍有提升空间，可以在第二阶段尝试：

```bash
lambda_code=1.0 lambda_recon=1000 lr=2e-4 epochs=100 \
bash adapter/scripts/train_adapter.sh
```
