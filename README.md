# UniversalCSI



#### WAIRD adapter

```bash
train_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/train.pt \
val_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt \
test_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt \
nt=64 nc=64 encoder=transnet decoder=transnet gpu=1 seed=3407 lr_init=3e-4 weight_decay=0 \
pretrained_encoder=exps/WAIRD/seed3407/transnet_transnet/checkpoints/best_nmse.pth \
pretrained_decoder=exps/WAIRD/seed42/transnet_transnet/checkpoints/best_nmse.pth \
teacher_code=exps/WAIRD/seed42/transnet_transnet/codewords/train_code.pt \
lambda_recon=1.0 \
lambda_code=0.0 \
exp_name=WAIRD/adapter/${adapter}/seed${seed}_lambda_recon${lambda_recon}_lambda_code${lambda_code}_lr${lr_init} \
bash scripts/train_adapter.sh
```

#### COST2100 adapter

```bash
train_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/train.pt \
val_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt \
test_path=/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt \
nt=64 nc=64 encoder=transnet decoder=transnet gpu=1 seed=3407 lr_init=3e-4 weight_decay=0 \
pretrained_encoder=exps/WAIRD/seed3407/transnet_transnet/checkpoints/best_nmse.pth \
pretrained_decoder=exps/WAIRD/seed42/transnet_transnet/checkpoints/best_nmse.pth \
teacher_code=exps/WAIRD/seed42/transnet_transnet/codewords/train_code.pt \
lambda_recon=1.0 \
lambda_code=0.0 \
exp_name=WAIRD/adapter/${adapter}/seed${seed}_recon${lambda_recon}_code${lambda_code}_lr${lr_init} \
bash scripts/train_adapter.sh
```





UniversalCSI 是一个用于 CSI 反馈压缩与重建的 PyTorch 实验框架。当前代码只保留端到端自编码训练路径：

```text
CSI input -> encoder -> decoder -> reconstructed CSI
```

默认单样本 CSI 维度为 `(2, 32, 32)`，其中 `2` 表示实部和虚部两个通道，`nt` 和 `nc` 分别表示天线维与延迟/频域维。实际维度由 `--channel`、`--nt`、`--nc` 控制。

## 项目结构

```text
main.py                   训练/评估入口
dataloader/dataloader.py  .pt 数据加载与 DataLoader 构建
models/UniversalCSI.py    通用模型工厂，组合 encoder 和 decoder
models/encoders/          CSI 压缩编码器
models/decoders/          CSI 重建解码器
utils/parser.py           CLI 参数定义
utils/init.py             设备初始化、模型初始化、整模型 checkpoint 加载
utils/solver.py           训练、验证、测试、checkpoint 保存和 codeword 导出
utils/statics.py          MSE/NMSE 统计
scripts/train.sh          基础端到端训练脚本
scripts/test.sh           整模型 checkpoint 评估脚本
```

已删除并不再支持的旧路径包括：partial encoder/decoder loading、frozen decoder training、CodeAdapter、teacher code loss、code-only training、LoRA、只训练 `decoder.fc_decoder`。

## 模型

所有 encoder 输出 `(B, code_dim)`，所有 decoder 接收 `(B, code_dim)` 并输出 `(B, channel, nt, nc)`。

```text
input_dim = channel * nt * nc
code_dim = input_dim // cr
```

默认 `channel=2, nt=32, nc=32, cr=4` 时，`input_dim=2048`，`code_dim=512`，压缩率为 `1/4`。

可选 encoder：

```text
csinet, cnn, cbam_cnn, crnet, clnet, transnet, resnet, dscnn,
convnext, mlp_mixer, attention_cnn, swin, mlp_ae, sparse_resnet
```

可选 decoder：

```text
transnet, cnn_residual, hybrid
```

## 数据格式

`main.py` 需要显式传入训练、验证和测试数据：

```text
--train_path path/to/train.pt
--val_path   path/to/val.pt
--test_path  path/to/test.pt
```

`.pt` 文件应保存 PyTorch Tensor，推荐形状为 `(N, channel, nt, nc)`。如果是二维展平张量，DataLoader 会按当前 `--channel --nt --nc` reshape 为 `(N, channel, nt, nc)`。数据会转换为 `float32`。

## 训练

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

也可以使用脚本：

```bash
bash scripts/train.sh
```

脚本参数通过环境变量覆盖，例如：

```bash
encoder=convnext decoder=hybrid seed=42 gpu=0 bash scripts/train.sh
```

## 评估

整模型评估使用 `--evaluate --pretrained`：

```bash
python main.py \
  --evaluate \
  --pretrained exps/COST2100/in/seed42/transnet_hybrid/checkpoints/best_nmse.pth \
  --exp_name COST2100/in/test/transnet_hybrid \
  --train_path ./COST2100/in_train.pt \
  --val_path ./COST2100/in_val.pt \
  --test_path ./COST2100/in_test.pt \
  --epochs 1 \
  --batch_size 32 \
  --workers 0 \
  --cr 4 \
  --encoder transnet \
  --decoder hybrid \
  --nt 32 \
  --nc 32
```

或使用：

```bash
bash scripts/test.sh
```

## 输出

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

`codewords/train_code.pt` 是训练集 encoder 输出，用于离线分析；当前训练代码不会再把它作为 teacher code 读回参与训练。

## 指标

训练损失是重建 MSE。测试阶段由 `utils/statics.py::evaluator()` 累加全测试集误差能量和信号能量，输出全局 NMSE：

```text
NMSE = 10 * log10(sum(|error|^2) / sum(|gt|^2))
```

当前 `evaluator()` 不做 `-0.5` 去中心化。