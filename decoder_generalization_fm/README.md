# decoder_generalization_fm

这个子项目用于研究：

```text
全量 encoder codewords -> 生成对应 transnet decoder 全部参数
```

它面向 decoder 参数生成泛化研究：一个实验目录是一条样本，模型根据该实验的全量码字生成同实验配套的 transnet decoder 参数。

## 任务定义

每个实验目录视为一个样本：

```text
sample = {
  code: exp_dir/codewords/train_code.pt,          # (N, 512)
  decoder: exp_dir/checkpoints/best_nmse.pth,     # transnet decoder 参数
  args: exp_dir/args.json
}
```

训练时输入全量 train code，生成该实验对应的 transnet decoder 参数。

```text
condition: full codewords Z
start: theta_0 ~ N(0, 1)
target: zscore(theta_decoder)
loss: parameter-space flow matching loss
```

真实 CSI 不进入训练 loss，只用于周期评估：

```text
generated decoder + train code -> reconstructed CSI
NMSE(reconstructed CSI, train CSI)
```

## 数据划分

`data.txt` 使用一行一个实验目录的格式：

```text
train,exps/COST2100/in/seed42/transnet_transnet
test,exps/COST2100/in/seed3407/transnet_transnet
```

每行第一列是 split，只允许：

```text
train
test
```

第二列是实验目录。代码会自动解析：

```text
args.json
codewords/train_code.pt
checkpoints/best_nmse.pth
```

默认 `data.txt` 的设计目标：

- `train`：包含 seed42 和 seed2026 的若干基础 encoder 架构。
- `test`：同时包含跨 seed 样本和跨架构 held-out 样本。
- `seed42/transnet_transnet` 只是普通 train 样本，不做特殊处理。

## 参数归一化

只使用按 tensor 的 global zscore。

对 train split 中所有 decoder 参数，按 tensor name 单独计算：

```text
mean[name] = mean over all train decoders' tensor[name]
std[name]  = std  over all train decoders' tensor[name]
```

训练和测试都使用 train split 的统计量：

```text
theta_norm[name] = (theta[name] - mean[name]) / std[name]
theta_real[name] = theta_norm[name] * std[name] + mean[name]
```

统计量会保存为本地 `.pt` 缓存。缓存存在时直接读取，不重复计算。

## 条件编码

条件输入是全量码字矩阵：

```text
Z: (100000, 512)
```

第一版会参考 `decoder_param_fm` 保留三类条件编码方式：

```text
random          # 固定 seed 下随机取 K 条码字作为条件 token
svd             # 用 SVD/PCA summary 表示全量码字
set_transformer # learnable query cross-attention 到全量码字
```

## 训练样本粒度

一个 epoch 会遍历一次 `train` split 中的实验目录样本。训练脚本中的 `batch_size` 表示每次 optimizer update 使用几个实验目录样本，而不是用户码字行数。

```text
num_updates_per_epoch = ceil(num_train_experiments / batch_size)
```

每个样本内部的条件仍然是该实验的全量 `train_code.pt`。`max_condition_codes` 只用于 smoke test 或显存/速度受限时截断条件码字；默认 `0` 表示使用全量码字。

## 文件结构

```text
decoder_generalization_fm/
  data/data.txt
  README.md
  dataset.py
  param_utils.py
  models.py
  train.py
  evaluate.py
  sample.py
  scripts/
    make_data_txt.py
    train.sh
    run.sh
    evaluate.sh
```

## 训练

```bash
gpu=4 \
exp_name=svd_film_h512_b4 \
condition_extract=svd \
condition_inject=film \
hidden_dim=512 \
num_blocks=4 \
epochs=400 \
batch_size=2 \
bash decoder_generalization_fm/scripts/train.sh
```

脚本默认后台运行，实验目录为：

```text
decoder_generalization_fm/exps/${exp_name}
```

主要输出：

```text
run.log
args.json
history.json
checkpoints/best_loss.pth
checkpoints/last.pth
artifacts/param_meta.json
artifacts/train_tensor_zscore_stats.pt
tensorboard/
```

## 评估

```bash
exp_dir=decoder_generalization_fm/exps/svd_film_h512_b4 \
gpu=4 \
bash decoder_generalization_fm/scripts/evaluate.sh
```

评估会对 `data.txt` 中指定 split 的每个实验目录生成 decoder，然后用对应的 `train_code.pt` 解码，并和该实验 `args.json` 里的 `train_path` CSI 计算 NMSE。

## 1 epoch smoke test

```bash
gpu=4 \
background=0 \
exp_name=smoke_1ep \
epochs=1 \
batch_size=2 \
condition_extract=random \
condition_tokens=32 \
hidden_dim=64 \
cond_dim=64 \
num_blocks=1 \
time_dim=32 \
max_condition_codes=128 \
eval_every=1 \
eval_max_entries=1 \
eval_max_samples=64 \
eval_batch_size=32 \
bash decoder_generalization_fm/scripts/train.sh
```
