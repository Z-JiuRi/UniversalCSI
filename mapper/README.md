# Mapper 实验说明

`mapper/` 是独立的 codeword-to-codeword 映射实验目录，用来验证：

```text
z_source = E_source(x)
z_teacher = E_seed42_transnet(x)
mapper(z_source) -> z_teacher
```

第一阶段可以不接 decoder，只训练 mapper 让不同 seed/架构的码字对齐到基准 teacher code。
第二阶段支持接入固定 seed42 decoder，把码字误差和 decoder 输出误差一起优化。

## 默认基准

```text
exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt
```

## 支持模型

- `identity`：不训练，用来做下限。
- `mlp` / `deep_mlp` / `residual_mlp`：多层 residual MLP。
- `flow` / `coupling_flow`：affine coupling flow，适合高维可逆坐标变换。
- `hybrid_flow_mlp`：先 flow，再 residual MLP 微调。

## 训练

单个实验：

```bash
source_name=seed2026_transnet_transnet \
source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
mapper=delta_mlp \
hidden_dim=256 \
num_blocks=2 \
residual_mapping=1 \
align_mode=affine \
residual_condition=source_start \
lambda_smoothl1=0.0 \
lambda_sample_tail=0.0 \
lambda_dim_tail=0.0 \
lambda_whiten=0.0 \
gpu=0 \
bash mapper/scripts/train_mapper.sh
```

批量实验：

```bash
bash mapper/scripts/run_mapper.sh
```

1 epoch 真实数据测试：

```bash
epochs=1 gpu=1 bash mapper/scripts/run_mapper.sh
```

每个实验会保存到：

```text
mapper/exps/<mapper>/<source>_to_seed42_transnet_...
```

产物包括：

- `args.json`
- `history.json`
- `metrics.json`
- `run.log`
- `checkpoints/best_loss.pth`
- `checkpoints/best_mse.pth`
- `codewords/mapped_code.pt`
- `tensorboard/events.*`
- `mapped_code.pt`：兼容旧分析脚本保留的一份副本

训练脚本内部会写 `run.log`，因此可以后台静默运行：

```bash
mapper=mlp gpu=0 bash mapper/scripts/train_mapper.sh > /dev/null 2>&1 &
```

批量脚本内部已经对子任务使用静默后台方式：

```bash
mapper=mlp epochs=400 gpus=0,4,6,7 bash mapper/scripts/run_mapper.sh
```

## 分析

运行：

```bash
python mapper/analyze_codewords.py
```

输出：

```text
mapper/reports/codeword_analysis/
  code_stats.csv
  pair_stats.csv
  codeword_analysis.md
  figures/*.png
```

分析会比较 source code 和 seed42 teacher code 的 MSE、cosine、L2、scale、effective rank、残差协方差结构等。

## Loss

默认只用纯 MSE：

```text
MSE(mapper(z_source), z_teacher)
```

可选辅助项：

- `lambda_cos`：加入 cosine loss。
- `lambda_cov`：加入残差 covariance offdiag 正则。

建议先用纯 MSE 判断 mapper 表达力。如果 MSE 无法降到 `1e-4` 级别，再考虑换模型结构，而不是先堆 loss。

### 第一阶段 code-only loss

当前第一阶段不引入 decoder 信息，只优化 `mapper(z_source) -> z_teacher`。除基础 MSE 外，训练脚本还支持以下可选项：

- `lambda_smoothl1` / `smoothl1_beta`：加入 `SmoothL1(z_a, z_t)`，用于稳定重尾 residual。
- `lambda_sample_tail` / `sample_tail_ratio`：对 batch 内 code MSE 最大的一部分样本加权，例如 top 20%。
- `lambda_dim_tail` / `dim_tail_ratio`：对 code 维度上 MSE 最大的一部分维度加权，例如 top 5%。
- `lambda_whiten` / `whiten_eps_ratio`：teacher PCA whitened pair loss，强调 teacher 低方差方向的 pairwise 对齐。

示例：

```bash
mapper=mlp lambda_smoothl1=0.5 lambda_sample_tail=0.1 \
lambda_dim_tail=0.05 lambda_whiten=1e-4 gpu=0 \
bash mapper/scripts/train_mapper.sh
```

批量跑多组 code-only loss，并在 `0,4,6,7` 上循环分配：

```bash
mapper=mlp epochs=400 gpus=0,4,6,7 bash mapper/scripts/run_mapper.sh
```

先看将要启动哪些任务：

```bash
dry_run=1 bash mapper/scripts/run_mapper.sh
```

### 第二阶段 decoder-aware loss

第二阶段仍然以 `mapper(z_source) -> z_teacher` 的 code MSE 为主，同时把固定 seed42 decoder 接入 loss。目标是让 mapper 不只在欧氏码字空间接近 teacher code，也在固定 decoder 真正敏感的重建空间接近。

训练脚本新增以下可选项：

- `lambda_recT`：`MSE(D_teacher(z_a), D_teacher(z_t))`，约束 adapter 后 code 经过固定 decoder 的输出接近 teacher code 的 decoder 输出。
- `lambda_rec`：`MSE(D_teacher(z_a), x)`，直接约束固定 decoder 重建结果接近原始 CSI。
- `lambda_fc`：`MSE(fc_decoder(z_a), fc_decoder(z_t))`，约束固定 decoder 第一层后的隐空间，通常比只看 code 更贴近 decoder 的有效坐标。
- `lambda_decoder_tail` / `decoder_tail_ratio`：对 batch 内重建误差最大的样本加权，处理 code MSE 平均值不大但少数样本重建很差的问题。

单个实验示例：

```bash
lambda_recT=1.0 lambda_rec=1.0 lambda_fc=1e-2 lambda_decoder_tail=0.1 \
gpu=0 bash mapper/scripts/train_mapper.sh
```

批量跑第二阶段实验，默认在 `0,4,6,7` 四张 GPU 上循环分配：

```bash
bash mapper/scripts/run_mapper_decoder_aware.sh
```

批量跑 code-only 约束和 decoder-aware 约束的组合实验：

```bash
bash mapper/scripts/run_mapper_combined_losses.sh
```

这个组合脚本会把 `SmoothL1`、sample tail、dim tail、teacher whitening 等 code-only 约束，与 `recT`、`rec`、`fc`、decoder tail 等固定 decoder 约束一起测试。默认输出到：

```text
mapper/exps_combined_losses/<mapper>/<config>/<source>_to_seed42_transnet_...
```

常用覆盖：

```bash
mapper=mlp epochs=400 gpus=0,4,6,7 bash mapper/scripts/run_mapper_decoder_aware.sh
mapper=mlp epochs=400 gpus=0,4,6,7 bash mapper/scripts/run_mapper_combined_losses.sh
dry_run=1 bash mapper/scripts/run_mapper_decoder_aware.sh
dry_run=1 bash mapper/scripts/run_mapper_combined_losses.sh
overwrite=1 bash mapper/scripts/run_mapper_decoder_aware.sh
```

默认固定 decoder 和 teacher code 都来自：

```text
exps/COST2100/in/seed42/transnet_transnet
```

第二阶段结果默认保存到：

```text
mapper/exps_decoder_aware/<mapper>/<config>/<source>_to_seed42_transnet_...
```

每个实验同样包含 `run.log`、`tensorboard/events.*`、`checkpoints/best_loss.pth`、`checkpoints/best_mse.pth`、`codewords/mapped_code_best_loss.pt`、`codewords/mapped_code_best_mse.pt` 和 `metrics.json`，可以直接后台静默运行。

## 固定 decoder NMSE 测试

测试某个已保存 codeword 在 seed42 固定 decoder 下的重建 NMSE：

```bash
code_path=mapper/exps/hybrid/seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400/mapped_code.pt \
gpu=1 bash mapper/scripts/test_decoder_nmse_from_code.sh
```

测试 teacher code 上限对照：

```bash
code_path=exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt \
result_name=teacher_code \
gpu=1 bash mapper/scripts/test_decoder_nmse_from_code.sh
```

批量测试 `mapper/exps` 下所有 `mapped_code.pt`：

```bash
gpu=1 bash mapper/scripts/run_decoder_nmse_for_mapped_codes.sh
```

结果默认保存到：

```text
mapper/reports/decoder_nmse/*.json
```
