# Mapper 实验说明

`mapper/` 是独立的 codeword-to-codeword 映射实验目录，用来验证：

```text
z_source = E_source(x)
z_teacher = E_seed42_transnet(x)
mapper(z_source) -> z_teacher
```

这里暂时不接 decoder，只训练 mapper 让不同 seed/架构的码字对齐到基准 teacher code。

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
source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
source_name=seed2026_transnet_transnet \
mapper=flow gpu=1 \
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
- `best_mapper.pth`
- `mapped_code.pt`
- 对应 `.log`

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
