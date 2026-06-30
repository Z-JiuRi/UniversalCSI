# teacher code 加噪声敏感性测试

目的：在 teacher code `z_t` 上直接加随机噪声，然后送入固定 seed42 decoder，观察 fixed decoder NMSE 下降 1 dB 时对应的 code loss。

固定配置：

```text
teacher code: exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt
fixed decoder: exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth
data: /storage/hujiacong/zxd/datasets/cost2100/in_train.pt
N = 100000
code_dim = 512
```

噪声定义：

```text
Gaussian: ε ~ N(0, σ^2)
Laplace:  Var(ε) = σ^2，即 scale b = σ / sqrt(2)
z_noisy = z_t + ε
code loss = MSE(z_noisy, z_t)
```

## 结果

teacher 原始解码：

```text
NMSE = -29.103 dB
reconstruction MSE = 5.55e-7
```

插值估计 fixed decoder NMSE 下降 1 dB 时：

| noise | sigma/std | code MSE | code RMSE | code MAE | NMSE | recon MSE |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 0.02735 | 7.54e-4 | 0.02735 | 0.02182 | -28.103 dB | 6.99e-7 |
| Laplace | 0.02734 | 7.54e-4 | 0.02734 | 0.01933 | -28.103 dB | 6.99e-7 |

完整扫描：

- `noise_sweep.csv`
- `noise_sweep.json`
- `noise_sweep.png`

## 解释

Gaussian 和 Laplace 在相同方差下几乎重合，说明对这种独立同方差噪声，fixed decoder 的 1 dB 容忍阈值主要由噪声方差决定，而不是由 Gaussian/Laplace 的边缘形状决定。

1 dB gap 对应：

```text
code MSE ≈ 7.5e-4
code RMSE ≈ 0.0273
```

这给 code-only mapper 一个经验门槛：

```text
如果 mapped residual 像独立随机噪声，
要让 fixed decoder NMSE gap <= 1 dB，
code MSE 至少应压到约 7.5e-4 以下。
```

但当前 mapper 的 residual 不是独立同方差噪声，而是尖峰重尾，并且存在样本/维度尾部和 teacher 低方差方向残差。因此实际目标应该更严格：

```text
mean code MSE <= 5e-4
最好接近 1e-4 ~ 3e-4
同时压低 sample p95/p99 和 dim RMSE max
```

对比当前最好 mapper：

```text
hybrid/seed2026_transnet:
  mapped code MSE = 3.12e-3
  fixed decoder NMSE = -25.39 dB
  teacher gap = 3.72 dB
```

当前 code MSE 约为 1 dB 阈值的：

```text
3.12e-3 / 7.54e-4 ≈ 4.1 倍
```

这与实际 NMSE gap 仍有 3.72 dB 基本一致。

## 对下一步 code-only mapper 的含义

如果当前阶段仍坚持不引入 decoder，只做 code-to-code 映射，目标不应只看 mean MSE，还要看尾部：

```text
mean code MSE <= 5e-4
sample code MSE p95/p99 显著下降
dim RMSE max 显著下降
mapped residual kurtosis 下降
teacher PCA low-variance residual 下降
```

训练上建议：

```text
L = MSE(z_a, z_t)
  + λ_sample_tail * mean(top20% sample_code_mse)
  + λ_dim_tail    * mean(top5% dim_code_mse)
  + λ_white       * mean((P_t^T(z_a-z_t))^2 / (λ_t + eps))
```

这个实验只说明 fixed decoder 对“随机独立噪声”的平均容忍阈值；真实 mapper residual 更结构化，所以需要更严格的 code loss 和尾部控制。
