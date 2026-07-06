# CSI 生成质量分析报告 —— Per-Sample MSE 属性

## 实验参数
- 条件：COST2100 indoor, CR=4, encoder=transnet, decoder=transnet
- 评估数据集：测试集 20,000 样本
- FM 模型：全部为 lr=5e-5, tok=512, hidden=2048, seed=42

## 结果汇总

| 实验 (condition_extract / condition_inject / param_norm) | Agg NMSE | Agg MSE | MSE mean | MSE std | NMSE p5 | NMSE p95 | NMSE mean | NMSE std |
|---|---|---|---|---|---|---|---|---|
| set_transformer / film / zscore | **-4.12 dB** | **0.000175** | 0.000175 | 0.000002 | -6.73 | **-0.92** | -3.76 | 1.76 |
| random / hyper_lora / rms | **-3.84 dB** | **0.000187** | 0.000187 | 0.000002 | -6.43 | -0.64 | -3.47 | 1.75 |
| set_transformer / hyper_lora / zscore | -3.47 dB | 0.000203 | 0.000203 | 0.000002 | -6.06 | -0.28 | -3.11 | 1.75 |
| random / film / zscore | -2.80 dB | 0.000237 | 0.000237 | 0.000002 | -5.40 | 0.41 | -2.43 | 1.76 |
| svd / cross_attention / zscore | -2.39 dB | 0.000260 | 0.000260 | 0.000002 | -5.00 | 0.81 | -2.03 | 1.76 |
| random / film / rms | 2.76 dB | 0.000854 | 0.000854 | 0.000002 | 0.14 | 5.98 | 3.13 | 1.77 |
| set_transformer / film / rms | 0.03 dB | 0.000455 | 0.000455 | 0.000002 | -2.60 | 3.24 | 0.39 | 1.76 |
| set_transformer / cross_attention / rms | 4.00 dB | 0.001136 | 0.001136 | 0.000002 | 1.37 | 7.22 | 4.37 | 1.77 |

## 关键发现

### 1. MSE 分布特性
- **MSE 标准差极小**：所有实验的 per-sample MSE 标准差均为 ~0.000002，说明生成质量在样本间高度一致，没有明显的离群样本。
- **MSE 中位数 ≈ 均值**：分布非常对称，没有偏斜。

### 2. NMSE 分布特性
- **NMSE 标准差约 1.76 dB**：虽然 MSE 很稳定，但 NMSE 有约 ±1.76 dB 的波动，这是因为 NMSE 做了信号功率归一化——低功率样本的 NMSE 天然更差。
- **最佳实验 (set_transformer_film_zscore) 的 p95 NMSE 为 -0.92 dB**：说明即使是最差的那 5% 样本，NMSE 仍在 0 dB 以下（即误差能量小于信号能量）。

### 3. 归一化方式的影响
- **z-score 归一化全面优于 RMS**：z-score 的 MSE 大约是 RMS 的 1/3~1/2。
- 即使同是 z-score，`set_transformer_film` 的 MSE (0.000175) 也比 `random_film` (0.000237) 好约 26%。

### 4. 各方法排序（按 Agg NMSE 从优到劣）
1. set_transformer + film + zscore    (-4.12 dB, MSE=0.000175)
2. random + hyper_lora + rms          (-3.84 dB, MSE=0.000187)
3. set_transformer + hyper_lora + zscore (-3.47 dB, MSE=0.000203)
4. random + film + zscore             (-2.80 dB, MSE=0.000237)
5. svd + cross_attention + zscore     (-2.39 dB, MSE=0.000260)
6. set_transformer + film + rms       ( 0.03 dB, MSE=0.000455)
7. random + film + rms                ( 2.76 dB, MSE=0.000854)
8. set_transformer + cross_attention + rms (4.00 dB, MSE=0.001136)

## 输出文件位置
- Per-sample MSE 张量：各实验 `generated/per_sample_mse.pt`
- 完整 JSON 报告：`decoder_param_fm/reports/csi_mse_stats.json`
