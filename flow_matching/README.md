# Flow Matching Codeword Translation

这个目录实现独立的 code-only flow matching 码字转换，不引入 decoder loss。

## 核心思路

给定 source encoder 产生的码字 `z_s` 和固定 decoder 对应 teacher 码字 `z_t`，先构造起点 `z0`：

- `align_mode=identity`：`z0 = z_s`
- `align_mode=procrustes`：先用确定性正交 Procrustes 把 `z_s` 旋转到 `z_t` 附近
- `align_mode=affine`：先用岭回归全仿射把 `z_s` 映射到 `z_t` 附近

训练时采样 `t in [0,1]`：

```text
x_t = (1 - t) z0 + t z_t
v_target = z_t - z0
v_theta = f_theta(x_t, t, z_s, z0)
loss = MSE(v_theta, v_target) + lambda_endpoint * MSE(x_t + (1 - t) v_theta, z_t)
```

推理时从 `z0` 出发积分 ODE：

```text
dz / dt = f_theta(z, t, z_s, z0)
```

最终导出 `mapped_code.pt`，可以继续用 `mapper/test_decoder_nmse_from_code.py` 测固定 decoder 下的 NMSE。

## 训练

```bash
bash flow_matching/scripts/train_flow_matching.sh
align_mode=affine gpu=4 bash flow_matching/scripts/train_flow_matching.sh
lr=1e-4 eta_min=1e-5 align_mode=affine gpu=4 bash flow_matching/scripts/train_flow_matching.sh
source_name=seed2026_clnet_transnet source_code=exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt gpu=4 bash flow_matching/scripts/train_flow_matching.sh
```

默认学习率调度使用主项目同款 `10% warmup + cosine annealing`：

```text
T_max = epochs * len(train_loader)
T_warmup = 0.1 * T_max
eta_min = 5e-5
```

如果初始学习率降到 `1e-4`，建议同时设置 `eta_min=1e-5`，否则退火下限相对偏高。

批量跑：

```bash
gpus="0 4 6 7" epochs=400 background=1 bash flow_matching/scripts/run_flow_matching.sh
```

## 测试和导出

```bash
exp_dir=flow_matching/exps/... bash flow_matching/scripts/test_flow_matching.sh
exp_dir=flow_matching/exps/... bash flow_matching/scripts/export_mapped_code.sh
```

固定 decoder NMSE 可以复用 mapper 里的测试脚本：

```bash
code_path=flow_matching/exps/.../codewords/mapped_code.pt result_name=flow_matching_xxx gpu=4 batch_size=1024 bash mapper/scripts/test_decoder_nmse_from_code.sh
```
