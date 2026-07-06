# staged_mlp_lora

这个目录实现“三阶段码字对齐 + decoder LoRA 精修”的方案。

## 核心想法

1. 先用闭式 affine 把 source encoder 的码字粗对齐到 seed42 teacher code。
2. 再训练一个 residual MLP mapper，只优化 `MSE(mapped_code, teacher_code)`，不引入 decoder loss。
3. 固定 mapped code，只训练 seed42 decoder 上的 LoRA，优化 `MSE(reconstruction, raw CSI)`。

这样做的目的，是把“码字空间对齐”和“decoder 轻量精修”拆开：

- mapper 负责把不同 seed/架构的码字变成固定 decoder 更容易识别的码字；
- LoRA 负责修复 mapper 仍然没有完全消除的 decoder 侧重建误差；
- Stage 3 不再拉 code/fc/teacher reconstruction，避免多个目标相互牵制。

## 主要脚本

- `scripts/train_mapper.sh`：启动 Stage 1/2，训练 `affine + residual MLP`，导出 `codewords/mapped_code.pt`。
- `scripts/train_lora.sh`：启动 Stage 3，读取 mapped code，只训练 decoder LoRA。
- `scripts/run_staged_mlp_lora.sh`：批量启动多组 mapper，并在 mapped code 生成后启动对应 LoRA。
- `scripts/test_staged_nmse.sh`：读取原始码字、mapper checkpoint、LoRA checkpoint，计算完整链路真实 NMSE。

## 单独训练 mapper

```bash
source_name=seed2026_transnet_transnet \
source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
hidden_dim=1024 \
num_blocks=4 \
gpu=0 \
bash staged_mlp_lora/scripts/train_mapper.sh
```

输出：

```text
staged_mlp_lora/exps/mapper/.../codewords/mapped_code.pt
```

## 单独训练 LoRA

```bash
mapper_exp_dir=staged_mlp_lora/exps/mapper/affine_mlp_h1024_b4_rs1.0_drop0.0_lr5e-4_ep400/seed2026_transnet_transnet_to_seed42 \
source_name=seed2026_transnet_transnet_mapped_h1024_b4 \
fc_lora_rank=256 \
fc_lora_alpha=1024 \
ffn_lora_rank=16 \
ffn_lora_alpha=64 \
gpu=4 \
bash staged_mlp_lora/scripts/train_lora.sh
```

默认 LoRA 训练配置：

```text
align_mode=identity
code_adapter=none
lambda_code=0
lambda_delta=0
lambda_recT=0
lambda_fc=0
```

因此 Stage 3 的优化目标就是重建 MSE。

## 批量运行

```bash
gpus="0 4 6 7" bash staged_mlp_lora/scripts/run_staged_mlp_lora.sh
```

脚本会先启动 mapper，再等待 `mapped_code.pt` 生成，然后启动对应 LoRA。

## 测试完整链路 NMSE

```bash
mapper_exp_dir=staged_mlp_lora/exps/mapper/affine_mlp_h1024_b4_rs1.0_drop0.0_lr5e-4_ep400/seed2026_transnet_transnet_to_seed42 \
lora_exp_dir=staged_mlp_lora/exps/lora/identity_fc_ffn_fcr256a1024_ffnr16a64_rec_only_lr5e-4_eta1e-4_ep400/seed2026_transnet_transnet_mapped_h1024_b4_to_seed42 \
source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
gpu=0 \
bash staged_mlp_lora/scripts/test_staged_nmse.sh
```

这个测试会输出三项：

```text
raw source_code -> seed42 base decoder
mapper(source_code) -> seed42 base decoder
mapper(source_code) -> seed42 LoRA decoder
```

如果想顺便保存当前 mapper checkpoint 生成的 mapped code：

```bash
save_mapped_code=/tmp/mapped_code.pt bash staged_mlp_lora/scripts/test_staged_nmse.sh
```

## 推荐大小

- 小 mapper：`hidden_dim=512, num_blocks=4`，约 2M 量级，适合先验证 transnet 跨 seed。
- 中 mapper：`hidden_dim=1024, num_blocks=4`，约 4M 量级，适合跨架构。
- LoRA 默认：`fc_lora_rank=256, ffn_lora_rank=16`，约 0.8M 量级。

总参数量大致为：

```text
小 mapper + LoRA: 约 3M
中 mapper + LoRA: 约 5M
```

这比当前较大的 mapper/FM 方案小，同时比只训练 decoder LoRA 多了一层显式码字对齐。
