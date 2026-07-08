# 显式列出 decoder LoRA + gated residual code adapter 实验命令。
#
# 数据流：
#   source code -> closed-form affine -> z0
#   z1 = z0
#      + gate_lr  * Up(Down(LN(z0)))
#      + gate_mlp * W2 GELU(W1 LN(z0))
#   z1 -> seed42 fixed decoder + LoRA(fc/FFN) -> reconstructed CSI
#
# 说明：
#   1. 每一段都是一条独立训练命令，按需注释/取消注释即可。
#   2. 默认后台运行，实际日志写到对应实验目录 run.log。
#   3. lambda_code 约束 z1 贴近 teacher code。
#   4. lambda_delta 约束 z1 不要偏离 affine 后的 z0 太远。
#   5. fc_lora_rank/alpha 和 ffn_lora_rank/alpha 已分开控制。
#
# 用法：
#   bash decoder_lora/scripts/run_decoder_lora.sh

set -euo pipefail

common_target_code=${target_code:-exps/COST2100/in/base/seed42/transnet_transnet/codewords/train_code.pt}
common_csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
common_val_csi_path=${val_csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_val.pt}
common_test_csi_path=${test_csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_test.pt}
common_decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/base/seed42/transnet_transnet/checkpoints/best_nmse.pth}
common_decoder_args_json=${decoder_args_json:-exps/COST2100/in/base/seed42/transnet_transnet/args.json}

# ==================== 模板（按需取消注释） ====================
# 用法：
#   挑选一段，取消注释，调整参数后直接运行。
#
# --- 仅 LoRA（无 code adapter）---
seed=2026 \
encoder=transnet \
source_name=seed${seed}_${encoder}_transnet \
source_code=exps/COST2100/in/base/seed${seed}/${encoder}_transnet/codewords/train_code.pt \
target_code=${common_target_code} \
csi_path=${common_csi_path} \
val_csi_path=${common_val_csi_path} \
test_csi_path=${common_test_csi_path} \
decoder_checkpoint=${common_decoder_checkpoint} \
decoder_args_json=${common_decoder_args_json} \
align_mode=affine \
align_with_val_code=0 \
lora_target=fc_ffn \
fc_lora_rank=256 \
fc_lora_alpha=2048 \
ffn_lora_rank=32 \
ffn_lora_alpha=256 \
code_adapter=none \
lambda_code=0 \
lambda_delta=0 \
gpu=7 \
lr=2e-4 \
eta_min=2e-5 \
epochs=400 \
bash decoder_lora/scripts/train_decoder_lora.sh
#
# --- LoRA + AffineResMLP 码字映射器（联合训练）---
# seed=2026 \
# encoder=transnet \
# source_name=seed${seed}_${encoder}_transnet \
# source_code=exps/COST2100/in/base/seed${seed}/${encoder}_transnet/codewords/train_code.pt \
# target_code=${common_target_code} \
# csi_path=${common_csi_path} \
# decoder_checkpoint=${common_decoder_checkpoint} \
# decoder_args_json=${common_decoder_args_json} \
# align_mode=identity \
# lora_target=fc_ffn \
# fc_lora_rank=32 \
# fc_lora_alpha=256 \
# ffn_lora_rank=8 \
# ffn_lora_alpha=64 \
# code_adapter=affine_res_mlp \
# code_hidden_dim=512 \
# code_num_blocks=2 \
# code_residual_scale=1.0 \
# code_use_final_norm=1 \
# lambda_code=0 \
# lambda_delta=0 \
# gpu=6 \
# lr=5e-4 \
# eta_min=2e-4 \
# epochs=400 \
# bash decoder_lora/scripts/train_decoder_lora.sh
#
# --- LoRA + GatedCodeResidualAdapter（联合训练）---
# seed=2026 \
# encoder=transnet \
# source_name=seed${seed}_${encoder}_transnet \
# source_code=exps/COST2100/in/base/seed${seed}/${encoder}_transnet/codewords/train_code.pt \
# target_code=${common_target_code} \
# csi_path=${common_csi_path} \
# decoder_checkpoint=${common_decoder_checkpoint} \
# decoder_args_json=${common_decoder_args_json} \
# align_mode=affine \
# lora_target=fc_ffn \
# fc_lora_rank=256 \
# fc_lora_alpha=2048 \
# ffn_lora_rank=16 \
# ffn_lora_alpha=128 \
# code_adapter=gated_lr_mlp \
# code_lowrank_rank=128 \
# code_mlp_hidden=512 \
# code_gate_lr=0.1 \
# code_gate_mlp=0.1 \
# lambda_code=0 \
# lambda_delta=0 \
# gpu=1 \
# lr=5e-4 \
# eta_min=2e-4 \
# epochs=400 \
# bash decoder_lora/scripts/train_decoder_lora.sh
