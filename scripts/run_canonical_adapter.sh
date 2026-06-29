#!/bin/bash

# 批量重跑 aux_pca_1e-2_code_mean1e-4_cov1e-4 的 adapter 实验。
#
# 目标：
# - 对 ep100 目录中已有的 11 个 encoder 实验，全部重跑 400 epoch。
# - decoder 固定为：
#     exps/COST2100/in/encoder_canonical/aux_pca_1e-2_code_mean1e-4_cov1e-4/seed42_transnet_transnet/checkpoints/best_nmse.pth
# - teacher code 固定为：
#     exps/COST2100/in/encoder_canonical/aux_pca_1e-2_code_mean1e-4_cov1e-4/seed42_transnet_transnet/codewords/train_code.pt
# - encoder 使用各自 seed 和架构：
#     seed2026/3407/42 x transnet/csinet/crnet/clnet 中已经跑过 ep100 的组合。
# - 任务按列表顺序交替分配到 GPU 1 和 GPU 4。
#
# 运行：
#   bash scripts/run_canonical_adapter.sh
#
# 覆盖参数示例：
#   epochs=400 gpu_list="1 4" bash scripts/run_canonical_adapter.sh
#   epochs=200 gpu_list="6" bash scripts/run_canonical_adapter.sh

set -euo pipefail

canonical_scheme=${canonical_scheme:-aux_pca_1e-2_code_mean1e-4_cov1e-4}
canonical_root=${canonical_root:-exps/COST2100/in/encoder_canonical/${canonical_scheme}}

decoder_seed=${decoder_seed:-42}
decoder_encoder=${decoder_encoder:-transnet}
decoder_decoder=${decoder_decoder:-transnet}

epochs=${epochs:-400}
gpu_list=${gpu_list:-"1 4"}

adapter=${adapter:-gated_lowrank_affine_mlp}
adapter_rank=${adapter_rank:-32}
adapter_hidden_dim=${adapter_hidden_dim:-2048}
adapter_gate_init=${adapter_gate_init:-0.1}
lambda_recon=${lambda_recon:-1.0}
lambda_code=${lambda_code:-1e-3}
lambda_fc=${lambda_fc:-1e-2}
lambda_recT=${lambda_recT:-0.0}
lr_init=${lr_init:-5e-4}

fixed_decoder_ckpt="${canonical_root}/seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}/checkpoints/best_nmse.pth"
teacher_code="${canonical_root}/seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}/codewords/train_code.pt"

if [ ! -f "${fixed_decoder_ckpt}" ]; then
  echo "Missing fixed decoder checkpoint: ${fixed_decoder_ckpt}" >&2
  exit 1
fi

if [ ! -f "${teacher_code}" ]; then
  echo "Missing teacher code: ${teacher_code}" >&2
  exit 1
fi

experiments=(
  "2026 transnet transnet"
  "2026 crnet transnet"
  "2026 csinet transnet"
  "2026 clnet transnet"
  "3407 transnet transnet"
  "3407 crnet transnet"
  "3407 csinet transnet"
  "3407 clnet transnet"
  "42 crnet transnet"
  "42 csinet transnet"
  "42 clnet transnet"
)

read -r -a gpus <<< "${gpu_list}"
if [ "${#gpus[@]}" -eq 0 ]; then
  echo "gpu_list is empty" >&2
  exit 1
fi

idx=0
for item in "${experiments[@]}"; do
  read -r seed encoder decoder <<< "${item}"
  gpu="${gpus[$((idx % ${#gpus[@]}))]}"
  idx=$((idx + 1))

  encoder_ckpt="${canonical_root}/seed${seed}_${encoder}_${decoder}/checkpoints/best_nmse.pth"
  if [ ! -f "${encoder_ckpt}" ]; then
    echo "[skip] missing encoder checkpoint: ${encoder_ckpt}" >&2
    continue
  fi

  exp_name="COST2100/in/encoder_canonical/adapter/${canonical_scheme}/${adapter}/rank${adapter_rank}_hidden${adapter_hidden_dim}_gate${adapter_gate_init}_code${lambda_code}_fc${lambda_fc}_lr${lr_init}_ep${epochs}/enc_seed${seed}_${encoder}_${decoder}_dec_seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}"

  echo "[launch] gpu=${gpu} encoder=seed${seed}_${encoder}_${decoder} decoder=seed${decoder_seed}_${decoder_encoder}_${decoder_decoder}"
  echo "         exp_name=${exp_name}"

  canonical_scheme="${canonical_scheme}" \
  encoder="${encoder}" decoder="${decoder}" \
  seed="${seed}" decoder_seed="${decoder_seed}" gpu="${gpu}" epochs="${epochs}" \
  pretrained_encoder="${encoder_ckpt}" \
  pretrained_decoder="${fixed_decoder_ckpt}" \
  teacher_code="${teacher_code}" \
  adapter="${adapter}" adapter_rank="${adapter_rank}" \
  adapter_hidden_dim="${adapter_hidden_dim}" \
  adapter_gate_init="${adapter_gate_init}" \
  lambda_recon="${lambda_recon}" lambda_code="${lambda_code}" \
  lambda_fc="${lambda_fc}" lambda_recT="${lambda_recT}" \
  lr_init="${lr_init}" exp_name="${exp_name}" \
  bash scripts/train_canonical_adapter.sh
done
