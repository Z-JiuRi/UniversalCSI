#!/bin/bash

# 测试完整链路真实 NMSE：
#   raw source_code -> saved mapper -> mapped_code -> saved LoRA decoder -> CSI reconstruction
#
# 同时会输出：
#   1. raw source_code 直接进 seed42 base decoder 的 NMSE
#   2. mapped_code 进 seed42 base decoder 的 NMSE
#   3. mapped_code 进 seed42 LoRA decoder 的 NMSE
#
# 用法示例：
#   mapper_exp_dir=staged_mlp_lora/exps/mapper/affine_mlp_h1024_b4_rs1.0_drop0.0_lr5e-4_ep400/seed2026_transnet_transnet_to_seed42 \
#   lora_exp_dir=staged_mlp_lora/exps/lora/identity_fc_ffn_fcr256a1024_ffnr16a64_rec_only_lr5e-4_eta1e-4_ep400/seed2026_transnet_transnet_mapped_h1024_b4_to_seed42 \
#   source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
#   gpu=0 \
#   bash staged_mlp_lora/scripts/test_staged_nmse.sh

set -euo pipefail

source_name=${source_name:-seed2026_transnet_transnet}
source_code=${source_code:-exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt}
target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}

mapper_exp_dir=${mapper_exp_dir:-staged_mlp_lora/exps/mapper/affine_mlp_h1024_b4_rs1.0_drop0.0_lr5e-4_ep400/${source_name}_to_seed42}
lora_exp_dir=${lora_exp_dir:-staged_mlp_lora/exps/lora/identity_fc_ffn_fcr256a1024_ffnr16a64_rec_only_lr5e-4_eta1e-4_ep400/${source_name}_mapped_h1024_b4_to_seed42}
mapper_checkpoint=${mapper_checkpoint:-${mapper_exp_dir}/checkpoints/best_mse.pth}
lora_checkpoint=${lora_checkpoint:-${lora_exp_dir}/checkpoints/best_nmse.pth}
result_name=${result_name:-${source_name}_staged_nmse}
output_json=${output_json:-${lora_exp_dir}/staged_nmse_${result_name}.json}
save_mapped_code=${save_mapped_code:-}

batch_size=${batch_size:-1024}
workers=${workers:-0}
max_samples=${max_samples:-0}
gpu=${gpu:-0}
cpu=${cpu:-0}

if [ ! -f "${source_code}" ]; then
  echo "Missing source_code: ${source_code}" >&2
  exit 1
fi
if [ ! -f "${target_code}" ]; then
  echo "Missing target_code: ${target_code}" >&2
  exit 1
fi
if [ ! -f "${csi_path}" ]; then
  echo "Missing csi_path: ${csi_path}" >&2
  exit 1
fi
if [ ! -f "${mapper_checkpoint}" ]; then
  echo "Missing mapper_checkpoint: ${mapper_checkpoint}" >&2
  exit 1
fi
if [ ! -f "${lora_checkpoint}" ]; then
  echo "Missing lora_checkpoint: ${lora_checkpoint}" >&2
  exit 1
fi

extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
else
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi
if [ -n "${save_mapped_code}" ]; then
  extra_args+=(--save_mapped_code "${save_mapped_code}")
fi

python -u staged_mlp_lora/test_staged_nmse.py \
  --source_code "${source_code}" \
  --target_code "${target_code}" \
  --csi_path "${csi_path}" \
  --mapper_checkpoint "${mapper_checkpoint}" \
  --lora_checkpoint "${lora_checkpoint}" \
  --decoder_checkpoint "${decoder_checkpoint}" \
  --decoder_args_json "${decoder_args_json}" \
  --output_json "${output_json}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --max_samples "${max_samples}" \
  --gpu "${gpu}" \
  "${extra_args[@]}"

echo "saved result: ${output_json}"
