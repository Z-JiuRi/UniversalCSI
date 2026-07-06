#!/bin/bash

# Stage 3：在已经对齐好的 mapped code 上，只训练 fixed seed42 decoder 的 LoRA。
#
# 数据流：
#   mapped_code -> fixed seed42 decoder + LoRA(fc_decoder + FFN) -> CSI
#
# 注意：
#   这里默认 align_mode=identity，因为 Stage 1/2 已经输出 mapped_code。
#   默认 code_adapter=none，并且 lambda_code/lambda_delta/lambda_recT/lambda_fc 都为 0，
#   因此优化目标就是 MSE(reconstruction, raw CSI)。
#
# 用法示例：
#   mapper_exp_dir=staged_mlp_lora/exps/mapper/affine_mlp_h1024_b4_rs1.0_drop0.0_lr5e-4_ep400/seed2026_transnet_transnet_to_seed42 \
#   source_name=seed2026_transnet_transnet_h1024_b4 \
#   fc_lora_rank=256 fc_lora_alpha=1024 ffn_lora_rank=16 ffn_lora_alpha=64 gpu=4 \
#   bash staged_mlp_lora/scripts/train_lora.sh

set -euo pipefail

mapper_exp_dir=${mapper_exp_dir:-staged_mlp_lora/exps/mapper/affine_mlp_h1024_b4_rs1.0_drop0.0_lr5e-4_ep400/seed2026_transnet_transnet_to_seed42}
source_code=${source_code:-${mapper_exp_dir}/codewords/mapped_code.pt}
source_name=${source_name:-seed2026_transnet_transnet_mapped}
target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}

align_mode=${align_mode:-identity}
align_ridge=${align_ridge:-1e-4}
lora_target=${lora_target:-fc_ffn}
lora_rank=${lora_rank:-8}
fc_lora_rank=${fc_lora_rank:-256}
ffn_lora_rank=${ffn_lora_rank:-16}
lora_alpha=${lora_alpha:-}
fc_lora_alpha=${fc_lora_alpha:-1024}
ffn_lora_alpha=${ffn_lora_alpha:-64}
lora_dropout=${lora_dropout:-0.0}
code_adapter=${code_adapter:-none}
code_lowrank_rank=${code_lowrank_rank:-0}
code_mlp_hidden=${code_mlp_hidden:-0}
code_gate_lr=${code_gate_lr:-0.1}
code_gate_mlp=${code_gate_mlp:-0.1}
code_adapter_dropout=${code_adapter_dropout:-0.0}

epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-0}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
scheduler=${scheduler:-cosine}
eta_min=${eta_min:-1e-4}
val_ratio=${val_ratio:-0}
max_samples=${max_samples:-0}
eval_decoder_every=${eval_decoder_every:-20}
eval_decoder_max_samples=${eval_decoder_max_samples:-0}
lambda_code=${lambda_code:-0.0}
lambda_delta=${lambda_delta:-0.0}
lambda_recT=${lambda_recT:-0.0}
lambda_fc=${lambda_fc:-0.0}
save_last=${save_last:-0}
gpu=${gpu:-0}
seed=${seed:-2026}
cpu=${cpu:-0}

rank_tag="fcr${fc_lora_rank}a${fc_lora_alpha}_ffnr${ffn_lora_rank}a${ffn_lora_alpha}"
tag="identity_${lora_target}_${rank_tag}_rec_only_lr${lr}_eta${eta_min}_ep${epochs}"
exp_dir=${exp_dir:-staged_mlp_lora/exps/lora/${tag}/${source_name}_to_seed42}

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
if [ ! -f "${decoder_checkpoint}" ]; then
  echo "Missing decoder_checkpoint: ${decoder_checkpoint}" >&2
  exit 1
fi
if [ ! -f "${decoder_args_json}" ]; then
  echo "Missing decoder_args_json: ${decoder_args_json}" >&2
  exit 1
fi

extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
else
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi
if [ "${save_last}" = "1" ]; then
  extra_args+=(--save_last)
fi
if [ -n "${lora_alpha}" ]; then
  extra_args+=(--lora_alpha "${lora_alpha}")
fi

mkdir -p "${exp_dir}"

python -u decoder_lora/train_decoder_lora.py \
  --source_code "${source_code}" \
  --target_code "${target_code}" \
  --csi_path "${csi_path}" \
  --decoder_checkpoint "${decoder_checkpoint}" \
  --decoder_args_json "${decoder_args_json}" \
  --exp_dir "${exp_dir}" \
  --source_name "${source_name}" \
  --align_mode "${align_mode}" \
  --align_ridge "${align_ridge}" \
  --lora_target "${lora_target}" \
  --lora_rank "${lora_rank}" \
  --fc_lora_rank "${fc_lora_rank}" \
  --ffn_lora_rank "${ffn_lora_rank}" \
  --fc_lora_alpha "${fc_lora_alpha}" \
  --ffn_lora_alpha "${ffn_lora_alpha}" \
  --lora_dropout "${lora_dropout}" \
  --code_adapter "${code_adapter}" \
  --code_lowrank_rank "${code_lowrank_rank}" \
  --code_mlp_hidden "${code_mlp_hidden}" \
  --code_gate_lr "${code_gate_lr}" \
  --code_gate_mlp "${code_gate_mlp}" \
  --code_adapter_dropout "${code_adapter_dropout}" \
  --epochs "${epochs}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --scheduler "${scheduler}" \
  --eta_min "${eta_min}" \
  --val_ratio "${val_ratio}" \
  --max_samples "${max_samples}" \
  --eval_decoder_every "${eval_decoder_every}" \
  --eval_decoder_max_samples "${eval_decoder_max_samples}" \
  --lambda_code "${lambda_code}" \
  --lambda_delta "${lambda_delta}" \
  --lambda_recT "${lambda_recT}" \
  --lambda_fc "${lambda_fc}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &

echo "started lora pid=$! exp_dir=${exp_dir}"
