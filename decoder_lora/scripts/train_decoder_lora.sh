#!/bin/bash

# 训练 fixed seed42 decoder 的 LoRA 适配器。
#
# 数据流：
#   source code -> closed-form affine/procrustes -> aligned code z0
#   z0 -> frozen seed42 decoder + LoRA(fc_decoder + FFN) -> CSI
#
# 常用命令：
#   source_name=seed2026_transnet_transnet \
#   source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
#   lora_target=fc_ffn lora_rank=8 gpu=0 bash decoder_lora/scripts/train_decoder_lora.sh
#
# 可调项：
#   align_mode=identity|procrustes|affine
#   lora_target=fc|ffn|fc_ffn
#   lora_rank=8|16|32
#   lambda_recT=0.0 lambda_fc=0.0 lambda_code=0.0

set -euo pipefail

source_name=${source_name:-seed2026_transnet_transnet}
source_code=${source_code:-exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt}
target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}

align_mode=${align_mode:-affine}
align_ridge=${align_ridge:-1e-4}
lora_target=${lora_target:-fc_ffn}
lora_rank=${lora_rank:-8}
lora_alpha=${lora_alpha:-}
lora_dropout=${lora_dropout:-0.0}
epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-0}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
scheduler=${scheduler:-cosine}
eta_min=${eta_min:-5e-5}
val_ratio=${val_ratio:-0}
max_samples=${max_samples:-0}
eval_decoder_every=${eval_decoder_every:-20}
eval_decoder_max_samples=${eval_decoder_max_samples:-0}
lambda_code=${lambda_code:-0.0}
lambda_recT=${lambda_recT:-0.0}
lambda_fc=${lambda_fc:-0.0}
save_last=${save_last:-0}
gpu=${gpu:-0}
seed=${seed:-2026}
cpu=${cpu:-0}

tag="align${align_mode}_${lora_target}_r${lora_rank}_recT${lambda_recT}_fc${lambda_fc}_code${lambda_code}"
exp_dir=${exp_dir:-decoder_lora/exps/${tag}/${source_name}_to_seed42_lr${lr}_eta_${eta_min}_ep${epochs}}

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
  --lora_dropout "${lora_dropout}" \
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
  --lambda_recT "${lambda_recT}" \
  --lambda_fc "${lambda_fc}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &
