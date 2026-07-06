#!/bin/bash

# Stage 1/2：闭式 affine 粗对齐 + residual MLP 码字对齐。
#
# 数据流：
#   source_code -> fixed affine -> z0 -> residual MLP -> mapped_code
#
# 用法示例：
#   source_name=seed2026_transnet_transnet \
#   source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
#   hidden_dim=1024 num_blocks=4 gpu=0 \
#   bash staged_mlp_lora/scripts/train_mapper.sh

set -euo pipefail

source_name=${source_name:-seed2026_transnet_transnet}
source_code=${source_code:-exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt}
target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}

hidden_dim=${hidden_dim:-1024}
num_blocks=${num_blocks:-4}
dropout=${dropout:-0.0}
residual_scale=${residual_scale:-1.0}
final_norm=${final_norm:-0}
align_ridge=${align_ridge:-1e-4}
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
gpu=${gpu:-0}
seed=${seed:-2026}
cpu=${cpu:-0}

norm_tag="fn${final_norm}"
tag="affine_mlp_h${hidden_dim}_b${num_blocks}_rs${residual_scale}_${norm_tag}_drop${dropout}_lr${lr}_ep${epochs}"
exp_dir=${exp_dir:-staged_mlp_lora/exps/mapper/${tag}/${source_name}_to_seed42}

if [ ! -f "${source_code}" ]; then
  echo "Missing source_code: ${source_code}" >&2
  exit 1
fi
if [ ! -f "${target_code}" ]; then
  echo "Missing target_code: ${target_code}" >&2
  exit 1
fi
if [ "${eval_decoder_every}" != "0" ]; then
  if [ ! -f "${decoder_checkpoint}" ]; then
    echo "Missing decoder_checkpoint: ${decoder_checkpoint}" >&2
    exit 1
  fi
  if [ ! -f "${decoder_args_json}" ]; then
    echo "Missing decoder_args_json: ${decoder_args_json}" >&2
    exit 1
  fi
  if [ ! -f "${csi_path}" ]; then
    echo "Missing csi_path: ${csi_path}" >&2
    exit 1
  fi
fi

extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
else
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi
if [ "${final_norm}" = "0" ]; then
  extra_args+=(--no_final_norm)
fi

mkdir -p "${exp_dir}"

python -u staged_mlp_lora/train_affine_mlp_mapper.py \
  --source_code "${source_code}" \
  --target_code "${target_code}" \
  --exp_dir "${exp_dir}" \
  --source_name "${source_name}" \
  --hidden_dim "${hidden_dim}" \
  --num_blocks "${num_blocks}" \
  --dropout "${dropout}" \
  --residual_scale "${residual_scale}" \
  --align_ridge "${align_ridge}" \
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
  --decoder_checkpoint "${decoder_checkpoint}" \
  --decoder_args_json "${decoder_args_json}" \
  --csi_path "${csi_path}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &

echo "started mapper pid=$! exp_dir=${exp_dir}"
echo "mapped_code=${exp_dir}/codewords/mapped_code.pt"
