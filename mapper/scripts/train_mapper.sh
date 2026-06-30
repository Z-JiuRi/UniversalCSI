#!/bin/bash

# 训练 codeword mapper：只做 z_source -> z_teacher 的码字 MSE 映射。
#
# 默认 teacher:
#   exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt
#
# 默认 source:
#   exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt
#
# 用法：
#   bash mapper/scripts/train_mapper.sh
#
# 常用覆盖：
#   mapper=flow source_name=seed2026_transnet gpu=1 bash mapper/scripts/train_mapper.sh
#   mapper=hybrid_flow_mlp epochs=400 gpu=1 bash mapper/scripts/train_mapper.sh
#   lambda_cos=1e-3 lambda_cov=1e-5 bash mapper/scripts/train_mapper.sh

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
source_code=${source_code:-exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt}
source_name=${source_name:-seed2026_transnet_transnet}

mapper=${mapper:-flow}
epochs=${epochs:-400}
batch_size=${batch_size:-512}
workers=${workers:-0}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
hidden_dim=${hidden_dim:-2048}
num_blocks=${num_blocks:-4}
flow_hidden_dim=${flow_hidden_dim:-1024}
flow_blocks=${flow_blocks:-8}
flow_clamp=${flow_clamp:-0.1}
dropout=${dropout:-0.0}
val_ratio=${val_ratio:-0.1}
max_samples=${max_samples:-0}
lambda_cos=${lambda_cos:-0.0}
lambda_cov=${lambda_cov:-0.0}
gpu=${gpu:-0}
seed=${seed:-2026}
cpu=${cpu:-0}

exp_dir=${exp_dir:-mapper/exps/${mapper}/${source_name}_to_seed42_transnet_code_mse${lambda_cos}_cov${lambda_cov}_lr${lr}_ep${epochs}}

mkdir -p "${exp_dir}"

if [ ! -f "${source_code}" ]; then
  echo "Missing source_code: ${source_code}" >&2
  exit 1
fi
if [ ! -f "${target_code}" ]; then
  echo "Missing target_code: ${target_code}" >&2
  exit 1
fi

extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
fi

python -u mapper/train_mapper.py \
  --source_code "${source_code}" \
  --target_code "${target_code}" \
  --exp_dir "${exp_dir}" \
  --mapper "${mapper}" \
  --epochs "${epochs}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --hidden_dim "${hidden_dim}" \
  --num_blocks "${num_blocks}" \
  --flow_hidden_dim "${flow_hidden_dim}" \
  --flow_blocks "${flow_blocks}" \
  --flow_clamp "${flow_clamp}" \
  --dropout "${dropout}" \
  --val_ratio "${val_ratio}" \
  --max_samples "${max_samples}" \
  --lambda_cos "${lambda_cos}" \
  --lambda_cov "${lambda_cov}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  "${extra_args[@]}" \
  2>&1 | tee "${exp_dir}/run.log"
