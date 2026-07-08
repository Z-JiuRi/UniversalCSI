#!/bin/bash

# 批量评估一个训练好的 decoder_generalization_fm 实验。
#
# 用法：
#   exp_dir=decoder_generalization_fm/exps/smoke gpu=4 bash decoder_generalization_fm/scripts/evaluate.sh

set -euo pipefail

exp_dir=${exp_dir:-decoder_generalization_fm/exps/generalization_fm}
checkpoint=${checkpoint:-${exp_dir}/checkpoints/best_loss.pth}
data_txt=${data_txt:-}
split=${split:-all}
ode_steps=${ode_steps:-16}
sample_seed=${sample_seed:-0}
batch_size=${batch_size:-1024}
max_samples=${max_samples:-0}
max_entries=${max_entries:-0}
output_json=${output_json:-${exp_dir}/generated/eval_${split}.json}
gpu=${gpu:-0}
cpu=${cpu:-0}

if [ ! -f "${checkpoint}" ]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 1
fi

extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
fi
if [ -n "${data_txt}" ]; then
  extra_args+=(--data_txt "${data_txt}")
fi

python -u decoder_generalization_fm/evaluate.py \
  --exp_dir "${exp_dir}" \
  --checkpoint "${checkpoint}" \
  --split "${split}" \
  --ode_steps "${ode_steps}" \
  --sample_seed "${sample_seed}" \
  --batch_size "${batch_size}" \
  --max_samples "${max_samples}" \
  --max_entries "${max_entries}" \
  --output_json "${output_json}" \
  --gpu "${gpu}" \
  "${extra_args[@]}"

