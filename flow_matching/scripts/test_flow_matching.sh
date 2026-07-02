#!/bin/bash

# 测试 flow_matching 实验的 code-only 映射指标。
#
# 用法：
#   exp_dir=flow_matching/exps/... bash flow_matching/scripts/test_flow_matching.sh
#   exp_dir=flow_matching/exps/... checkpoint=flow_matching/exps/.../checkpoints/best_loss.pth gpu=4 bash flow_matching/scripts/test_flow_matching.sh

set -euo pipefail

exp_dir=${exp_dir:?需要指定 exp_dir}
checkpoint=${checkpoint:-}
batch_size=${batch_size:-2048}
workers=${workers:-0}
gpu=${gpu:-0}
cpu=${cpu:-0}
ode_steps=${ode_steps:-}
ode_method=${ode_method:-}
output_json=${output_json:-${exp_dir}/test_metrics.json}

extra_args=()
if [ -n "${checkpoint}" ]; then
  extra_args+=(--checkpoint "${checkpoint}")
fi
if [ -n "${ode_steps}" ]; then
  extra_args+=(--ode_steps "${ode_steps}")
fi
if [ -n "${ode_method}" ]; then
  extra_args+=(--ode_method "${ode_method}")
fi
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
fi

python -u flow_matching/test_flow_matching.py \
  --exp_dir "${exp_dir}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --gpu "${gpu}" \
  --output_json "${output_json}" \
  "${extra_args[@]}"

