#!/bin/bash

# 从 flow_matching checkpoint 重新导出 mapped_code.pt。
#
# 用法：
#   exp_dir=flow_matching/exps/... bash flow_matching/scripts/export_mapped_code.sh
#   exp_dir=flow_matching/exps/... checkpoint=flow_matching/exps/.../checkpoints/best_loss.pth gpu=4 bash flow_matching/scripts/export_mapped_code.sh

set -euo pipefail

exp_dir=${exp_dir:?需要指定 exp_dir}
checkpoint=${checkpoint:-}
batch_size=${batch_size:-2048}
workers=${workers:-0}
gpu=${gpu:-0}
cpu=${cpu:-0}
ode_steps=${ode_steps:-}
ode_method=${ode_method:-}

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

python -u flow_matching/export_mapped_code.py \
  --exp_dir "${exp_dir}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --gpu "${gpu}" \
  "${extra_args[@]}"

