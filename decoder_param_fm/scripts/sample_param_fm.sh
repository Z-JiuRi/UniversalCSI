#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${python_bin:-/home/hujiacong/zxd/.envs/miniconda3/envs/torch/bin/python}"
exp_dir="${exp_dir:-decoder_param_fm/exps/random_film_rms_tok512_h512_lr2e-4_ep400_seed42}"
checkpoint="${checkpoint:-${exp_dir}/checkpoints/best_loss.pth}"
guide_code_path="${guide_code_path:-}"
target_checkpoint="${target_checkpoint:-}"
ode_steps="${ode_steps:-16}"
gpu="${gpu:-0}"
max_guide_codes="${max_guide_codes:-0}"
output="${output:-${exp_dir}/generated/generated_decoder.pth}"

cmd=("$python_bin" -u decoder_param_fm/sample_param_fm.py
  --exp_dir "$exp_dir" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --ode_steps "$ode_steps" \
  --gpu "$gpu" \
  --max_guide_codes "$max_guide_codes")

if [[ -n "$guide_code_path" ]]; then
  cmd+=(--guide_code_path "$guide_code_path")
fi
if [[ -n "$target_checkpoint" ]]; then
  cmd+=(--target_checkpoint "$target_checkpoint")
fi

"${cmd[@]}"
