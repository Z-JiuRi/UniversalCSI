#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${python_bin:-/home/hujiacong/zxd/.envs/miniconda3/envs/torch/bin/python}"
target_exp="${target_exp:-exps/COST2100/in/seed42/transnet_transnet}"
decoder_args_json="${decoder_args_json:-${target_exp}/args.json}"
target_checkpoint="${target_checkpoint:-${target_exp}/checkpoints/best_nmse.pth}"
guide_code_path="${guide_code_path:-${target_exp}/codewords/train_code.pt}"
data_txt="${data_txt:-}"

condition_extract="${condition_extract:-random}"
condition_inject="${condition_inject:-film}"
param_norm="${param_norm:-rms}"
epochs="${epochs:-400}"
steps_per_epoch="${steps_per_epoch:-100}"
lr="${lr:-2e-4}"
warmup_ratio="${warmup_ratio:-0.1}"
token_size="${token_size:-512}"
hidden_dim="${hidden_dim:-512}"
num_blocks="${num_blocks:-4}"
condition_tokens="${condition_tokens:-512}"
cond_dim="${cond_dim:-512}"
num_heads="${num_heads:-8}"
set_layers="${set_layers:-2}"
lambda_endpoint="${lambda_endpoint:-1.0}"
base_seed="${base_seed:-2026}"
seed="${seed:-42}"
gpu="${gpu:-0}"
max_guide_codes="${max_guide_codes:-0}"

run_name="${run_name:-${condition_extract}_${condition_inject}_${param_norm}_tok${token_size}_h${hidden_dim}_lr${lr}_ep${epochs}_seed${seed}}"
exp_dir="${exp_dir:-decoder_param_fm/exps/${run_name}}"

cmd=("$python_bin" -u decoder_param_fm/train_param_fm.py
  --exp_dir "$exp_dir" \
  --decoder_args_json "$decoder_args_json" \
  --condition_extract "$condition_extract" \
  --condition_inject "$condition_inject" \
  --param_norm "$param_norm" \
  --epochs "$epochs" \
  --steps_per_epoch "$steps_per_epoch" \
  --lr "$lr" \
  --warmup_ratio "$warmup_ratio" \
  --token_size "$token_size" \
  --hidden_dim "$hidden_dim" \
  --num_blocks "$num_blocks" \
  --condition_tokens "$condition_tokens" \
  --cond_dim "$cond_dim" \
  --num_heads "$num_heads" \
  --set_layers "$set_layers" \
  --lambda_endpoint "$lambda_endpoint" \
  --base_seed "$base_seed" \
  --seed "$seed" \
  --gpu "$gpu" \
  --max_guide_codes "$max_guide_codes")

if [[ -n "$data_txt" ]]; then
  cmd+=(--data_txt "$data_txt")
else
  cmd+=(--target_checkpoint "$target_checkpoint")
  cmd+=(--guide_code_path "$guide_code_path")
fi

"${cmd[@]}" > /dev/null 2>&1 &
