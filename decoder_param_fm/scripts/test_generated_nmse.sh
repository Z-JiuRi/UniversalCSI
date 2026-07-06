#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${python_bin:-/home/hujiacong/zxd/.envs/miniconda3/envs/torch/bin/python}"
target_exp="${target_exp:-exps/COST2100/in/seed42/transnet_transnet}"
decoder_args_json="${decoder_args_json:-${target_exp}/args.json}"
code_path="${code_path:-${target_exp}/codewords/train_code.pt}"
csi_path="${csi_path:-}"
exp_dir="${exp_dir:-decoder_param_fm/exps/random_film_rms_tok512_h512_lr2e-4_ep400_seed42}"
decoder_state="${decoder_state:-${exp_dir}/generated/generated_decoder.pth}"
batch_size="${batch_size:-1024}"
max_samples="${max_samples:-0}"
gpu="${gpu:-0}"
output_json="${output_json:-${exp_dir}/generated/nmse.json}"

cmd=("$python_bin" -u decoder_param_fm/test_generated_nmse.py
  --decoder_state "$decoder_state"
  --decoder_args_json "$decoder_args_json"
  --code_path "$code_path"
  --batch_size "$batch_size"
  --max_samples "$max_samples"
  --gpu "$gpu"
  --output_json "$output_json")

if [[ -n "$csi_path" ]]; then
  cmd+=(--csi_path "$csi_path")
fi

"${cmd[@]}"
