#!/bin/bash

# 用固定 decoder 测试已保存 codeword 的重建 NMSE。
#
# 默认测试当前 mapper 里完整跑完且效果最好的 hybrid mapped code：
#   mapper/exps/hybrid/seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400/mapped_code.pt
#
# 常用用法：
#   bash mapper/scripts/test_decoder_nmse_from_code.sh
#
# 换成 teacher code，得到固定 decoder 的上限对照：
#   code_path=exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt \
#   result_name=teacher_code \
#   bash mapper/scripts/test_decoder_nmse_from_code.sh
#
# 换成某个 mapper 实验：
#   code_path=mapper/exps/mlp/seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400/mapped_code.pt \
#   result_name=mlp_seed2026 \
#   gpu=1 bash mapper/scripts/test_decoder_nmse_from_code.sh

set -euo pipefail

code_path=${code_path:-mapper/exps/hybrid/seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400/mapped_code.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}
data_path=${data_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}

batch_size=${batch_size:-512}
workers=${workers:-0}
gpu=${gpu:-0}
cpu=${cpu:-0}
max_samples=${max_samples:-0}
result_name=${result_name:-$(basename "$(dirname "${code_path}")")}
output_json=${output_json:-mapper/reports/decoder_nmse/${result_name}.json}

if [ ! -f "${code_path}" ]; then
  echo "Missing code_path: ${code_path}" >&2
  exit 1
fi
if [ ! -f "${decoder_checkpoint}" ]; then
  echo "Missing decoder_checkpoint: ${decoder_checkpoint}" >&2
  exit 1
fi
if [ ! -f "${data_path}" ]; then
  echo "Missing data_path: ${data_path}" >&2
  exit 1
fi

extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
fi
if [ "${max_samples}" != "0" ]; then
  extra_args+=(--max_samples "${max_samples}")
fi

python mapper/test_decoder_nmse_from_code.py \
  --code_path "${code_path}" \
  --decoder_checkpoint "${decoder_checkpoint}" \
  --decoder_args_json "${decoder_args_json}" \
  --data_path "${data_path}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --gpu "${gpu}" \
  --output_json "${output_json}" \
  "${extra_args[@]}"

echo "saved: ${output_json}"
