#!/bin/bash

# 批量测试 mapper/exps 下所有已保存 mapped_code.pt 在固定 decoder 下的 NMSE。
#
# 用法：
#   gpu=1 bash mapper/scripts/run_decoder_nmse_for_mapped_codes.sh
#
# 可选 teacher 上限对照：
#   include_teacher=1 gpu=1 bash mapper/scripts/run_decoder_nmse_for_mapped_codes.sh

set -euo pipefail

gpu=${gpu:-0}
batch_size=${batch_size:-1024}
workers=${workers:-0}
include_teacher=${include_teacher:-1}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}
data_path=${data_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}

if [ "${include_teacher}" = "1" ]; then
  code_path=exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt \
  result_name=teacher_code \
  decoder_checkpoint="${decoder_checkpoint}" \
  decoder_args_json="${decoder_args_json}" \
  data_path="${data_path}" \
  gpu="${gpu}" batch_size="${batch_size}" workers="${workers}" \
  bash mapper/scripts/test_decoder_nmse_from_code.sh
fi

while IFS= read -r code_path; do
  mapper_name=$(basename "$(dirname "$(dirname "${code_path}")")")
  exp_name=$(basename "$(dirname "${code_path}")")
  result_name="${mapper_name}_${exp_name}"
  echo "[test] ${result_name}"
  code_path="${code_path}" \
  result_name="${result_name}" \
  decoder_checkpoint="${decoder_checkpoint}" \
  decoder_args_json="${decoder_args_json}" \
  data_path="${data_path}" \
  gpu="${gpu}" batch_size="${batch_size}" workers="${workers}" \
  bash mapper/scripts/test_decoder_nmse_from_code.sh
done < <(find mapper/exps -path '*/mapped_code.pt' -type f | sort)
