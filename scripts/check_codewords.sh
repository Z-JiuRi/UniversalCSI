#!/bin/bash
# 自动检测 exps/COST2100/in/seed*/ 下所有的实验目录，
# 检查 codewords/train_code.pt 是否存在。
#
# 用法：
#   bash scripts/check_codewords.sh

set -euo pipefail

found=0
missing=0

for seed_dir in exps/COST2100/in/seed*/; do
  seed=$(basename "$seed_dir")
  for exp_dir in "${seed_dir}"*_transnet/; do
    exp=$(basename "$exp_dir")
    path="${exp_dir}codewords/train_code.pt"
    if [ -f "$path" ]; then
      echo "${exp_dir}"
      found=$((found + 1))
    else
      missing=$((missing + 1))
      # echo "[MISS] ${seed}/${exp}"
    fi
  done
done

echo "=== Found: $found  Missing: $missing ==="
