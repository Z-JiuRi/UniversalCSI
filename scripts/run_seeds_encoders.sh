#!/bin/bash
# 批量训练多种 encoder × seed 组合。
# 如果对应实验已有码字文件，说明已跑完，跳过。
#
# 用法：
#   bash scripts/run_seeds_encoders.sh

# 所有encoder：
# csinet cnn cbam_cnn crnet clnet transnet resnet dscnn convnext mlp_mixer attention_cnn swin mlp_ae sparse_resnet

set -euo pipefail

# ========== 配置 ==========

ENCODERS=(
  csinet cnn cbam_cnn crnet clnet transnet resnet dscnn
  convnext mlp_mixer attention_cnn swin mlp_ae sparse_resnet
)

SEEDS=(287 314)

# ========== run_one ==========

run_one_exp() {
  local encoder=$1
  local seed=$2
  local gpu=$3

  local exp_dir="exps/COST2100/in/seed${seed}/${encoder}_transnet"

  if [ -d "$exp_dir" ]; then
    echo "[SKIP] seed=${seed} encoder=${encoder} — directory exists"
    return 0
  fi

  echo "[RUN]   seed=${seed} encoder=${encoder} gpu=${gpu}"
  encoder="${encoder}" \
  decoder="transnet" \
  seed="${seed}" \
  gpu="${gpu}" \
  exp_name="COST2100/in/seed${seed}/${encoder}_transnet" \
  bash scripts/train.sh
}


for encoder in "${ENCODERS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one_exp "$encoder" "$seed" 0
    sleep 5
  done
done