#!/bin/bash
# 批量训练多种 encoder × seed 组合。
# 检测该 seed 下有没有 transnet_transnet 的码字：
#   - 有 → 说明 transnet decoder + 该 seed 的联合训练已跑完，跳过该 seed 所有 encoder
#   - 无 → 跑该 seed 下所有 encoder，但每次最多跑 10 个实验
#
# 用法：
#   bash scripts/run_seeds_encoders2.sh

# 所有encoder：
# csinet cnn cbam_cnn crnet clnet transnet resnet dscnn convnext mlp_mixer attention_cnn swin mlp_ae sparse_resnet

set -euo pipefail

# ========== 配置 ==========

ENCODERS=(transnet)

# 自动扫描 exps/COST2100/in/ 下所有 seedxxx 目录
SEEDS=()
for d in exps/COST2100/in/seed*/; do
  seed="${d#exps/COST2100/in/seed}"
  seed="${seed%/}"
  SEEDS+=("$seed")
done

MAX_RUNS=30          # 每次执行最多跑多少个实验
RUN_COUNT=0          # 已启动实验计数

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


for seed in "${SEEDS[@]}"; do
  # 检测该 seed 下有没有 transnet_transnet 目录
  ref_dir="exps/COST2100/in/seed${seed}/transnet_transnet"

  if [ -d "$ref_dir" ]; then
    echo "[SKIP ALL] seed=${seed} — transnet_transnet direxists, skipping all encoders"
    continue
  fi

  echo "--- seed=${seed} has no transnet_transnet, will run encoders ---"

  for encoder in "${ENCODERS[@]}"; do
    if [ "$RUN_COUNT" -ge "$MAX_RUNS" ]; then
      echo "[BATCH LIMIT] Reached ${MAX_RUNS} runs, stopping."
      exit 0
    fi

    run_one_exp "$encoder" "$seed" 4
    RUN_COUNT=$((RUN_COUNT + 1))
    sleep 5
  done
done

echo "[DONE] All seeds processed. Total runs: ${RUN_COUNT}"