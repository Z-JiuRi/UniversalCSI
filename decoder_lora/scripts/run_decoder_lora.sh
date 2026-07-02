#!/bin/bash

# 批量跑 affine code + fixed decoder LoRA 实验。
#
# 默认 source:
#   seed2026 transnet/clnet/crnet/csinet + seed3407 transnet
#
# 默认配置：
#   fc rank=8/16
#   fc_ffn rank=8/16
#
# 用法：
#   bash decoder_lora/scripts/run_decoder_lora.sh
#   gpus=0,4,6,7 overwrite=1 bash decoder_lora/scripts/run_decoder_lora.sh
#   sources=seed2026_transnet_transnet bash decoder_lora/scripts/run_decoder_lora.sh

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}

align_mode=${align_mode:-affine}
align_ridge=${align_ridge:-1e-4}
epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-0}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
scheduler=${scheduler:-cosine}
eta_min=${eta_min:-5e-5}
val_ratio=${val_ratio:-0}
max_samples=${max_samples:-0}
eval_decoder_every=${eval_decoder_every:-20}
eval_decoder_max_samples=${eval_decoder_max_samples:-0}
lambda_code=${lambda_code:-0.0}
lambda_recT=${lambda_recT:-0.0}
lambda_fc=${lambda_fc:-0.0}
gpus=${gpus:-0,4,6,7}
dry_run=${dry_run:-0}
overwrite=${overwrite:-0}
wait_existing=${wait_existing:-1}
wait_seconds=${wait_seconds:-600}
wait_by_source=${wait_by_source:-0}

IFS=',' read -r -a GPU_LIST <<< "${gpus}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "No GPUs configured: gpus=${gpus}" >&2
  exit 1
fi

if [ "${dry_run}" != "1" ] && [ "${wait_existing}" = "1" ]; then
  while pgrep -f "python -u decoder_lora/train_decoder_lora.py" > /dev/null; do
    echo "[wait] existing decoder_lora training process found; sleep ${wait_seconds}s"
    sleep "${wait_seconds}"
  done
fi

ALL_SOURCES=(
  "seed2026_transnet_transnet|exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt"
  "seed3407_transnet_transnet|exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt"
  "seed2026_clnet_transnet|exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt"
  "seed2026_crnet_transnet|exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt"
  "seed2026_csinet_transnet|exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt"
)

SOURCES=("${ALL_SOURCES[@]}")
if [ "${sources:-}" != "" ]; then
  SOURCES=()
  IFS=',' read -r -a SOURCE_FILTER <<< "${sources}"
  for wanted in "${SOURCE_FILTER[@]}"; do
    found=0
    for item in "${ALL_SOURCES[@]}"; do
      IFS='|' read -r name path <<< "${item}"
      if [ "${name}" = "${wanted}" ]; then
        SOURCES+=("${item}")
        found=1
      fi
    done
    if [ "${found}" = "0" ]; then
      echo "Unknown source filter: ${wanted}" >&2
      exit 1
    fi
  done
fi

# tag|lora_target|lora_rank|lambda_recT|lambda_fc|lambda_code
CONFIGS=(
  "fc_r8|fc|8|0.0|0.0|0.0"
  "fc_r16|fc|16|0.0|0.0|0.0"
  "fc_ffn_r8|fc_ffn|8|0.0|0.0|0.0"
  "fc_ffn_r16|fc_ffn|16|0.0|0.0|0.0"
)

task_id=0
running=0

launch_job() {
  gpu="${GPU_LIST[$((task_id % ${#GPU_LIST[@]}))]}"
  tag="align${align_mode}_${config_tag}_recT${cfg_recT}_fc${cfg_fc}_code${cfg_code}"
  exp_dir="decoder_lora/exps/${tag}/${source_name}_to_seed42_lr${lr}_ep${epochs}"

  if [ "${overwrite}" != "1" ] && [ -f "${exp_dir}/metrics.json" ]; then
    echo "[skip] ${exp_dir}"
    task_id=$((task_id + 1))
    return
  fi

  echo "[launch] gpu=${gpu} config=${config_tag} source=${source_name}"
  echo "         exp_dir=${exp_dir}"

  if [ "${dry_run}" = "1" ]; then
    task_id=$((task_id + 1))
    return
  fi

  source_name="${source_name}" \
  source_code="${source_code}" \
  target_code="${target_code}" \
  csi_path="${csi_path}" \
  decoder_checkpoint="${decoder_checkpoint}" \
  decoder_args_json="${decoder_args_json}" \
  exp_dir="${exp_dir}" \
  align_mode="${align_mode}" \
  align_ridge="${align_ridge}" \
  lora_target="${cfg_target}" \
  lora_rank="${cfg_rank}" \
  epochs="${epochs}" \
  batch_size="${batch_size}" \
  workers="${workers}" \
  lr="${lr}" \
  weight_decay="${weight_decay}" \
  scheduler="${scheduler}" \
  eta_min="${eta_min}" \
  val_ratio="${val_ratio}" \
  max_samples="${max_samples}" \
  eval_decoder_every="${eval_decoder_every}" \
  eval_decoder_max_samples="${eval_decoder_max_samples}" \
  lambda_recT="${cfg_recT}" \
  lambda_fc="${cfg_fc}" \
  lambda_code="${cfg_code}" \
  gpu="${gpu}" \
  bash decoder_lora/scripts/train_decoder_lora.sh > /dev/null 2>&1 &

  task_id=$((task_id + 1))
  running=$((running + 1))
  if [ "${running}" -ge "${#GPU_LIST[@]}" ]; then
    wait
    running=0
  fi
  sleep 5
}

if [ "${wait_by_source}" = "1" ]; then
  for source in "${SOURCES[@]}"; do
    IFS='|' read -r source_name source_code <<< "${source}"
    for config in "${CONFIGS[@]}"; do
      IFS='|' read -r config_tag cfg_target cfg_rank cfg_recT cfg_fc cfg_code <<< "${config}"
      launch_job
    done
    wait
    running=0
  done
else
  for config in "${CONFIGS[@]}"; do
    IFS='|' read -r config_tag cfg_target cfg_rank cfg_recT cfg_fc cfg_code <<< "${config}"
    for source in "${SOURCES[@]}"; do
      IFS='|' read -r source_name source_code <<< "${source}"
      launch_job
    done
  done
fi

wait
echo "All decoder_lora jobs finished."
