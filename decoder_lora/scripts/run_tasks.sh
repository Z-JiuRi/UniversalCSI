#!/bin/bash

# Sweep only-LoRA decoder adaptation configs for rank/alpha tradeoff.
#
# It launches decoder_lora/scripts/train_decoder_lora.sh once per task.
# That script starts the actual Python process in the background, so this
# wrapper captures the child pid, launches all tasks, waits for them, then
# summarizes LoRA parameter count and final decoder NMSE from run.log.
#
# Usage:
#   bash decoder_lora/scripts/run_tasks.sh
#
# Useful overrides:
#   source_seed=2026 gpu_list=1,4,6,7 epochs=400 \
#     bash decoder_lora/scripts/run_tasks.sh
#   dry_run=1 bash decoder_lora/scripts/run_tasks.sh
#
# Task format:
#   label:fc_rank:fc_alpha:ffn_rank:ffn_alpha
#
# Label convention:
#   sN means LoRA scaling alpha/rank=N, not raw alpha=N.

set -euo pipefail

source_seed=${source_seed:-2026}
encoder=${encoder:-transnet}
target_seed=${target_seed:-42}
gpu_list=${gpu_list:-0,1,4,6,7}
dry_run=${dry_run:-0}
overwrite=${overwrite:-0}

align_mode=${align_mode:-affine}
lora_target=${lora_target:-fc_ffn}
code_adapter=${code_adapter:-none}
lambda_code=${lambda_code:-0}
lambda_delta=${lambda_delta:-0}
lambda_recT=${lambda_recT:-0}
lambda_fc=${lambda_fc:-0}
lr=${lr:-5e-4}
eta_min=${eta_min:-2e-4}
epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-0}
eval_decoder_every=${eval_decoder_every:-20}
eval_decoder_max_samples=${eval_decoder_max_samples:-0}

source_name=${source_name:-seed${source_seed}_${encoder}_transnet}
source_code=${source_code:-exps/COST2100/in/base/seed${source_seed}/${encoder}_transnet/codewords/train_code.pt}
target_code=${target_code:-exps/COST2100/in/base/seed${target_seed}/transnet_transnet/codewords/train_code.pt}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/base/seed${target_seed}/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/base/seed${target_seed}/transnet_transnet/args.json}

sweep_root=${sweep_root:-decoder_lora/exps/lora_rank_alpha_sweep}
summary_path=${summary_path:-${sweep_root}/summary.tsv}

IFS=',' read -r -a gpus <<< "${gpu_list}"
if [ "${#gpus[@]}" -eq 0 ]; then
  echo "gpu_list is empty" >&2
  exit 1
fi

TASKS=${TASKS:-"
fc32_s8_ffn4_s8:32:256:4:32
fc64_s8_ffn8_s8:64:512:8:64
fc128_s8_ffn16_s8:128:1024:16:128
fc256_s8_ffn16_s8:256:2048:16:128
fc256_s8_ffn32_s8:256:2048:32:256
fc128_s2_ffn16_s2:128:256:16:32
fc128_s4_ffn16_s4:128:512:16:64
fc128_s16_ffn16_s16:128:2048:16:256
fc64_s4_ffn8_s4:64:256:8:32
fc64_s16_ffn8_s16:64:1024:8:128
"}

mkdir -p "${sweep_root}"

estimate_lora_params() {
  local fc_rank=$1
  local ffn_rank=$2
  # fc_decoder: rank * (512 + 2048)
  # decoder FFN: 2 decoder layers * [(64 + 2048) + (2048 + 64)] * rank
  echo $((2560 * fc_rank + 8448 * ffn_rank))
}

active_pids=()

summarize_one() {
  local label=$1
  local exp_dir=$2
  local fc_rank=$3
  local fc_alpha=$4
  local ffn_rank=$5
  local ffn_alpha=$6
  local log_path="${exp_dir}/run.log"
  local status="missing_log"
  local lora_params=""
  local best_loss_nmse=""
  local best_loss_epoch=""
  local best_nmse=""
  local best_nmse_epoch=""

  if [ -f "${log_path}" ]; then
    status="done"
    lora_params=$(
      grep -E "=> Parameters: trainable=.* lora=" "${log_path}" \
        | tail -1 \
        | sed -E 's/.* lora=([0-9,]+).*/\1/' \
        | tr -d ','
    )
    best_loss_nmse=$(
      grep -E "all_best_loss_decoder_nmse=" "${log_path}" \
        | tail -1 \
        | sed -E 's/.*all_best_loss_decoder_nmse=([-0-9.]+)dB.*/\1/'
    )
    best_loss_epoch=$(
      grep -E "all_best_loss_decoder_nmse=" "${log_path}" \
        | tail -1 \
        | sed -E 's/.* epoch=([0-9]+).*/\1/'
    )
    best_nmse=$(
      grep -E "all_best_nmse_decoder_nmse=" "${log_path}" \
        | tail -1 \
        | sed -E 's/.*all_best_nmse_decoder_nmse=([-0-9.]+)dB.*/\1/'
    )
    best_nmse_epoch=$(
      grep -E "all_best_nmse_decoder_nmse=" "${log_path}" \
        | tail -1 \
        | sed -E 's/.* epoch=([0-9]+).*/\1/'
    )
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${label}" "${status}" "${fc_rank}" "${fc_alpha}" \
    "${ffn_rank}" "${ffn_alpha}" "$(estimate_lora_params "${fc_rank}" "${ffn_rank}")" \
    "${lora_params}" "${best_loss_nmse}" "${best_loss_epoch}" \
    "${best_nmse}" "${best_nmse_epoch}"
}

echo "source_code=${source_code}"
echo "target_code=${target_code}"
echo "decoder_checkpoint=${decoder_checkpoint}"
echo "sweep_root=${sweep_root}"
echo "gpu_list=${gpu_list}"

for required in "${source_code}" "${target_code}" "${csi_path}" \
  "${decoder_checkpoint}" "${decoder_args_json}"; do
  if [ ! -f "${required}" ]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

task_index=0
while IFS= read -r task; do
  task=${task#"${task%%[![:space:]]*}"}
  task=${task%"${task##*[![:space:]]}"}
  if [ -z "${task}" ] || [[ "${task}" == \#* ]]; then
    continue
  fi

  IFS=':' read -r label fc_rank fc_alpha ffn_rank ffn_alpha <<< "${task}"
  if [ -z "${label}" ] || [ -z "${fc_rank}" ] || [ -z "${fc_alpha}" ] \
    || [ -z "${ffn_rank}" ] || [ -z "${ffn_alpha}" ]; then
    echo "Bad task entry: ${task}" >&2
    exit 1
  fi

  exp_dir="${sweep_root}/${source_name}_to_seed${target_seed}/${label}_lr${lr}_eta${eta_min}_ep${epochs}"
  if [ "${overwrite}" != "1" ] && [ -f "${exp_dir}/history.json" ]; then
    echo "[skip] ${label}: ${exp_dir}/history.json exists"
    continue
  fi

  gpu="${gpus[$((task_index % ${#gpus[@]}))]}"
  task_index=$((task_index + 1))

  echo "[task] label=${label} gpu=${gpu} fc=${fc_rank}/${fc_alpha} ffn=${ffn_rank}/${ffn_alpha}"
  echo "       exp_dir=${exp_dir}"

  if [ "${dry_run}" = "1" ]; then
    continue
  fi

  launch_output=$(
    source_name="${source_name}" \
    source_code="${source_code}" \
    target_code="${target_code}" \
    csi_path="${csi_path}" \
    decoder_checkpoint="${decoder_checkpoint}" \
    decoder_args_json="${decoder_args_json}" \
    align_mode="${align_mode}" \
    lora_target="${lora_target}" \
    fc_lora_rank="${fc_rank}" \
    fc_lora_alpha="${fc_alpha}" \
    ffn_lora_rank="${ffn_rank}" \
    ffn_lora_alpha="${ffn_alpha}" \
    code_adapter="${code_adapter}" \
    lambda_code="${lambda_code}" \
    lambda_delta="${lambda_delta}" \
    lambda_recT="${lambda_recT}" \
    lambda_fc="${lambda_fc}" \
    gpu="${gpu}" \
    lr="${lr}" \
    eta_min="${eta_min}" \
    epochs="${epochs}" \
    batch_size="${batch_size}" \
    workers="${workers}" \
    eval_decoder_every="${eval_decoder_every}" \
    eval_decoder_max_samples="${eval_decoder_max_samples}" \
    exp_dir="${exp_dir}" \
    seed="${source_seed}" \
    bash decoder_lora/scripts/train_decoder_lora.sh
  )
  echo "${launch_output}"
  pid=$(echo "${launch_output}" | sed -nE 's/.*started pid=([0-9]+).*/\1/p' | tail -1)
  if [ -z "${pid}" ]; then
    echo "Failed to parse pid from launch output: ${launch_output}" >&2
    exit 1
  fi
  active_pids+=("${pid}")
done <<< "${TASKS}"

if [ "${dry_run}" != "1" ]; then
  for pid in "${active_pids[@]}"; do
    wait "${pid}"
  done
fi

{
  printf "label\tstatus\tfc_rank\tfc_alpha\tffn_rank\tffn_alpha\test_lora_params\tlog_lora_params\tbest_loss_nmse_db\tbest_loss_epoch\tbest_nmse_db\tbest_nmse_epoch\n"
  while IFS= read -r task; do
    task=${task#"${task%%[![:space:]]*}"}
    task=${task%"${task##*[![:space:]]}"}
    if [ -z "${task}" ] || [[ "${task}" == \#* ]]; then
      continue
    fi
    IFS=':' read -r label fc_rank fc_alpha ffn_rank ffn_alpha <<< "${task}"
    exp_dir="${sweep_root}/${source_name}_to_seed${target_seed}/${label}_lr${lr}_eta${eta_min}_ep${epochs}"
    summarize_one "${label}" "${exp_dir}" "${fc_rank}" "${fc_alpha}" "${ffn_rank}" "${ffn_alpha}"
  done <<< "${TASKS}"
} > "${summary_path}"

echo "summary saved to ${summary_path}"
