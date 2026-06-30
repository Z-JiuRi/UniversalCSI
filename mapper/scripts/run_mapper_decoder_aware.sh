#!/bin/bash

# 批量跑第二阶段 decoder-aware mapper 实验。
# 在 code loss 基础上引入固定 seed42 decoder 的 recT / rec / fc / tail loss。
#
# 默认 fixed decoder:
#   exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth
#
# 默认 teacher code:
#   exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt
#
# GPU 调度：
#   默认在 0,4,6,7 四张卡上循环分配，每 4 个任务并发一批。
#
# 用法：
#   bash mapper/scripts/run_mapper_decoder_aware.sh
#
# 常用覆盖：
#   mapper=mlp epochs=400 gpus=0,4,6,7 bash mapper/scripts/run_mapper_decoder_aware.sh
#   mapper=hybrid_flow_mlp epochs=400 gpus=0,4,6,7 bash mapper/scripts/run_mapper_decoder_aware.sh
#   dry_run=1 bash mapper/scripts/run_mapper_decoder_aware.sh
#   overwrite=1 bash mapper/scripts/run_mapper_decoder_aware.sh

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}

epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
mapper=${mapper:-mlp}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
flow_blocks=${flow_blocks:-8}
flow_hidden_dim=${flow_hidden_dim:-1024}
flow_clamp=${flow_clamp:-0.1}
hidden_dim=${hidden_dim:-2048}
num_blocks=${num_blocks:-4}
dropout=${dropout:-0.0}
max_samples=${max_samples:-0}
val_ratio=${val_ratio:-0}
smoothl1_beta=${smoothl1_beta:-0.05}
whiten_eps_ratio=${whiten_eps_ratio:-1e-3}
decoder_tail_ratio=${decoder_tail_ratio:-0.2}
gpus=${gpus:-0,4,6,7}
dry_run=${dry_run:-0}
overwrite=${overwrite:-0}

IFS=',' read -r -a GPU_LIST <<< "${gpus}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "No GPUs configured: gpus=${gpus}" >&2
  exit 1
fi

SOURCES=(
  "seed2026_transnet_transnet|exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt"
  "seed3407_transnet_transnet|exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt"
  "seed2026_clnet_transnet|exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt"
  "seed2026_crnet_transnet|exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt"
  "seed2026_csinet_transnet|exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt"
)

# 字段：
# tag|lambda_smoothl1|lambda_sample_tail|lambda_dim_tail|lambda_whiten|lambda_rec|lambda_recT|lambda_fc|lambda_decoder_tail
CONFIGS=(
  "recT|0.0|0.0|0.0|0.0|0.0|1.0|0.0|0.0"
  "recT_rec|0.0|0.0|0.0|0.0|1.0|1.0|0.0|0.0"
  "recT_fc|0.0|0.0|0.0|0.0|0.0|1.0|1e-2|0.0"
  "recT_rec_fc|0.0|0.0|0.0|0.0|1.0|1.0|1e-2|0.0"
  "recT_rec_fc_tail|0.0|0.0|0.0|0.0|1.0|1.0|1e-2|0.1"
  "smooth_tail_white_recT_rec_fc|0.5|0.1|0.05|1e-4|1.0|1.0|1e-2|0.1"
)

task_id=0
running=0

for config in "${CONFIGS[@]}"; do
  IFS='|' read -r config_tag lambda_smoothl1 lambda_sample_tail lambda_dim_tail lambda_whiten lambda_rec lambda_recT lambda_fc lambda_decoder_tail <<< "${config}"
  for source in "${SOURCES[@]}"; do
    IFS='|' read -r source_name source_code <<< "${source}"
    gpu="${GPU_LIST[$((task_id % ${#GPU_LIST[@]}))]}"
    exp_dir="mapper/exps_decoder_aware/${mapper}/${config_tag}/${source_name}_to_seed42_transnet_lr${lr}_ep${epochs}"

    if [ "${overwrite}" != "1" ] && [ -f "${exp_dir}/metrics.json" ]; then
      echo "[skip] ${exp_dir}"
      task_id=$((task_id + 1))
      continue
    fi

    echo "[launch] gpu=${gpu} mapper=${mapper} config=${config_tag} source=${source_name}"
    echo "         exp_dir=${exp_dir}"

    if [ "${dry_run}" = "1" ]; then
      task_id=$((task_id + 1))
      continue
    fi

    source_code="${source_code}" \
    source_name="${source_name}" \
    target_code="${target_code}" \
    decoder_checkpoint="${decoder_checkpoint}" \
    decoder_args_json="${decoder_args_json}" \
    csi_path="${csi_path}" \
    exp_dir="${exp_dir}" \
    mapper="${mapper}" \
    epochs="${epochs}" \
    batch_size="${batch_size}" \
    workers="${workers}" \
    lr="${lr}" \
    weight_decay="${weight_decay}" \
    flow_blocks="${flow_blocks}" \
    flow_hidden_dim="${flow_hidden_dim}" \
    flow_clamp="${flow_clamp}" \
    hidden_dim="${hidden_dim}" \
    num_blocks="${num_blocks}" \
    dropout="${dropout}" \
    max_samples="${max_samples}" \
    val_ratio="${val_ratio}" \
    lambda_smoothl1="${lambda_smoothl1}" \
    smoothl1_beta="${smoothl1_beta}" \
    lambda_sample_tail="${lambda_sample_tail}" \
    sample_tail_ratio=0.2 \
    lambda_dim_tail="${lambda_dim_tail}" \
    dim_tail_ratio=0.05 \
    lambda_whiten="${lambda_whiten}" \
    whiten_eps_ratio="${whiten_eps_ratio}" \
    lambda_rec="${lambda_rec}" \
    lambda_recT="${lambda_recT}" \
    lambda_fc="${lambda_fc}" \
    lambda_decoder_tail="${lambda_decoder_tail}" \
    decoder_tail_ratio="${decoder_tail_ratio}" \
    gpu="${gpu}" \
    bash mapper/scripts/train_mapper.sh

    task_id=$((task_id + 1))
    running=$((running + 1))
    if [ "${running}" -ge "${#GPU_LIST[@]}" ]; then
      wait
      running=0
    fi
    sleep 5
  done
done

wait
echo "All decoder-aware mapper jobs finished."
