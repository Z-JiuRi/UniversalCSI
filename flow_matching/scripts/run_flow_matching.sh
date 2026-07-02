#!/bin/bash

# 批量启动 code-only flow matching 实验。
#
# 默认覆盖五个 source：
#   seed2026_transnet_transnet, seed3407_transnet_transnet,
#   seed2026_clnet_transnet, seed2026_crnet_transnet, seed2026_csinet_transnet
#
# 默认配置：
#   1. identity 起点：直接从原始 source code 流到 teacher code
#   2. procrustes 起点：先做正交 Procrustes，再学习残差流
#   3. affine 起点：先做全仿射，再学习残差流
#
# 用法：
#   bash flow_matching/scripts/run_flow_matching.sh
#   gpus="0 4 6 7" epochs=400 background=1 bash flow_matching/scripts/run_flow_matching.sh
#   lr=1e-4 eta_min=1e-5 bash flow_matching/scripts/run_flow_matching.sh

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
epochs=${epochs:-400}
batch_size=${batch_size:-512}
lr=${lr:-5e-4}
eta_min=${eta_min:-5e-5}
scheduler=${scheduler:-cosine}
hidden_dim=${hidden_dim:-2048}
num_blocks=${num_blocks:-4}
ode_steps=${ode_steps:-16}
condition=${condition:-source_start}
lambda_endpoint=${lambda_endpoint:-0.0}
background=${background:-1}
gpus=${gpus:-"0 1 4 6 7"}

SOURCES=(
  "seed2026_transnet_transnet exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt"
  "seed3407_transnet_transnet exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt"
  "seed2026_clnet_transnet exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt"
  "seed2026_crnet_transnet exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt"
  "seed2026_csinet_transnet exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt"
)

# ALIGN_MODES=(affine procrustes identity)
ALIGN_MODES=(affine)

read -r -a GPU_LIST <<< "${gpus}"
gpu_idx=0

for item in "${SOURCES[@]}"; do
  read -r source_name source_code <<< "${item}"
  for align_mode in "${ALIGN_MODES[@]}"; do
    gpu=${GPU_LIST[$((gpu_idx % ${#GPU_LIST[@]}))]}
    gpu_idx=$((gpu_idx + 1))
    source_name="${source_name}" \
    source_code="${source_code}" \
    target_code="${target_code}" \
    align_mode="${align_mode}" \
    condition="${condition}" \
    lambda_endpoint="${lambda_endpoint}" \
    epochs="${epochs}" \
    batch_size="${batch_size}" \
    lr="${lr}" \
    eta_min="${eta_min}" \
    scheduler="${scheduler}" \
    hidden_dim="${hidden_dim}" \
    num_blocks="${num_blocks}" \
    ode_steps="${ode_steps}" \
    gpu="${gpu}" \
    background="${background}" \
    bash flow_matching/scripts/train_flow_matching.sh
  done
done
