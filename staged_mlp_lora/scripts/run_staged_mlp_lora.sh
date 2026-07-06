#!/bin/bash

# 三阶段实验：
#   1. 闭式 affine 先把 source code 粗对齐到 seed42 teacher code。
#   2. residual MLP 只用 MSE(mapped_code, teacher_code) 继续对齐码字。
#   3. mapped_code 固定不变，只训练 seed42 decoder 上的 LoRA，loss 只用 MSE(reconstruction, raw CSI)。
#
# 运行方式：
#   bash staged_mlp_lora/scripts/run_staged_mlp_lora.sh
#
# 脚本会先并行启动所有 mapper 实验，然后等待每个 mapped_code.pt 生成，再启动对应 LoRA 实验。
# GPU 分配可通过 gpus 覆盖，例如：
#   gpus="0 4 6 7" bash staged_mlp_lora/scripts/run_staged_mlp_lora.sh

set -euo pipefail

gpus=(${gpus:-0 4 6 7})
poll_seconds=${poll_seconds:-120}

declare -a mapper_specs=(
  "seed2026_transnet_transnet exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt 512 4 5e-4 400"
  "seed2026_transnet_transnet exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt 1024 4 5e-4 400"
  "seed2026_clnet_transnet exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt 512 4 5e-4 400"
  "seed2026_clnet_transnet exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt 1024 4 5e-4 400"
  "seed2026_crnet_transnet exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt 1024 4 5e-4 400"
  "seed2026_csinet_transnet exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt 1024 4 5e-4 400"
  "seed3407_transnet_transnet exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt 1024 4 5e-4 400"
)

mapper_exp_dir() {
  local source_name="$1"
  local hidden_dim="$2"
  local num_blocks="$3"
  local lr="$4"
  local epochs="$5"
  echo "staged_mlp_lora/exps/mapper/affine_mlp_h${hidden_dim}_b${num_blocks}_rs1.0_drop0.0_lr${lr}_ep${epochs}/${source_name}_to_seed42"
}

gpu_at() {
  local idx="$1"
  echo "${gpus[$((idx % ${#gpus[@]}))]}"
}

echo "== Stage 1/2: start affine + MLP mapper jobs =="
for i in "${!mapper_specs[@]}"; do
  read -r source_name source_code hidden_dim num_blocks lr epochs <<< "${mapper_specs[$i]}"
  gpu=$(gpu_at "${i}")
  source_name="${source_name}" \
    source_code="${source_code}" \
    hidden_dim="${hidden_dim}" \
    num_blocks="${num_blocks}" \
    lr="${lr}" \
    epochs="${epochs}" \
    gpu="${gpu}" \
    bash staged_mlp_lora/scripts/train_mapper.sh
done

echo "== Stage 3: wait mapped_code.pt then start rec-only LoRA jobs =="
for i in "${!mapper_specs[@]}"; do
  read -r source_name _source_code hidden_dim num_blocks lr epochs <<< "${mapper_specs[$i]}"
  exp_dir=$(mapper_exp_dir "${source_name}" "${hidden_dim}" "${num_blocks}" "${lr}" "${epochs}")
  mapped_code="${exp_dir}/codewords/mapped_code.pt"
  while [ ! -f "${mapped_code}" ]; do
    echo "waiting ${mapped_code}"
    sleep "${poll_seconds}"
  done

  gpu=$(gpu_at "$((i + 1))")
  mapper_exp_dir="${exp_dir}" \
    source_name="${source_name}_mapped_h${hidden_dim}_b${num_blocks}" \
    fc_lora_rank=256 \
    fc_lora_alpha=1024 \
    ffn_lora_rank=16 \
    ffn_lora_alpha=64 \
    lr=5e-4 \
    eta_min=1e-4 \
    epochs=400 \
    gpu="${gpu}" \
    bash staged_mlp_lora/scripts/train_lora.sh
done

echo "all staged jobs submitted"
