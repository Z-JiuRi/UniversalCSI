#!/bin/bash

# 训练 code-only flow matching 码字转换器。
#
# 目标：
#   给定 source code z_s 和 teacher code z_t，学习一个连续速度场，
#   从起点 z0 流到 z_t。z0 可以是原始 z_s，也可以是确定性的
#   Procrustes/全仿射预对齐结果。
#
# 常用运行方式：
#   bash flow_matching/scripts/train_flow_matching.sh
#   source_name=seed2026_clnet_transnet source_code=exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt gpu=4 bash flow_matching/scripts/train_flow_matching.sh
#   align_mode=procrustes lambda_endpoint=0.1 ode_steps=16 bash flow_matching/scripts/train_flow_matching.sh
#   align_mode=affine hidden_dim=2048 num_blocks=4 epochs=400 gpu=0 bash flow_matching/scripts/train_flow_matching.sh
#   background=1 gpu=7 bash flow_matching/scripts/train_flow_matching.sh
#
# 关键参数：
#   align_mode=identity|procrustes|affine
#   condition=source|start|source_start|none
#   lambda_endpoint=0 表示只训练速度 MSE；>0 额外约束单步终点。
#   ode_steps 推理导出 mapped_code 时的 ODE 步数。
#   scheduler=cosine 使用主项目同款 10% warmup + cosine annealing。

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
source_code=${source_code:-exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt}
source_name=${source_name:-seed2026_transnet_transnet}

epochs=${epochs:-400}
batch_size=${batch_size:-512}
workers=${workers:-0}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
scheduler=${scheduler:-cosine}
eta_min=${eta_min:-5e-5}
hidden_dim=${hidden_dim:-2048}
num_blocks=${num_blocks:-4}
time_dim=${time_dim:-128}
condition=${condition:-source_start}
dropout=${dropout:-0.0}
align_mode=${align_mode:-identity}
align_ridge=${align_ridge:-1e-4}
val_ratio=${val_ratio:-0}
max_samples=${max_samples:-0}
t_eps=${t_eps:-1e-4}
lambda_endpoint=${lambda_endpoint:-0.0}
ode_steps=${ode_steps:-16}
ode_method=${ode_method:-euler}
eval_ode_every=${eval_ode_every:-0}
save_last=${save_last:-0}
gpu=${gpu:-0}
seed=${seed:-2026}
cpu=${cpu:-0}
background=${background:-1}

tag="align${align_mode}_cond${condition}_end${lambda_endpoint}_ode${ode_steps}_${ode_method}"
exp_dir=${exp_dir:-flow_matching/exps/code_only/${tag}/${source_name}_to_seed42_transnet_lr${lr}_ep${epochs}}

if [ ! -f "${source_code}" ]; then
  echo "Missing source_code: ${source_code}" >&2
  exit 1
fi
if [ ! -f "${target_code}" ]; then
  echo "Missing target_code: ${target_code}" >&2
  exit 1
fi

mkdir -p "${exp_dir}"
extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
fi
if [ "${save_last}" = "1" ]; then
  extra_args+=(--save_last)
fi

cmd=(
  python -u flow_matching/train_flow_matching.py
  --source_code "${source_code}"
  --target_code "${target_code}"
  --exp_dir "${exp_dir}"
  --epochs "${epochs}"
  --batch_size "${batch_size}"
  --workers "${workers}"
  --lr "${lr}"
  --weight_decay "${weight_decay}"
  --scheduler "${scheduler}"
  --eta_min "${eta_min}"
  --hidden_dim "${hidden_dim}"
  --num_blocks "${num_blocks}"
  --time_dim "${time_dim}"
  --condition "${condition}"
  --dropout "${dropout}"
  --align_mode "${align_mode}"
  --align_ridge "${align_ridge}"
  --val_ratio "${val_ratio}"
  --max_samples "${max_samples}"
  --t_eps "${t_eps}"
  --lambda_endpoint "${lambda_endpoint}"
  --ode_steps "${ode_steps}"
  --ode_method "${ode_method}"
  --eval_ode_every "${eval_ode_every}"
  --gpu "${gpu}"
  --seed "${seed}"
  "${extra_args[@]}"
)

if [ "${background}" = "1" ]; then
  "${cmd[@]}" > /dev/null 2>&1 &
  echo "started pid=$! exp_dir=${exp_dir}"
else
  "${cmd[@]}"
fi
