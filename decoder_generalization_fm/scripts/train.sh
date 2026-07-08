#!/bin/bash

# 训练 decoder_generalization_fm。
#
# 常用命令：
#   gpu=4 epochs=1 batch_size=2 max_condition_codes=512 eval_every=1 eval_max_entries=1 eval_max_samples=128 \
#     exp_name=smoke bash decoder_generalization_fm/scripts/train.sh
#
# 关键参数：
#   condition_extract=random|svd|set_transformer
#   condition_inject=film|cross_attention|hyper_lora
#   batch_size 表示每次 optimizer update 使用几个实验目录样本。
#   max_condition_codes=0 表示使用完整 train_code.pt；smoke test 建议设小。

set -euo pipefail

data_txt=${data_txt:-decoder_generalization_fm/data/data.txt}
exp_name=${exp_name:-generalization_fm}
exp_dir=${exp_dir:-decoder_generalization_fm/exps/${exp_name}}
stats_cache=${stats_cache:-decoder_generalization_fm/data/train_tensor_zscore_stats.pt}

epochs=${epochs:-400}
batch_size=${batch_size:-4}
steps_per_epoch=${steps_per_epoch:-0}
lr=${lr:-2e-4}
eta_min=${eta_min:-1e-6}
weight_decay=${weight_decay:-0.0}
warmup_ratio=${warmup_ratio:-0.1}
warmup_steps=${warmup_steps:-0}
grad_clip=${grad_clip:-1.0}

token_size=${token_size:-512}
condition_extract=${condition_extract:-svd}
condition_inject=${condition_inject:-film}
condition_tokens=${condition_tokens:-512}
hidden_dim=${hidden_dim:-512}
num_blocks=${num_blocks:-4}
time_dim=${time_dim:-128}
cond_dim=${cond_dim:-512}
num_heads=${num_heads:-8}
set_layers=${set_layers:-2}
hyper_lora_rank=${hyper_lora_rank:-16}
dropout=${dropout:-0.0}
lambda_endpoint=${lambda_endpoint:-1.0}
t_eps=${t_eps:-1e-4}
ode_steps=${ode_steps:-16}
max_condition_codes=${max_condition_codes:-0}

eval_every=${eval_every:-20}
eval_batch_size=${eval_batch_size:-1024}
eval_max_samples=${eval_max_samples:-0}
eval_max_entries=${eval_max_entries:-0}
save_every=${save_every:-0}
seed=${seed:-42}
gpu=${gpu:-0}
cpu=${cpu:-0}
background=${background:-1}

if [ ! -f "${data_txt}" ]; then
  echo "Missing data_txt: ${data_txt}" >&2
  exit 1
fi

mkdir -p "${exp_dir}"
extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
fi
cmd=(
  python -u decoder_generalization_fm/train.py
  --data_txt "${data_txt}"
  --exp_dir "${exp_dir}"
  --stats_cache "${stats_cache}"
  --epochs "${epochs}"
  --batch_size "${batch_size}"
  --steps_per_epoch "${steps_per_epoch}"
  --lr "${lr}"
  --eta_min "${eta_min}"
  --weight_decay "${weight_decay}"
  --warmup_ratio "${warmup_ratio}"
  --warmup_steps "${warmup_steps}"
  --grad_clip "${grad_clip}"
  --token_size "${token_size}"
  --condition_extract "${condition_extract}"
  --condition_inject "${condition_inject}"
  --condition_tokens "${condition_tokens}"
  --hidden_dim "${hidden_dim}"
  --num_blocks "${num_blocks}"
  --time_dim "${time_dim}"
  --cond_dim "${cond_dim}"
  --num_heads "${num_heads}"
  --set_layers "${set_layers}"
  --hyper_lora_rank "${hyper_lora_rank}"
  --dropout "${dropout}"
  --lambda_endpoint "${lambda_endpoint}"
  --t_eps "${t_eps}"
  --ode_steps "${ode_steps}"
  --max_condition_codes "${max_condition_codes}"
  --eval_every "${eval_every}"
  --eval_batch_size "${eval_batch_size}"
  --eval_max_samples "${eval_max_samples}"
  --eval_max_entries "${eval_max_entries}"
  --save_every "${save_every}"
  --seed "${seed}"
  --gpu "${gpu}"
  "${extra_args[@]}"
)

if [ "${background}" = "1" ]; then
  "${cmd[@]}" > /dev/null 2>&1 &
  echo "started pid=$! exp_dir=${exp_dir}"
else
  "${cmd[@]}"
fi
