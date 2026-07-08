#!/bin/bash
#
# 泛化 FM 训练启动脚本（run_one 模式）。
#
# 用法：
#   直接取消注释需要的配置行后运行，或临时指定：
#     gpu=0 background=0 bash decoder_generalization_fm/scripts/run.sh
#
# 背景运行（默认）：
#   train.sh 内部 background=1 时会将进程放到后台，日志写入文件。
#   设为 background=0 则前台运行。

set -euo pipefail

# ========== 全局默认值 ==========

data_txt=${data_txt:-decoder_generalization_fm/data/data.txt}
stats_cache=${stats_cache:-decoder_generalization_fm/data/train_tensor_zscore_stats.pt}

token_size=${token_size:-64}
condition_tokens=${condition_tokens:-512}
hidden_dim=${hidden_dim:-2048}
num_blocks=${num_blocks:-4}
cond_dim=${cond_dim:-512}
num_heads=${num_heads:-8}
set_layers=${set_layers:-2}
epochs=${epochs:-400}
batch_size=${batch_size:-1}
steps_per_epoch=${steps_per_epoch:-0}
lr=${lr:-2e-4}
warmup_ratio=${warmup_ratio:-0.1}
lambda_endpoint=${lambda_endpoint:-1.0}
t_eps=${t_eps:-1e-4}
seed=${seed:-42}
max_condition_codes=${max_condition_codes:-0}
gpu=${gpu:-0}
background=${background:-1}

# ========== run_one ==========

run_one() {
  exp_name="${condition_extract}_${condition_inject}_h${hidden_dim}_b${num_blocks}_lr${lr}_ep${epochs}"
  exp_dir="decoder_generalization_fm/exps/${exp_name}"

  data_txt="${data_txt}" \
  stats_cache="${stats_cache}" \
  exp_name="${exp_name}" \
  exp_dir="${exp_dir}" \
  condition_extract="${condition_extract}" \
  condition_inject="${condition_inject}" \
  token_size="${token_size}" \
  condition_tokens="${condition_tokens}" \
  hidden_dim="${hidden_dim}" \
  num_blocks="${num_blocks}" \
  cond_dim="${cond_dim}" \
  num_heads="${num_heads}" \
  set_layers="${set_layers}" \
  epochs="${epochs}" \
  batch_size="${batch_size}" \
  steps_per_epoch="${steps_per_epoch}" \
  lr="${lr}" \
  warmup_ratio="${warmup_ratio}" \
  lambda_endpoint="${lambda_endpoint}" \
  t_eps="${t_eps}" \
  seed="${seed}" \
  max_condition_codes="${max_condition_codes}" \
  gpu="${gpu}" \
  background="${background}" \
  bash decoder_generalization_fm/scripts/train.sh
}

# ========== 配置列表 ==========

# 取消注释所需行即可启动，gpu 按需指定。

# ##################################################
# hidden_dim=512  num_blocks=4  lr=2e-4
# ##################################################

# condition_extract=svd                 condition_inject=film              gpu=0 run_one
# condition_extract=svd                 condition_inject=cross_attention   gpu=1 run_one
# condition_extract=random              condition_inject=film              gpu=4 run_one
# condition_extract=set_transformer     condition_inject=cross_attention   gpu=6 run_one

# ##################################################
# hidden_dim=512  num_blocks=4  lr=5e-4
# ##################################################

# lr=5e-4 condition_extract=svd               condition_inject=film              gpu=0 run_one
# lr=5e-4 condition_extract=svd               condition_inject=cross_attention   gpu=1 run_one
# lr=5e-4 condition_extract=random            condition_inject=film              gpu=4 run_one
steps_per_epoch=1000 hidden_dim=1024 lr=5e-4 batch_size=1 epochs=400 condition_extract=set_transformer condition_inject=film gpu=0 run_one

# ##################################################
# hidden_dim=1024  num_blocks=6  lr=2e-4
# ##################################################

# hidden_dim=1024 num_blocks=6 condition_extract=set_transformer condition_inject=film gpu=0 run_one
