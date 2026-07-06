#!/usr/bin/env bash
set -euo pipefail

target_exp=exps/COST2100/in/seed42/transnet_transnet
token_size=64
condition_tokens=512
hidden_dim=512
num_blocks=4
cond_dim=512
num_heads=8
set_layers=2
epochs=400
steps_per_epoch=100
lr=1e-4
warmup_ratio=0.1
lambda_endpoint=0.0
base_seed=2026
seed=42
max_guide_codes=0
ode_steps=16
batch_size=1024
max_samples=0

run_one() {
  run_name=${condition_extract}_${condition_inject}_${param_norm}_tok${token_size}_h${hidden_dim}_lr${lr}_ep${epochs}_seed${seed}
  exp_dir=decoder_param_fm/exps/${run_name}

  target_exp=$target_exp \
  condition_extract=$condition_extract \
  condition_inject=$condition_inject \
  param_norm=$param_norm \
  token_size=$token_size \
  condition_tokens=$condition_tokens \
  hidden_dim=$hidden_dim \
  num_blocks=$num_blocks \
  cond_dim=$cond_dim \
  num_heads=$num_heads \
  set_layers=$set_layers \
  epochs=$epochs \
  steps_per_epoch=$steps_per_epoch \
  lr=$lr \
  warmup_ratio=$warmup_ratio \
  lambda_endpoint=$lambda_endpoint \
  base_seed=$base_seed \
  seed=$seed \
  gpu=$gpu \
  max_guide_codes=$max_guide_codes \
  run_name=$run_name \
  exp_dir=$exp_dir \
  bash decoder_param_fm/scripts/train_param_fm.sh

  # exp_dir=$exp_dir \
  # ode_steps=$ode_steps \
  # gpu=$gpu \
  # max_guide_codes=$max_guide_codes \
  # output=${exp_dir}/generated/generated_decoder.pth \
  # bash decoder_param_fm/scripts/sample_param_fm.sh

  # target_exp=$target_exp \
  # exp_dir=$exp_dir \
  # decoder_state=${exp_dir}/generated/generated_decoder.pth \
  # batch_size=$batch_size \
  # max_samples=$max_samples \
  # gpu=$gpu \
  # output_json=${exp_dir}/generated/nmse.json \
  # bash decoder_param_fm/scripts/test_generated_nmse.sh
}

# ##################################################
# param_norm=rms condition_extract=random condition_inject=film gpu=0 run_one
# param_norm=rms condition_extract=random condition_inject=cross_attention gpu=1 run_one
# param_norm=rms condition_extract=random condition_inject=hyper_lora gpu=4 run_one

# param_norm=rms condition_extract=svd condition_inject=film gpu=6 run_one
# param_norm=rms condition_extract=svd condition_inject=cross_attention gpu=7 run_one
# param_norm=rms condition_extract=svd condition_inject=hyper_lora gpu=0 run_one

# param_norm=rms condition_extract=set_transformer condition_inject=film gpu=1 run_one
# param_norm=rms condition_extract=set_transformer condition_inject=cross_attention gpu=4 run_one
# param_norm=rms condition_extract=set_transformer condition_inject=hyper_lora gpu=6 run_one
# ##################################################

# ##################################################
# param_norm=zscore condition_extract=random condition_inject=film gpu=7 run_one
# param_norm=zscore condition_extract=random condition_inject=cross_attention gpu=0 run_one
# param_norm=zscore condition_extract=random condition_inject=hyper_lora gpu=1 run_one

# param_norm=zscore condition_extract=svd condition_inject=film gpu=4 run_one
# param_norm=zscore condition_extract=svd condition_inject=cross_attention gpu=6 run_one
# param_norm=zscore condition_extract=svd condition_inject=hyper_lora gpu=4 run_one

epochs=1000 lr=1e-4 param_norm=zscore condition_extract=set_transformer condition_inject=film gpu=0 run_one
# param_norm=zscore condition_extract=set_transformer condition_inject=cross_attention gpu=0 run_one
# param_norm=zscore condition_extract=set_transformer condition_inject=hyper_lora gpu=1 run_one
# ##################################################
