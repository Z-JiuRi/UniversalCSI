#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

target_exp=exps/COST2100/in/seed42/transnet_transnet
token_size=512
hidden_dim=2048
lr=2e-4
epochs=400
seed=42
ode_steps=16
batch_size=1024
max_samples=0
max_guide_codes=0
report_dir=decoder_param_fm/reports/eval_all
mkdir -p "$report_dir"

run_eval_one() {
  run_name=${condition_extract}_${condition_inject}_${param_norm}_tok${token_size}_h${hidden_dim}_lr${lr}_ep${epochs}_seed${seed}
  exp_dir=decoder_param_fm/exps/${run_name}
  log_path=${report_dir}/${run_name}.log

  {
    echo "run_name=${run_name}"
    echo "exp_dir=${exp_dir}"
    echo "gpu=${gpu}"
    date

    exp_dir=$exp_dir \
    ode_steps=$ode_steps \
    gpu=$gpu \
    max_guide_codes=$max_guide_codes \
    output=${exp_dir}/generated/generated_decoder.pth \
    bash decoder_param_fm/scripts/sample_param_fm.sh

    target_exp=$target_exp \
    exp_dir=$exp_dir \
    decoder_state=${exp_dir}/generated/generated_decoder.pth \
    batch_size=$batch_size \
    max_samples=$max_samples \
    gpu=$gpu \
    output_json=${exp_dir}/generated/nmse.json \
    bash decoder_param_fm/scripts/test_generated_nmse.sh

    cp "${exp_dir}/generated/nmse.json" "${report_dir}/${run_name}.json"
    date
  } > "$log_path" 2>&1
}

run_batch() {
  param_norm=rms condition_extract=random condition_inject=film gpu=0 run_eval_one &
  param_norm=rms condition_extract=random condition_inject=cross_attention gpu=1 run_eval_one &
  param_norm=rms condition_extract=random condition_inject=hyper_lora gpu=4 run_eval_one &
  param_norm=rms condition_extract=svd condition_inject=film gpu=6 run_eval_one &
  param_norm=rms condition_extract=svd condition_inject=cross_attention gpu=7 run_eval_one &
  param_norm=rms condition_extract=svd condition_inject=hyper_lora gpu=5 run_eval_one &
  wait

  param_norm=rms condition_extract=set_transformer condition_inject=film gpu=0 run_eval_one &
  param_norm=rms condition_extract=set_transformer condition_inject=cross_attention gpu=1 run_eval_one &
  param_norm=rms condition_extract=set_transformer condition_inject=hyper_lora gpu=4 run_eval_one &
  param_norm=zscore condition_extract=random condition_inject=film gpu=6 run_eval_one &
  param_norm=zscore condition_extract=random condition_inject=cross_attention gpu=7 run_eval_one &
  param_norm=zscore condition_extract=random condition_inject=hyper_lora gpu=5 run_eval_one &
  wait

  param_norm=zscore condition_extract=svd condition_inject=film gpu=0 run_eval_one &
  param_norm=zscore condition_extract=svd condition_inject=cross_attention gpu=1 run_eval_one &
  param_norm=zscore condition_extract=svd condition_inject=hyper_lora gpu=4 run_eval_one &
  param_norm=zscore condition_extract=set_transformer condition_inject=film gpu=6 run_eval_one &
  param_norm=zscore condition_extract=set_transformer condition_inject=cross_attention gpu=7 run_eval_one &
  param_norm=zscore condition_extract=set_transformer condition_inject=hyper_lora gpu=5 run_eval_one &
  wait
}

run_batch
