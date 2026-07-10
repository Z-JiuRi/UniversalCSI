#!/usr/bin/env bash
set -euo pipefail

# Default experiment:
#   source: seed1014/transnet_transnet
#   target: seed1024/transnet_transnet
#   mapper: closed-form affine + residual MLP
#
# Usage:
#   bash adapter/scripts/run_adapter.sh
#   epochs=200 batch_size=1024 bash adapter/scripts/run_adapter.sh
#   python_bin=/nfs5/zxd/.envs/miniconda3/envs/torch/bin/python bash adapter/scripts/run_adapter.sh

export source_seed="${source_seed:-1014}"
export target_seed="${target_seed:-1024}"
export source_encoder="${source_encoder:-transnet}"
export source_decoder="${source_decoder:-transnet}"
export target_encoder="${target_encoder:-transnet}"
export target_decoder="${target_decoder:-transnet}"

export mapper_type="${mapper_type:-affine_residual_mlp}"
export hidden_dim="${hidden_dim:-2048}"
export num_blocks="${num_blocks:-4}"
export dropout="${dropout:-0.0}"
export use_final_norm="${use_final_norm:-0}"

export epochs="${epochs:-400}"
export batch_size="${batch_size:-256}"
export eval_every="${eval_every:-10}"
export workers="${workers:-0}"
export weight_decay="${weight_decay:-1e-4}"
export scheduler="${scheduler:-cosine}"
export export_codewords="${export_codewords:-1}"
export max_train_samples="${max_train_samples:-0}"
export max_eval_samples="${max_eval_samples:-0}"

export seed="${seed:-${target_seed}}"
export cpu="${cpu:-0}"
export python_bin="${python_bin:-python}"


launch_case() {
  local case_gpu="$1"
  local case_residual_scale="$2"
  local case_no_block_norm="$3"
  local case_align_ridge="$4"
  local case_train_affine="$5"
  local case_lambda_code="$6"
  local case_lambda_recon="$7"
  local case_lr="$8"
  local case_eta_min="$9"

  local norm_name="block_norm"
  if [[ "${case_no_block_norm}" == "1" ]]; then
    norm_name="no_block_norm"
  fi
  local case_name="code${case_lambda_code}_rec${case_lambda_recon}_lr${case_lr}_ep${epochs}_${norm_name}_ridge${case_align_ridge}_ta${case_train_affine}_rs${case_residual_scale}_h${hidden_dim}"

  echo "launch ${case_name} on gpu=${case_gpu}"
  gpu="${case_gpu}" \
  residual_scale="${case_residual_scale}" \
  no_block_norm="${case_no_block_norm}" \
  align_ridge="${case_align_ridge}" \
  train_affine="${case_train_affine}" \
  lambda_code="${case_lambda_code}" \
  lambda_recon="${case_lambda_recon}" \
  lr="${case_lr}" \
  eta_min="${case_eta_min}" \
  exp_name="${case_name}" \
  bash adapter/scripts/train_adapter.sh
}

# launch_case | gpu | residual_scale | no_block_norm | align_ridge | train_affine | lambda_code | lambda_recon | lr | eta_min
launch_case      0          0.1              0             0.0            0             1.0            0.0      5e-4   1e-4
launch_case      0          0.5              0             0.0            0             1.0            0.0      5e-4   1e-4
launch_case      0          1.0              0             0.0            0             1.0            0.0      5e-4   1e-4
launch_case      0          0.1              1             0.0            0             1.0            0.0      5e-4   1e-4
launch_case      0          0.1              0             1.0            0             1.0            0.0      5e-4   1e-4
launch_case      0          0.1              0             0.0            1             1.0            0.0      5e-4   1e-4
launch_case      0          0.1              0             1.0            0             1.0          500.0      5e-4   1e-4
launch_case      0          0.1              0             1.0            0             1.0         1000.0      5e-4   1e-4
