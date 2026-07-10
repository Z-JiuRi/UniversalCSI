#!/usr/bin/env bash
set -euo pipefail

source_seed="${source_seed:-1014}"
target_seed="${target_seed:-1024}"
source_encoder="${source_encoder:-transnet}"
source_decoder="${source_decoder:-transnet}"
target_encoder="${target_encoder:-transnet}"
target_decoder="${target_decoder:-transnet}"
source_arch="${source_arch:-${source_encoder}_${source_decoder}}"
target_arch="${target_arch:-${target_encoder}_${target_decoder}}"
source_exp="${source_exp:-exps/COST2100/in/base/seed${source_seed}/${source_arch}}"
target_exp="${target_exp:-exps/COST2100/in/base/seed${target_seed}/${target_arch}}"
target_decoder_exp="${target_decoder_exp:-${target_exp}}"
decoder_checkpoint="${decoder_checkpoint:-${target_decoder_exp}/checkpoints/best_nmse.pth}"
decoder_args_json="${decoder_args_json:-}"
if [[ -z "${decoder_args_json}" && -f "${target_decoder_exp}/args.json" ]]; then
  decoder_args_json="${target_decoder_exp}/args.json"
fi
source_train_code="${source_train_code:-}"
source_val_code="${source_val_code:-}"
source_test_code="${source_test_code:-}"
target_train_code="${target_train_code:-}"
target_val_code="${target_val_code:-}"
target_test_code="${target_test_code:-}"

train_csi="${train_csi:-/nfs5/zxd/Huawei/datasets/COST2100/in_train.pt}"
val_csi="${val_csi:-/nfs5/zxd/Huawei/datasets/COST2100/in_val.pt}"
test_csi="${test_csi:-/nfs5/zxd/Huawei/datasets/COST2100/in_test.pt}"

mapper_type="${mapper_type:-affine_residual_mlp}"
hidden_dim="${hidden_dim:-512}"
lowrank_rank="${lowrank_rank:-64}"
num_blocks="${num_blocks:-4}"
dropout="${dropout:-0.0}"
residual_scale="${residual_scale:-0.1}"
learnable_residual_gate="${learnable_residual_gate:-0}"
gate_max="${gate_max:-0.5}"
use_final_norm="${use_final_norm:-0}"
no_block_norm="${no_block_norm:-0}"
train_affine="${train_affine:-0}"
align_ridge="${align_ridge:-1.0}"

lambda_code="${lambda_code:-1.0}"
lambda_recon="${lambda_recon:-0.0}"
code_loss_type="${code_loss_type:-mse}"
std_weight_min="${std_weight_min:-0.25}"
std_weight_max="${std_weight_max:-4.0}"
std_weight_eps="${std_weight_eps:-1e-6}"
epochs="${epochs:-100}"
batch_size="${batch_size:-256}"
workers="${workers:-0}"
lr="${lr:-1e-3}"
weight_decay="${weight_decay:-1e-4}"
scheduler="${scheduler:-cosine}"
eta_min="${eta_min:-1e-4}"
eval_every="${eval_every:-10}"
export_codewords="${export_codewords:-1}"
max_train_samples="${max_train_samples:-0}"
max_eval_samples="${max_eval_samples:-0}"
seed="${seed:-2026}"
gpu="${gpu:-2}"
cpu="${cpu:-0}"
python_bin="${python_bin:-python}"

channel="${channel:-2}"
nt="${nt:-32}"
nc="${nc:-32}"
cr="${cr:-4}"
d_model="${d_model:-64}"
dim_feedforward="${dim_feedforward:-2048}"
hidden="${hidden:-16}"
decoder_num_blocks="${decoder_num_blocks:-2}"
decoder="${decoder:-${target_decoder}}"

exp_seed="${exp_seed:-seed${source_seed}}"
exp_arch="${exp_arch:-${source_encoder}}"
exp_name="${exp_name:-code${lambda_code}_rec${lambda_recon}_lr${lr}_ep${epochs}}"
exp_dir="${exp_dir:-adapter/exps/${mapper_type}/${exp_seed}/${exp_arch}/${exp_name}}"

cmd=(
  "${python_bin}" -u adapter/train_adapter.py
  --source_exp "${source_exp}"
  --target_exp "${target_exp}"
  --train_csi "${train_csi}"
  --val_csi "${val_csi}"
  --test_csi "${test_csi}"
  --decoder_checkpoint "${decoder_checkpoint}"
  --exp_dir "${exp_dir}"
  --mapper_type "${mapper_type}"
  --hidden_dim "${hidden_dim}"
  --lowrank_rank "${lowrank_rank}"
  --num_blocks "${num_blocks}"
  --dropout "${dropout}"
  --residual_scale "${residual_scale}"
  --gate_max "${gate_max}"
  --align_ridge "${align_ridge}"
  --lambda_code "${lambda_code}"
  --lambda_recon "${lambda_recon}"
  --code_loss_type "${code_loss_type}"
  --std_weight_min "${std_weight_min}"
  --std_weight_max "${std_weight_max}"
  --std_weight_eps "${std_weight_eps}"
  --epochs "${epochs}"
  --batch_size "${batch_size}"
  --workers "${workers}"
  --lr "${lr}"
  --weight_decay "${weight_decay}"
  --scheduler "${scheduler}"
  --eta_min "${eta_min}"
  --eval_every "${eval_every}"
  --max_train_samples "${max_train_samples}"
  --max_eval_samples "${max_eval_samples}"
  --seed "${seed}"
  --channel "${channel}"
  --nt "${nt}"
  --nc "${nc}"
  --decoder "${decoder}"
  --cr "${cr}"
  --d_model "${d_model}"
  --dim_feedforward "${dim_feedforward}"
  --hidden "${hidden}"
  --decoder_num_blocks "${decoder_num_blocks}"
)

if [[ -n "${decoder_args_json}" ]]; then
  cmd+=(--decoder_args_json "${decoder_args_json}")
fi
if [[ -n "${source_train_code}" ]]; then
  cmd+=(--source_train_code "${source_train_code}")
fi
if [[ -n "${source_val_code}" ]]; then
  cmd+=(--source_val_code "${source_val_code}")
fi
if [[ -n "${source_test_code}" ]]; then
  cmd+=(--source_test_code "${source_test_code}")
fi
if [[ -n "${target_train_code}" ]]; then
  cmd+=(--target_train_code "${target_train_code}")
fi
if [[ -n "${target_val_code}" ]]; then
  cmd+=(--target_val_code "${target_val_code}")
fi
if [[ -n "${target_test_code}" ]]; then
  cmd+=(--target_test_code "${target_test_code}")
fi
if [[ "${use_final_norm}" == "1" ]]; then
  cmd+=(--use_final_norm)
fi
if [[ "${no_block_norm}" == "1" ]]; then
  cmd+=(--no_block_norm)
fi
if [[ "${train_affine}" == "1" ]]; then
  cmd+=(--train_affine)
fi
if [[ "${learnable_residual_gate}" == "1" ]]; then
  cmd+=(--learnable_residual_gate)
fi
if [[ "${export_codewords}" == "1" ]]; then
  cmd+=(--export_codewords)
fi
if [[ "${cpu}" == "1" ]]; then
  cmd+=(--cpu)
else
  cmd+=(--gpu "${gpu}")
fi

"${cmd[@]}" > /dev/null 2>&1 &
