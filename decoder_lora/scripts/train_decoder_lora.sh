#!/bin/bash

# 训练 fixed seed42 decoder 的 LoRA 适配器。
#
# 数据流：
#   source code -> closed-form affine/procrustes -> aligned code z0
#   z0 -> optional code adapter (gated_lr_mlp / affine_res_mlp) -> z1
#   z1 -> frozen seed42 decoder + LoRA(fc_decoder + FFN) -> CSI
#
# 常用命令：
#   source_name=seed2026_transnet_transnet \
#   source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
#   lora_target=fc_ffn lora_rank=8 gpu=0 bash decoder_lora/scripts/train_decoder_lora.sh
#
# 可调项：
#   align_mode=identity|procrustes|affine
#   align_with_val_code=0|1  # 1 means fit affine with train+val+test code
#   lora_target=fc|ffn|fc_ffn
#   lora_rank=8|16|32
#   fc_lora_rank=128 ffn_lora_rank=8
#   code_adapter=none|gated_lr_mlp|affine_res_mlp
#   code_lowrank_rank=128 code_mlp_hidden=512 code_gate_lr=0.1 code_gate_mlp=0.1
#   code_hidden_dim=1024 code_num_blocks=4 code_residual_scale=1.0
#   lambda_recT=0.0 lambda_fc=0.0 lambda_code=0.0 lambda_delta=0.0

set -euo pipefail

source_name=${source_name:-seed2026_transnet_transnet}
source_code=${source_code:-exps/COST2100/in/base/seed2026/transnet_transnet/codewords/train_code.pt}
target_code=${target_code:-exps/COST2100/in/base/seed42/transnet_transnet/codewords/train_code.pt}
source_val_code=${source_val_code:-}
target_val_code=${target_val_code:-}
source_test_code=${source_test_code:-}
target_test_code=${target_test_code:-}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
val_csi_path=${val_csi_path:-}
test_csi_path=${test_csi_path:-}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/base/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/base/seed42/transnet_transnet/args.json}
source_checkpoint=${source_checkpoint:-}
source_args_json=${source_args_json:-}

align_mode=${align_mode:-affine}
align_ridge=${align_ridge:-1e-4}
align_with_val_code=${align_with_val_code:-1}
lora_target=${lora_target:-fc_ffn}
lora_rank=${lora_rank:-8}
fc_lora_rank=${fc_lora_rank:-}
ffn_lora_rank=${ffn_lora_rank:-}
lora_alpha=${lora_alpha:-}
fc_lora_alpha=${fc_lora_alpha:-}
ffn_lora_alpha=${ffn_lora_alpha:-}
lora_dropout=${lora_dropout:-0.0}
code_adapter=${code_adapter:-none}
code_lowrank_rank=${code_lowrank_rank:-0}
code_mlp_hidden=${code_mlp_hidden:-0}
code_gate_lr=${code_gate_lr:-0.1}
code_gate_mlp=${code_gate_mlp:-0.1}
code_adapter_dropout=${code_adapter_dropout:-0.0}
code_hidden_dim=${code_hidden_dim:-1024}
code_num_blocks=${code_num_blocks:-4}
code_residual_scale=${code_residual_scale:-1.0}
code_use_final_norm=${code_use_final_norm:-0}
epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-0}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
scheduler=${scheduler:-cosine}
eta_min=${eta_min:-5e-5}
val_ratio=${val_ratio:-0}
max_samples=${max_samples:-0}
eval_decoder_every=${eval_decoder_every:-20}
eval_decoder_max_samples=${eval_decoder_max_samples:-0}
eval_external_max_samples=${eval_external_max_samples:-0}
lambda_code=${lambda_code:-0.0}
lambda_delta=${lambda_delta:-0.0}
lambda_recT=${lambda_recT:-0.0}
lambda_fc=${lambda_fc:-0.0}
save_last=${save_last:-0}
gpu=${gpu:-0}
seed=${seed:-2026}
cpu=${cpu:-0}

rank_tag="r${lora_rank}"
if [ -n "${fc_lora_rank}" ] || [ -n "${ffn_lora_rank}" ]; then
  fc_rank_tag=${fc_lora_rank:-${lora_rank}}
  ffn_rank_tag=${ffn_lora_rank:-${lora_rank}}
  fc_alpha_tag=${fc_lora_alpha:-auto}
  ffn_alpha_tag=${ffn_lora_alpha:-auto}
  rank_tag="fcr${fc_rank_tag}a${fc_alpha_tag}_ffnr${ffn_rank_tag}a${ffn_alpha_tag}"
fi
adapter_tag="code_adapter_none"
if [ "${code_adapter}" = "gated_lr_mlp" ]; then
  adapter_tag="code_${code_adapter}_lr${code_lowrank_rank}_h${code_mlp_hidden}_glr${code_gate_lr}_gmlp${code_gate_mlp}_drop${code_adapter_dropout}_delta${lambda_delta}"
elif [ "${code_adapter}" = "affine_res_mlp" ]; then
  adapter_tag="code_${code_adapter}_h${code_hidden_dim}_b${code_num_blocks}_rs${code_residual_scale}_fn${code_use_final_norm}_drop${code_adapter_dropout}"
fi
align_tag="align${align_mode}"
if [ "${align_with_val_code}" = "1" ]; then
  align_tag="${align_tag}_fittrainvaltest"
fi
tag="${align_tag}_${lora_target}_${rank_tag}_${adapter_tag}_recT${lambda_recT}_fc${lambda_fc}_code${lambda_code}"
exp_dir=${exp_dir:-decoder_lora/exps/${tag}/${source_name}_to_seed42_lr${lr}_eta_${eta_min}_ep${epochs}}

extra_args=()

if [ ! -f "${source_code}" ]; then
  echo "Missing source_code: ${source_code}" >&2
  exit 1
fi
if [ ! -f "${target_code}" ]; then
  echo "Missing target_code: ${target_code}" >&2
  exit 1
fi
if [ "${align_with_val_code}" = "1" ]; then
  source_val_code=${source_val_code:-$(dirname "${source_code}")/val_code.pt}
  target_val_code=${target_val_code:-$(dirname "${target_code}")/val_code.pt}
  source_test_code=${source_test_code:-$(dirname "${source_code}")/test_code.pt}
  target_test_code=${target_test_code:-$(dirname "${target_code}")/test_code.pt}
fi
if [ -n "${source_val_code}" ] || [ -n "${target_val_code}" ]; then
  if [ -z "${source_val_code}" ] || [ -z "${target_val_code}" ]; then
    echo "source_val_code and target_val_code must be set together" >&2
    exit 1
  fi
  if [ ! -f "${source_val_code}" ]; then
    echo "Missing source_val_code: ${source_val_code}" >&2
    exit 1
  fi
  if [ ! -f "${target_val_code}" ]; then
    echo "Missing target_val_code: ${target_val_code}" >&2
    exit 1
  fi
  extra_args+=(--source_align_code "${source_val_code}")
  extra_args+=(--target_align_code "${target_val_code}")
fi
if [ -n "${source_test_code}" ] || [ -n "${target_test_code}" ]; then
  if [ -z "${source_test_code}" ] || [ -z "${target_test_code}" ]; then
    echo "source_test_code and target_test_code must be set together" >&2
    exit 1
  fi
  if [ ! -f "${source_test_code}" ]; then
    echo "Missing source_test_code: ${source_test_code}" >&2
    exit 1
  fi
  if [ ! -f "${target_test_code}" ]; then
    echo "Missing target_test_code: ${target_test_code}" >&2
    exit 1
  fi
  extra_args+=(--source_align_code "${source_test_code}")
  extra_args+=(--target_align_code "${target_test_code}")
fi
if [ ! -f "${csi_path}" ]; then
  echo "Missing csi_path: ${csi_path}" >&2
  exit 1
fi
if [ ! -f "${decoder_checkpoint}" ]; then
  echo "Missing decoder_checkpoint: ${decoder_checkpoint}" >&2
  exit 1
fi
if [ ! -f "${decoder_args_json}" ]; then
  echo "Missing decoder_args_json: ${decoder_args_json}" >&2
  exit 1
fi

if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
else
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi
if [ "${save_last}" = "1" ]; then
  extra_args+=(--save_last)
fi
if [ -n "${lora_alpha}" ]; then
  extra_args+=(--lora_alpha "${lora_alpha}")
fi
if [ -n "${fc_lora_rank}" ]; then
  extra_args+=(--fc_lora_rank "${fc_lora_rank}")
fi
if [ -n "${ffn_lora_rank}" ]; then
  extra_args+=(--ffn_lora_rank "${ffn_lora_rank}")
fi
if [ -n "${fc_lora_alpha}" ]; then
  extra_args+=(--fc_lora_alpha "${fc_lora_alpha}")
fi
if [ -n "${ffn_lora_alpha}" ]; then
  extra_args+=(--ffn_lora_alpha "${ffn_lora_alpha}")
fi
if [ -n "${val_csi_path}" ]; then
  if [ ! -f "${val_csi_path}" ]; then
    echo "Missing val_csi_path: ${val_csi_path}" >&2
    exit 1
  fi
  extra_args+=(--val_csi_path "${val_csi_path}")
fi
if [ -n "${test_csi_path}" ]; then
  if [ ! -f "${test_csi_path}" ]; then
    echo "Missing test_csi_path: ${test_csi_path}" >&2
    exit 1
  fi
  extra_args+=(--test_csi_path "${test_csi_path}")
fi
if [ -n "${source_checkpoint}" ]; then
  if [ ! -f "${source_checkpoint}" ]; then
    echo "Missing source_checkpoint: ${source_checkpoint}" >&2
    exit 1
  fi
  extra_args+=(--source_checkpoint "${source_checkpoint}")
fi
if [ -n "${source_args_json}" ]; then
  if [ ! -f "${source_args_json}" ]; then
    echo "Missing source_args_json: ${source_args_json}" >&2
    exit 1
  fi
  extra_args+=(--source_args_json "${source_args_json}")
fi
if [ "${code_adapter}" = "affine_res_mlp" ]; then
  extra_args+=(--code_hidden_dim "${code_hidden_dim}")
  extra_args+=(--code_num_blocks "${code_num_blocks}")
  extra_args+=(--code_residual_scale "${code_residual_scale}")
fi
if [ "${code_use_final_norm}" = "1" ]; then
  extra_args+=(--code_use_final_norm)
fi

mkdir -p "${exp_dir}"

python -u decoder_lora/train_decoder_lora.py \
  --source_code "${source_code}" \
  --target_code "${target_code}" \
  --csi_path "${csi_path}" \
  --decoder_checkpoint "${decoder_checkpoint}" \
  --decoder_args_json "${decoder_args_json}" \
  --exp_dir "${exp_dir}" \
  --source_name "${source_name}" \
  --align_mode "${align_mode}" \
  --align_ridge "${align_ridge}" \
  --lora_target "${lora_target}" \
  --lora_rank "${lora_rank}" \
  --lora_dropout "${lora_dropout}" \
  --code_adapter "${code_adapter}" \
  --code_lowrank_rank "${code_lowrank_rank}" \
  --code_mlp_hidden "${code_mlp_hidden}" \
  --code_gate_lr "${code_gate_lr}" \
  --code_gate_mlp "${code_gate_mlp}" \
  --code_adapter_dropout "${code_adapter_dropout}" \
  --epochs "${epochs}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --scheduler "${scheduler}" \
  --eta_min "${eta_min}" \
  --val_ratio "${val_ratio}" \
  --max_samples "${max_samples}" \
  --eval_decoder_every "${eval_decoder_every}" \
  --eval_decoder_max_samples "${eval_decoder_max_samples}" \
  --eval_external_max_samples "${eval_external_max_samples}" \
  --lambda_code "${lambda_code}" \
  --lambda_delta "${lambda_delta}" \
  --lambda_recT "${lambda_recT}" \
  --lambda_fc "${lambda_fc}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &

echo "started pid=$! exp_dir=${exp_dir}"
