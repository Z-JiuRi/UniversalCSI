#!/bin/bash

# 训练 codeword mapper：只做 z_source -> z_teacher 的码字 MSE 映射。
#
# 默认 teacher:
#   exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt
#
# 默认 source:
#   exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt
#
# 用法：
#   bash mapper/scripts/train_mapper.sh
#
# 常用覆盖：
#   mapper=delta_mlp residual_mapping=1 align_mode=affine source_name=seed2026_transnet gpu=1 bash mapper/scripts/train_mapper.sh
#   mapper=hybrid_flow_mlp residual_mapping=0 epochs=400 gpu=1 bash mapper/scripts/train_mapper.sh
#   lambda_cos=1e-3 lambda_cov=1e-5 bash mapper/scripts/train_mapper.sh
#   lambda_smoothl1=0.5 lambda_sample_tail=0.1 lambda_dim_tail=0.05 lambda_whiten=1e-4 bash mapper/scripts/train_mapper.sh
#   lambda_recT=1.0 lambda_rec=1.0 lambda_fc=1e-2 lambda_decoder_tail=0.1 bash mapper/scripts/train_mapper.sh
#   eval_decoder_every=20 eval_decoder_max_samples=0 bash mapper/scripts/train_mapper.sh

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
source_code=${source_code:-exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt}
source_name=${source_name:-seed2026_transnet_transnet}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}
csi_path=${csi_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}

mapper=${mapper:-delta_mlp}
epochs=${epochs:-400}
batch_size=${batch_size:-512}
workers=${workers:-0}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
scheduler=${scheduler:-cosine}
eta_min=${eta_min:-5e-5}
hidden_dim=${hidden_dim:-2048}
num_blocks=${num_blocks:-4}
flow_hidden_dim=${flow_hidden_dim:-1024}
flow_blocks=${flow_blocks:-8}
flow_clamp=${flow_clamp:-0.1}
dropout=${dropout:-0.0}
residual_mapping=${residual_mapping:-1}
align_mode=${align_mode:-affine}
align_ridge=${align_ridge:-1e-4}
residual_condition=${residual_condition:-source_start}
residual_scale=${residual_scale:-1.0}
val_ratio=${val_ratio:-0}
max_samples=${max_samples:-0}
save_last=${save_last:-0}
lambda_cos=${lambda_cos:-0.0}
lambda_cov=${lambda_cov:-0.0}
lambda_smoothl1=${lambda_smoothl1:-0.0}
smoothl1_beta=${smoothl1_beta:-0.05}
lambda_sample_tail=${lambda_sample_tail:-0.0}
sample_tail_ratio=${sample_tail_ratio:-0.2}
lambda_dim_tail=${lambda_dim_tail:-0.0}
dim_tail_ratio=${dim_tail_ratio:-0.05}
lambda_whiten=${lambda_whiten:-0.0}
whiten_eps_ratio=${whiten_eps_ratio:-1e-3}
lambda_rec=${lambda_rec:-0.0}
lambda_recT=${lambda_recT:-0.0}
lambda_fc=${lambda_fc:-0.0}
lambda_decoder_tail=${lambda_decoder_tail:-0.0}
decoder_tail_ratio=${decoder_tail_ratio:-0.2}
eval_decoder_every=${eval_decoder_every:-20}
eval_decoder_max_samples=${eval_decoder_max_samples:-0}
gpu=${gpu:-0}
seed=${seed:-2026}
cpu=${cpu:-0}

align_tag="direct"
if [ "${residual_mapping}" = "1" ]; then
  align_tag="align${align_mode}_cond${residual_condition}_scale${residual_scale}"
fi
loss_tag="sl1${lambda_smoothl1}_st${lambda_sample_tail}r${sample_tail_ratio}_dt${lambda_dim_tail}r${dim_tail_ratio}_white${lambda_whiten}_rec${lambda_rec}_recT${lambda_recT}_fc${lambda_fc}_decTail${lambda_decoder_tail}r${decoder_tail_ratio}_cos${lambda_cos}_cov${lambda_cov}"
exp_dir=${exp_dir:-mapper/exps/${mapper}/${align_tag}/${source_name}_to_seed42_transnet_${loss_tag}_lr${lr}_ep${epochs}}

mkdir -p "${exp_dir}"

if [ ! -f "${source_code}" ]; then
  echo "Missing source_code: ${source_code}" >&2
  exit 1
fi
if [ ! -f "${target_code}" ]; then
  echo "Missing target_code: ${target_code}" >&2
  exit 1
fi
need_decoder=0
if [ "${lambda_rec}" != "0.0" ] || [ "${lambda_recT}" != "0.0" ] || \
   [ "${lambda_fc}" != "0.0" ] || [ "${lambda_decoder_tail}" != "0.0" ] || \
   [ "${eval_decoder_every}" != "0" ]; then
  need_decoder=1
fi
if [ "${need_decoder}" = "1" ]; then
  if [ ! -f "${decoder_checkpoint}" ]; then
    echo "Missing decoder_checkpoint: ${decoder_checkpoint}" >&2
    exit 1
  fi
  if [ ! -f "${decoder_args_json}" ]; then
    echo "Missing decoder_args_json: ${decoder_args_json}" >&2
    exit 1
  fi
  if [ ! -f "${csi_path}" ]; then
    echo "Missing csi_path: ${csi_path}" >&2
    exit 1
  fi
fi

extra_args=()
if [ "${cpu}" = "1" ]; then
  extra_args+=(--cpu)
fi
if [ "${save_last}" = "1" ]; then
  extra_args+=(--save_last)
fi
if [ "${residual_mapping}" = "1" ]; then
  extra_args+=(--residual_mapping)
fi
if [ "${cpu}" != "1" ]; then
  export CUDA_VISIBLE_DEVICES="${gpu}"
fi

python -u mapper/train_mapper.py \
  --source_code "${source_code}" \
  --target_code "${target_code}" \
  --exp_dir "${exp_dir}" \
  --mapper "${mapper}" \
  --epochs "${epochs}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --lr "${lr}" \
  --weight_decay "${weight_decay}" \
  --scheduler "${scheduler}" \
  --eta_min "${eta_min}" \
  --hidden_dim "${hidden_dim}" \
  --num_blocks "${num_blocks}" \
  --flow_hidden_dim "${flow_hidden_dim}" \
  --flow_blocks "${flow_blocks}" \
  --flow_clamp "${flow_clamp}" \
  --dropout "${dropout}" \
  --align_mode "${align_mode}" \
  --align_ridge "${align_ridge}" \
  --residual_condition "${residual_condition}" \
  --residual_scale "${residual_scale}" \
  --val_ratio "${val_ratio}" \
  --max_samples "${max_samples}" \
  --lambda_cos "${lambda_cos}" \
  --lambda_cov "${lambda_cov}" \
  --lambda_smoothl1 "${lambda_smoothl1}" \
  --smoothl1_beta "${smoothl1_beta}" \
  --lambda_sample_tail "${lambda_sample_tail}" \
  --sample_tail_ratio "${sample_tail_ratio}" \
  --lambda_dim_tail "${lambda_dim_tail}" \
  --dim_tail_ratio "${dim_tail_ratio}" \
  --lambda_whiten "${lambda_whiten}" \
  --whiten_eps_ratio "${whiten_eps_ratio}" \
  --lambda_rec "${lambda_rec}" \
  --lambda_recT "${lambda_recT}" \
  --lambda_fc "${lambda_fc}" \
  --lambda_decoder_tail "${lambda_decoder_tail}" \
  --decoder_tail_ratio "${decoder_tail_ratio}" \
  --decoder_checkpoint "${decoder_checkpoint}" \
  --decoder_args_json "${decoder_args_json}" \
  --csi_path "${csi_path}" \
  --eval_decoder_every "${eval_decoder_every}" \
  --eval_decoder_max_samples "${eval_decoder_max_samples}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &
