#!/bin/bash

# 训练 teacher-code adapter。
#
# 目标场景：
# - baseline encoder/decoder 训练阶段完全不加 aux_pca/canonical/code_reg 约束。
# - 选择一个固定 target decoder，例如 seed42 的 transnet decoder。
# - 冻结 source encoder、target decoder，只训练 gated_lowrank_affine_mlp adapter。
# - adapter 输出对齐 target decoder 原配 encoder 的 teacher code：
#
#     z_src = E_source(x)
#     z_a   = Adapter(z_src)
#     z_t   = E_target(x)
#     x_hat = D_target(z_a)
#
#   loss = lambda_recon * MSE(x_hat, x)
#        + lambda_code  * MSE(z_a, z_t)
#        + lambda_fc    * MSE(fc_decoder(z_a), fc_decoder(z_t))
#        + lambda_recT  * MSE(D_target(z_a), D_target(z_t))
#        + lambda_teacher_pca    * MSE(PCA(z_a), PCA(z_t))
#        + lambda_teacher_whiten * MSE(Whiten(z_a), Whiten(z_t))
#
# 默认用法：
#   source_seed=2026 target_seed=42 gpu=1 bash scripts/train_teacher_code_adapter.sh
#
# 跨架构 source encoder：
#   source_seed=2026 source_encoder=clnet target_seed=42 target_encoder=transnet \
#     gpu=1 bash scripts/train_teacher_code_adapter.sh
#
# 只做 teacher-code 对齐，不加 PCA：
#   lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0 \
#     source_seed=2026 target_seed=42 gpu=1 bash scripts/train_teacher_code_adapter.sh
#
# 加 teacher-code PCA/whitening 对齐：
#   lambda_teacher_pca=1e-3 teacher_pca_dim=128 \
#     source_seed=2026 target_seed=42 gpu=1 bash scripts/train_teacher_code_adapter.sh
#
#   lambda_teacher_whiten=1e-4 teacher_pca_dim=128 \
#     source_seed=2026 target_seed=42 gpu=1 bash scripts/train_teacher_code_adapter.sh
#
# smoke run：
#   epochs=1 batch_size=4 workers=0 exp_name=tmp/teacher_code_adapter_smoke \
#     source_seed=2026 target_seed=42 gpu=0 bash scripts/train_teacher_code_adapter.sh

set -euo pipefail

train_path=${train_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/cost2100/in_val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/cost2100/in_test.pt}

baseline_root=${baseline_root:-exps/COST2100/in}

source_seed=${source_seed:-2026}
target_seed=${target_seed:-42}
source_encoder=${source_encoder:-transnet}
source_decoder=${source_decoder:-transnet}
target_encoder=${target_encoder:-transnet}
target_decoder=${target_decoder:-transnet}

d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
hidden=${hidden:-16}
num_blocks=${num_blocks:-2}
cr=${cr:-4}

epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-5e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-0}
foreground=${foreground:-0}

adapter_rank=${adapter_rank:-32}
adapter_hidden_dim=${adapter_hidden_dim:-2048}
adapter_gate_init=${adapter_gate_init:-0.1}

lambda_recon=${lambda_recon:-1.0}
lambda_code=${lambda_code:-1e-3}
lambda_fc=${lambda_fc:-1e-2}
lambda_recT=${lambda_recT:-0.0}
lambda_teacher_pca=${lambda_teacher_pca:-0.0}
lambda_teacher_whiten=${lambda_teacher_whiten:-0.0}
teacher_pca_dim=${teacher_pca_dim:-0}
source_exp=${source_exp:-${baseline_root}/seed${source_seed}/${source_encoder}_${source_decoder}}
target_exp=${target_exp:-${baseline_root}/seed${target_seed}/${target_encoder}_${target_decoder}}

pretrained_encoder=${pretrained_encoder:-${source_exp}/checkpoints/best_nmse.pth}
pretrained_decoder=${pretrained_decoder:-${target_exp}/checkpoints/best_nmse.pth}
teacher_code=${teacher_code:-${target_exp}/codewords/train_code.pt}

exp_name=${exp_name:-COST2100/in/teacher_code_adapter/gated_lowrank_affine_mlp/src_seed${source_seed}_${source_encoder}_${source_decoder}_tgt_seed${target_seed}_${target_encoder}_${target_decoder}_code${lambda_code}_fc${lambda_fc}_recT${lambda_recT}_tPCA${lambda_teacher_pca}_tWhite${lambda_teacher_whiten}_tDim${teacher_pca_dim}_lr${lr_init}_ep${epochs}}

if [ ! -f "${pretrained_encoder}" ]; then
  echo "Missing pretrained_encoder: ${pretrained_encoder}" >&2
  exit 1
fi

if [ ! -f "${pretrained_decoder}" ]; then
  echo "Missing pretrained_decoder: ${pretrained_decoder}" >&2
  exit 1
fi

if [ ! -f "${teacher_code}" ]; then
  echo "Missing teacher_code: ${teacher_code}" >&2
  exit 1
fi

extra_args=()
add_arg() {
  local flag=$1
  local val=$2
  [ -n "${val}" ] && extra_args+=("${flag}" "${val}")
}

add_arg --adapter gated_lowrank_affine_mlp
add_arg --adapter_hidden_dim "${adapter_hidden_dim}"
add_arg --adapter_rank "${adapter_rank}"
add_arg --adapter_gate_init "${adapter_gate_init}"
add_arg --pretrained_encoder "${pretrained_encoder}"
add_arg --pretrained_decoder "${pretrained_decoder}"
add_arg --teacher_code "${teacher_code}"
add_arg --lambda_recon "${lambda_recon}"
add_arg --lambda_code "${lambda_code}"
add_arg --lambda_fc "${lambda_fc}"
add_arg --lambda_recT "${lambda_recT}"
add_arg --lambda_teacher_pca "${lambda_teacher_pca}"
add_arg --lambda_teacher_whiten "${lambda_teacher_whiten}"
add_arg --teacher_pca_dim "${teacher_pca_dim}"

echo "Training teacher-code adapter:"
echo "  source encoder checkpoint: ${pretrained_encoder}"
echo "  target decoder checkpoint: ${pretrained_decoder}"
echo "  target teacher code: ${teacher_code}"
echo "  adapter: gated_lowrank_affine_mlp rank=${adapter_rank} hidden=${adapter_hidden_dim} gate=${adapter_gate_init}"
echo "  lambda_recon/code/fc/recT/teacher_pca/teacher_whiten: ${lambda_recon}/${lambda_code}/${lambda_fc}/${lambda_recT}/${lambda_teacher_pca}/${lambda_teacher_whiten}"
echo "  teacher_pca_dim: ${teacher_pca_dim}"
echo "  exp_name: ${exp_name}"

cmd=(
python -u main.py
  --exp_name "${exp_name}" \
  --train_path "${train_path}" \
  --val_path "${val_path}" \
  --test_path "${test_path}" \
  --epochs "${epochs}" \
  --d_model "${d_model}" \
  --nt "${nt}" \
  --nc "${nc}" \
  --dim_feedforward "${dim_feedforward}" \
  --hidden "${hidden}" \
  --num_blocks "${num_blocks}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --cr "${cr}" \
  --encoder "${source_encoder}" \
  --decoder "${target_decoder}" \
  --scheduler "${scheduler}" \
  --lr_init "${lr_init}" \
  --weight_decay "${weight_decay}" \
  --gpu "${gpu}" \
  --seed "${source_seed}" \
  "${extra_args[@]}"
)

if [ "${foreground}" = "1" ]; then
  "${cmd[@]}"
else
  "${cmd[@]}" > /dev/null 2>&1 &
fi
