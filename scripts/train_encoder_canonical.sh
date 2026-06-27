#!/bin/bash

# Example commands:
#
# 0. Baseline through this script, no canonical constraint:
#   scheme=none seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/baseline/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 1. Fixed random row-orthogonal Q only:
#    features(2048) -> fixed Q(anchor_seed=0) -> code(512)
#   scheme=fixed_q_lowrank canonical_lowrank_rank=0 canonical_anchor_seed=0 \
#     seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/fixed_q/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 2. Fixed Q + low-rank residual:
#    code = features @ Q.T + residual_scale * features @ U @ V
#   scheme=fixed_q_lowrank canonical_anchor_seed=0 \
#     canonical_lowrank_rank=16 canonical_lowrank_scale=0.05 \
#     seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/fixed_q_rank16/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 3. Fixed random codebook:
#    features -> assignment logits -> softmax -> fixed codebook convex combination
#   scheme=codebook canonical_anchor_seed=0 \
#     canonical_codebook_size=1024 canonical_codebook_temperature=1.0 \
#     seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/codebook1024/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 4. Raw-CSI PCA auxiliary target:
#    z_anchor = PCA(raw CSI flatten), loss += lambda_anchor * dist(z, z_anchor)
#   scheme=aux_pca lambda_anchor=1e-3 anchor_loss=mse \
#     seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/aux_pca_1e-3/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 5. Raw-CSI DCT auxiliary target:
#    z_anchor = fixed 2D-DCT(raw CSI), loss += lambda_anchor * dist(z, z_anchor)
#   scheme=aux_dct lambda_anchor=1e-3 anchor_loss=mse \
#     seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/aux_dct_1e-3/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 6. Fixed Q + low-rank residual + code statistics regularization:
#   scheme=fixed_q_lowrank canonical_anchor_seed=0 \
#     canonical_lowrank_rank=16 canonical_lowrank_scale=0.05 \
#     lambda_code_mean=1e-4 lambda_code_var=1e-4 lambda_code_cov=1e-4 \
#     code_var_tau=256 seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/fixed_q_rank16_code_reg/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 7. Fixed Q + low-rank residual + PCA auxiliary target:
#   scheme=fixed_q_lowrank canonical_anchor_seed=0 \
#     canonical_lowrank_rank=16 canonical_lowrank_scale=0.05 \
#     anchor_target=pca lambda_anchor=1e-3 anchor_loss=mse \
#     seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/fixed_q_rank16_pca/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# 8. Fixed codebook + DCT auxiliary target + weak code regularization:
#   scheme=codebook canonical_anchor_seed=0 canonical_codebook_size=1024 \
#     anchor_target=dct lambda_anchor=1e-3 anchor_loss=mse \
#     lambda_code_mean=1e-4 lambda_code_cov=1e-4 \
#     seed=2026 gpu=0 \
#     exp_name=COST2100/in/encoder_canonical/codebook1024_dct_reg/seed2026_transnet_transnet \
#     bash scripts/train_encoder_canonical.sh
#
# Notes:
# - canonical_anchor_seed is independent of the training seed. Keep it fixed
#   at 0 across all seed experiments to share the same Q/codebook.
# - canonical_head currently supports encoder=transnet only.

train_path=${train_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/cost2100/in_val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/cost2100/in_test.pt}

d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
hidden=${hidden:-16}
num_blocks=${num_blocks:-2}
cr=${cr:-4}
encoder=${encoder:-transnet}
decoder=${decoder:-transnet}

epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-0}
seed=${seed:-2026}

scheme=${scheme:-fixed_q_lowrank}
canonical_anchor_seed=${canonical_anchor_seed:-0}
canonical_lowrank_rank=${canonical_lowrank_rank:-16}
canonical_lowrank_scale=${canonical_lowrank_scale:-0.05}
canonical_codebook_size=${canonical_codebook_size:-1024}
canonical_codebook_temperature=${canonical_codebook_temperature:-1.0}

anchor_target=${anchor_target:-none}
lambda_anchor=${lambda_anchor:-0}
anchor_loss=${anchor_loss:-mse}

lambda_code_mean=${lambda_code_mean:-0}
lambda_code_var=${lambda_code_var:-0}
lambda_code_cov=${lambda_code_cov:-0}
lambda_code_l1=${lambda_code_l1:-0}
code_var_tau=${code_var_tau:-256}

pretrained_encoder=${pretrained_encoder:-}
pretrained_decoder=${pretrained_decoder:-}
teacher_code=${teacher_code:-}
lambda_recon=${lambda_recon:-1.0}
lambda_code=${lambda_code:-0.0}

canonical_head=none
case "$scheme" in
  fixed_q_lowrank)
    canonical_head=fixed_q_lowrank
    ;;
  codebook)
    canonical_head=codebook
    ;;
  aux_pca)
    anchor_target=pca
    ;;
  aux_dct)
    anchor_target=dct
    ;;
  none)
    ;;
  *)
    echo "Unknown scheme: $scheme" >&2
    exit 1
    ;;
esac

exp_name=${exp_name:-COST2100/in/encoder_canonical/${scheme}/seed${seed}_${encoder}_${decoder}}

extra_args=()
add_arg() { local flag=$1 val=$2; [ -n "$val" ] && extra_args+=("$flag" "$val"); }

add_arg --pretrained_encoder "$pretrained_encoder"
add_arg --pretrained_decoder "$pretrained_decoder"
add_arg --teacher_code "$teacher_code"

add_arg --canonical_head "$canonical_head"
add_arg --canonical_anchor_seed "$canonical_anchor_seed"
add_arg --canonical_lowrank_rank "$canonical_lowrank_rank"
add_arg --canonical_lowrank_scale "$canonical_lowrank_scale"
add_arg --canonical_codebook_size "$canonical_codebook_size"
add_arg --canonical_codebook_temperature "$canonical_codebook_temperature"

add_arg --anchor_target "$anchor_target"
add_arg --lambda_anchor "$lambda_anchor"
add_arg --anchor_loss "$anchor_loss"

add_arg --lambda_recon "$lambda_recon"
add_arg --lambda_code "$lambda_code"
add_arg --lambda_code_mean "$lambda_code_mean"
add_arg --lambda_code_var "$lambda_code_var"
add_arg --lambda_code_cov "$lambda_code_cov"
add_arg --lambda_code_l1 "$lambda_code_l1"
add_arg --code_var_tau "$code_var_tau"

python -u main.py \
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
  --encoder "${encoder}" \
  --decoder "${decoder}" \
  --scheduler "${scheduler}" \
  --lr_init "${lr_init}" \
  --weight_decay "${weight_decay}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &
