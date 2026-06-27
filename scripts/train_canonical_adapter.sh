#!/bin/bash

# 在规范化 encoder/decoder checkpoint 之间训练 adapter。
#
# 这个脚本做的事情：
# - 构建一个 transnet_transnet 自编码模型，并在 encoder 和 decoder 中间插入 adapter。
# - 从规范化自编码实验中加载 --pretrained_encoder 和 --pretrained_decoder。
# - encoder 和 decoder 会在 utils/init.py 中被冻结。
# - 训练时只有 adapter 参数可训练。
#
# 默认用法：
#   seed=2026 decoder_seed=42 gpu=1 bash scripts/train_canonical_adapter.sh
#
# 默认路径展开后是：
#   encoder checkpoint：
#     exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed2026_transnet_transnet/checkpoints/best_nmse.pth
#   decoder checkpoint：
#     exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed42_transnet_transnet/checkpoints/best_nmse.pth
#   teacher code，仅在 lambda_code > 0 时使用：
#     exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed42_transnet_transnet/codewords/train_code.pt
#
# 1. 基础 aux_pca_1e-3 adapter 测试，只用重建损失：
  # seed=2026 decoder_seed=42 lambda_recon=1.0 lambda_code=0.0 gpu=1 \
  #   bash scripts/train_canonical_adapter.sh
#
# 2. 同样的测试，但额外用 decoder seed 的码字做 code 对齐：
# seed=2026 decoder_seed=42 lambda_recon=1.0 lambda_code=1e-2 gpu=1 lr_init=1e-3 \
# bash scripts/train_canonical_adapter.sh
#
# 3. 换另一个 encoder seed，仍然接同一个 decoder seed：
#   seed=1024 decoder_seed=42 gpu=1 bash scripts/train_canonical_adapter.sh
#
# 4. 手动 sweep 多个 encoder seed：
#   for s in 42 796 1024 2026 3407; do
#     seed=${s} decoder_seed=42 gpu=0 bash scripts/train_canonical_adapter.sh
#   done
#
# 5. 修改 adapter 类型或隐藏层维度：
#   adapter=mlp adapter_hidden_dim=1024 seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
#   adapter=transformer adapter_hidden_dim=256 seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
# 6. 使用 aux_dct_1e-3 规范化 checkpoint：
#   canonical_scheme=aux_dct_1e-3 seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
# 7. 使用 fixed Q checkpoint。加载这类 checkpoint 时，必须使用相同的
#    canonical head 结构：
#   canonical_scheme=fixed_q seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
# 8. 使用 fixed Q + 低秩残差 checkpoint：
#   canonical_scheme=fixed_q_rank16 canonical_lowrank_rank=16 \
#     canonical_lowrank_scale=0.05 seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
# 9. 使用 fixed Q + 低秩残差 + PCA checkpoint：
#   canonical_scheme=fixed_q_rank16_pca canonical_lowrank_rank=16 \
#     canonical_lowrank_scale=0.05 seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
# 10. 使用 codebook checkpoint：
#   canonical_scheme=codebook1024 canonical_codebook_size=1024 \
#     canonical_codebook_temperature=1.0 seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
# 11. 使用 codebook + DCT/code 正则 checkpoint：
#   canonical_scheme=codebook1024_dct_reg canonical_codebook_size=1024 \
#     seed=2026 decoder_seed=42 gpu=0 bash scripts/train_canonical_adapter.sh
#
# 12. 和同一 canonical 实验目录结构下的无规范化 baseline checkpoint 对比：
#   canonical_scheme=baseline seed=2026 decoder_seed=42 gpu=0 \
#     bash scripts/train_canonical_adapter.sh
#
# 13. 手动覆盖 checkpoint 路径。适合测试自定义 encoder/decoder 组合，
#     或者测试没有按默认目录组织的实验：
#   pretrained_encoder=exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed2026_transnet_transnet/checkpoints/best_nmse.pth \
#   pretrained_decoder=exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed42_transnet_transnet/checkpoints/best_nmse.pth \
#   teacher_code=exps/COST2100/in/encoder_canonical/aux_pca_1e-3/seed42_transnet_transnet/codewords/train_code.pt \
#   seed=2026 decoder_seed=42 gpu=0 bash scripts/train_canonical_adapter.sh
#
# 14. 手动覆盖输出实验名：
#   exp_name=COST2100/in/encoder_canonical_adapter/debug_aux_pca_seed2026_to_42 \
#     seed=2026 decoder_seed=42 gpu=0 bash scripts/train_canonical_adapter.sh
#
# 15. 短 smoke run，用来检查脚本和 checkpoint 是否能正常跑通：
#   epochs=1 batch_size=4 workers=0 seed=2026 decoder_seed=42 gpu=0 \
#     exp_name=tmp/canonical_adapter_smoke bash scripts/train_canonical_adapter.sh
#
# 常用参数：
#   canonical_scheme:
#     baseline | aux_pca_1e-3 | aux_dct_1e-3 | fixed_q |
#     fixed_q_rank16 | fixed_q_rank16_code_reg | fixed_q_rank16_pca |
#     codebook1024 | codebook1024_dct_reg
#
#   seed:
#     加载哪个 seed 的规范化 encoder checkpoint。
#
#   decoder_seed:
#     加载哪个 seed 的规范化 decoder checkpoint。默认 teacher_code 也来自这个 seed。
#
#   lambda_recon:
#     重建损失权重。做 adapter 性能测试时通常保持 1.0。
#
#   lambda_code:
#     code 对齐损失权重。0.0 表示不加载 teacher_code；非 0 时要求 teacher_code 存在。
#
#   canonical_head:
#     通常不要手动设置，会根据 canonical_scheme 自动推断。
#     aux_pca_1e-3 和 aux_dct_1e-3 使用 canonical_head=none，因为 PCA/DCT 只是
#     源自编码器训练时的辅助目标，不是模型结构的一部分。
#
# 注意：
# - 模型结构参数必须和源 checkpoint 的结构一致。对于 fixed_q/codebook 方案，
#   canonical_anchor_seed 和 head 相关参数必须和原 canonical 训练实验保持一致。
# - 这个脚本和现有训练脚本一样，会在后台启动 main.py，并把 stdout/stderr 重定向到
#   /dev/null。进度和错误请看 exps/${exp_name}/run.log。

set -euo pipefail

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
decoder_seed=${decoder_seed:-42}

adapter=${adapter:-mlp}
adapter_hidden_dim=${adapter_hidden_dim:-2048}
lambda_recon=${lambda_recon:-1.0}
lambda_code=${lambda_code:-0.0}

canonical_scheme=${canonical_scheme:-aux_pca_1e-3}
canonical_root=${canonical_root:-exps/COST2100/in/encoder_canonical/${canonical_scheme}}
canonical_anchor_seed=${canonical_anchor_seed:-0}
canonical_lowrank_rank=${canonical_lowrank_rank:-16}
canonical_lowrank_scale=${canonical_lowrank_scale:-0.05}
canonical_codebook_size=${canonical_codebook_size:-1024}
canonical_codebook_temperature=${canonical_codebook_temperature:-1.0}

canonical_head=${canonical_head:-none}
case "${canonical_scheme}" in
  fixed_q)
    canonical_head=fixed_q_lowrank
    canonical_lowrank_rank=0
    canonical_lowrank_scale=0.0
    ;;
  fixed_q_rank16|fixed_q_rank16_code_reg|fixed_q_rank16_pca)
    canonical_head=fixed_q_lowrank
    ;;
  codebook1024|codebook1024_dct_reg)
    canonical_head=codebook
    ;;
  baseline|aux_pca_1e-3|aux_dct_1e-3)
    canonical_head=none
    ;;
esac

pretrained_encoder=${pretrained_encoder:-${canonical_root}/seed${seed}_${encoder}_${decoder}/checkpoints/best_nmse.pth}
pretrained_decoder=${pretrained_decoder:-${canonical_root}/seed${decoder_seed}_${encoder}_${decoder}/checkpoints/best_nmse.pth}
teacher_code=${teacher_code:-${canonical_root}/seed${decoder_seed}_${encoder}_${decoder}/codewords/train_code.pt}

exp_name=${exp_name:-COST2100/in/encoder_canonical/adapter/${canonical_scheme}/${adapter}/enc_seed${seed}_dec_seed${decoder_seed}_recon${lambda_recon}_code${lambda_code}_lr${lr_init}}

if [ ! -f "${pretrained_encoder}" ]; then
  echo "Missing pretrained_encoder: ${pretrained_encoder}" >&2
  exit 1
fi

if [ ! -f "${pretrained_decoder}" ]; then
  echo "Missing pretrained_decoder: ${pretrained_decoder}" >&2
  exit 1
fi

if [ "${lambda_code}" != "0" ] && [ "${lambda_code}" != "0.0" ] && [ ! -f "${teacher_code}" ]; then
  echo "Missing teacher_code for lambda_code=${lambda_code}: ${teacher_code}" >&2
  exit 1
fi

extra_args=()
add_arg() { local flag=$1 val=$2; [ -n "${val}" ] && extra_args+=("${flag}" "${val}"); }

add_arg --adapter "${adapter}"
add_arg --adapter_hidden_dim "${adapter_hidden_dim}"
add_arg --pretrained_encoder "${pretrained_encoder}"
add_arg --pretrained_decoder "${pretrained_decoder}"
add_arg --lambda_recon "${lambda_recon}"
add_arg --lambda_code "${lambda_code}"

if [ "${lambda_code}" != "0" ] && [ "${lambda_code}" != "0.0" ]; then
  add_arg --teacher_code "${teacher_code}"
fi

add_arg --canonical_head "${canonical_head}"
add_arg --canonical_anchor_seed "${canonical_anchor_seed}"
add_arg --canonical_lowrank_rank "${canonical_lowrank_rank}"
add_arg --canonical_lowrank_scale "${canonical_lowrank_scale}"
add_arg --canonical_codebook_size "${canonical_codebook_size}"
add_arg --canonical_codebook_temperature "${canonical_codebook_temperature}"

echo "Training canonical adapter:"
echo "  encoder checkpoint: ${pretrained_encoder}"
echo "  decoder checkpoint: ${pretrained_decoder}"
echo "  teacher code: ${teacher_code}"
echo "  canonical_head: ${canonical_head}"
echo "  exp_name: ${exp_name}"

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
