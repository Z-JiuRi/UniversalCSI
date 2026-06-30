#!/bin/bash

# 批量启动 teacher-code anchored adapter 实验。
#
# 所有命令都保持“参数=值 bash scripts/train_teacher_code_adapter.sh”的形式。
# baseline encoder/decoder 训练阶段不加 aux_pca/canonical/code_reg 约束；
# 这里只在 adapter 训练阶段用 target decoder 的 teacher_code/fc/recT 做对齐。
#
# 默认固定 target decoder：
#   seed42 transnet_transnet
#
# 默认 source：
#   普通 baseline 实验目录 exps/COST2100/in/seed*/encoder_decoder
#
# 使用方式：
#   bash scripts/run_teacher_code_adapter.sh
#
# 常用覆盖：
#   gpu=1 bash scripts/run_teacher_code_adapter.sh
#   epochs=200 gpu=4 bash scripts/run_teacher_code_adapter.sh
#   batch_size=128 lr_init=2e-4 bash scripts/run_teacher_code_adapter.sh

set -euo pipefail

# gpu=${gpu:-6}
epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
lr_init=${lr_init:-5e-4}
weight_decay=${weight_decay:-1e-3}

target_seed=${target_seed:-42}
target_encoder=${target_encoder:-transnet}
target_decoder=${target_decoder:-transnet}

adapter_rank=${adapter_rank:-32}
adapter_hidden_dim=${adapter_hidden_dim:-2048}
adapter_gate_init=${adapter_gate_init:-0.1}
teacher_pca_dim=${teacher_pca_dim:-128}

# # 1. 同架构 transnet，不同 seed，主配置：recon + teacher code + fc。
# source_seed=2026 source_encoder=transnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh

# source_seed=3407 source_encoder=transnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh

# # 2. 同架构 transnet，测试 teacher reconstruction consistency。
# source_seed=2026 source_encoder=transnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=1.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh

# # 3. 同架构 transnet，弱化 fc，只看 code 对齐是否足够。
# source_seed=2026 source_encoder=transnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=0.0 lambda_recT=0.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh

# # 4. 同架构 transnet，强一点 code 对齐。
# source_seed=2026 source_encoder=transnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-2 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh

# 5. teacher-code PCA/whitening：对 target decoder 原配 teacher code 的主方向或 whiten 后方向加权对齐。
source_seed=2026 source_encoder=transnet source_decoder=transnet \
target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=4 \
lr_init=${lr_init} weight_decay=${weight_decay} \
adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
lambda_teacher_pca=1e-3 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
bash scripts/train_teacher_code_adapter.sh

source_seed=2026 source_encoder=transnet source_decoder=transnet \
target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=1 \
lr_init=${lr_init} weight_decay=${weight_decay} \
adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
lambda_teacher_pca=0.0 lambda_teacher_whiten=1e-4 teacher_pca_dim=${teacher_pca_dim} \
bash scripts/train_teacher_code_adapter.sh

source_seed=2026 source_encoder=transnet source_decoder=transnet \
target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=6 \
lr_init=${lr_init} weight_decay=${weight_decay} \
adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
lambda_teacher_pca=1e-3 lambda_teacher_whiten=1e-4 teacher_pca_dim=${teacher_pca_dim} \
bash scripts/train_teacher_code_adapter.sh

# # 6. adapter 侧 raw CSI aux_pca 对照：不约束 baseline encoder，只在 adapter 输出端加弱 PCA anchor。
# source_seed=2026 source_encoder=transnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=1e-4 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh

# 7. 跨架构 source encoder。
# 当前原始 baseline 目录里未必已经有这些 checkpoint：
#   exps/COST2100/in/seed2026/clnet_transnet/checkpoints/best_nmse.pth
#   exps/COST2100/in/seed2026/crnet_transnet/checkpoints/best_nmse.pth
#   exps/COST2100/in/seed2026/csinet_transnet/checkpoints/best_nmse.pth
# 如果后续补跑了这些 baseline encoder/decoder，取消下面注释即可。
#
# source_seed=2026 source_encoder=clnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh
#
# source_seed=2026 source_encoder=crnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh
#
# source_seed=2026 source_encoder=csinet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=0.0 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh
#
# 8. 跨架构 + adapter 侧 PCA 对照。
#
# source_seed=2026 source_encoder=clnet source_decoder=transnet \
# target_seed=${target_seed} target_encoder=${target_encoder} target_decoder=${target_decoder} \
# epochs=${epochs} batch_size=${batch_size} workers=${workers} gpu=${gpu} \
# lr_init=${lr_init} weight_decay=${weight_decay} \
# adapter_rank=${adapter_rank} adapter_hidden_dim=${adapter_hidden_dim} adapter_gate_init=${adapter_gate_init} \
# lambda_recon=1.0 lambda_code=1e-3 lambda_fc=1e-2 lambda_recT=0.0 lambda_pca=1e-4 \
# lambda_teacher_pca=0.0 lambda_teacher_whiten=0.0 teacher_pca_dim=${teacher_pca_dim} \
# bash scripts/train_teacher_code_adapter.sh
