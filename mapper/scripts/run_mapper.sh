#!/bin/bash

# 显式运行 affine + mapper 实验。
#
# 结构：
#   start = affine(z_source)
#   pred  = start + flow(start)

set -euo pipefail

target_code=exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt \
source_name=seed2026_clnet_transnet \
source_code=exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt \
mapper=flow \
residual_mapping=1 \
align_mode=affine \
residual_condition=start \
residual_scale=1.0 \
lr=5e-4 \
eta_min=5e-5 \
epochs=400 \
batch_size=1024 \
gpu=0 \
bash mapper/scripts/train_mapper.sh

target_code=exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt \
source_name=seed2026_crnet_transnet \
source_code=exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt \
mapper=flow \
residual_mapping=1 \
align_mode=affine \
residual_condition=start \
residual_scale=1.0 \
lr=5e-4 \
eta_min=5e-5 \
epochs=400 \
batch_size=1024 \
gpu=6 \
bash mapper/scripts/train_mapper.sh

target_code=exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt \
source_name=seed2026_csinet_transnet \
source_code=exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt \
mapper=flow \
residual_mapping=1 \
align_mode=affine \
residual_condition=start \
residual_scale=1.0 \
lr=5e-4 \
eta_min=5e-5 \
epochs=400 \
batch_size=1024 \
gpu=6 \
bash mapper/scripts/train_mapper.sh

target_code=exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt \
source_name=seed2026_transnet_transnet \
source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
mapper=flow \
residual_mapping=1 \
align_mode=affine \
residual_condition=start \
residual_scale=1.0 \
lr=5e-4 \
eta_min=5e-5 \
epochs=400 \
batch_size=1024 \
gpu=7 \
bash mapper/scripts/train_mapper.sh