#!/bin/bash

# 批量跑 mapper 实验，teacher 固定为 seed42 transnet_transnet。
# 每个实验产物保存到 mapper/exps/<mapper>/<source>_to_seed42...。
#
# 默认只启动几个最关键的真实 codeword 测试：
# - 同架构不同 seed：seed2026/seed3407 transnet_transnet
# - 跨架构：seed2026 clnet/crnet/csinet -> seed42 transnet
#
# 用法：
#   bash mapper/scripts/run_mapper.sh
#
# 1ep 真实数据测试：
#   epochs=1 gpu=1 bash mapper/scripts/run_mapper.sh

set -euo pipefail

target_code=${target_code:-exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt}
epochs=${epochs:-400}
gpu=${gpu:-6}
batch_size=${batch_size:-256}
workers=${workers:-0}
mapper=${mapper:-mlp}
lr=${lr:-5e-4}
weight_decay=${weight_decay:-1e-4}
flow_blocks=${flow_blocks:-8}
flow_hidden_dim=${flow_hidden_dim:-1024}
hidden_dim=${hidden_dim:-2048}
num_blocks=${num_blocks:-4}
max_samples=${max_samples:-0}

source_code=exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt \
source_name=seed2026_transnet_transnet \
target_code=${target_code} mapper=${mapper} epochs=${epochs} gpu=${gpu} \
batch_size=${batch_size} workers=${workers} lr=${lr} weight_decay=${weight_decay} \
flow_blocks=${flow_blocks} flow_hidden_dim=${flow_hidden_dim} \
hidden_dim=${hidden_dim} num_blocks=${num_blocks} max_samples=${max_samples} \
bash mapper/scripts/train_mapper.sh

source_code=exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt \
source_name=seed3407_transnet_transnet \
target_code=${target_code} mapper=${mapper} epochs=${epochs} gpu=${gpu} \
batch_size=${batch_size} workers=${workers} lr=${lr} weight_decay=${weight_decay} \
flow_blocks=${flow_blocks} flow_hidden_dim=${flow_hidden_dim} \
hidden_dim=${hidden_dim} num_blocks=${num_blocks} max_samples=${max_samples} \
bash mapper/scripts/train_mapper.sh

source_code=exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt \
source_name=seed2026_clnet_transnet \
target_code=${target_code} mapper=${mapper} epochs=${epochs} gpu=${gpu} \
batch_size=${batch_size} workers=${workers} lr=${lr} weight_decay=${weight_decay} \
flow_blocks=${flow_blocks} flow_hidden_dim=${flow_hidden_dim} \
hidden_dim=${hidden_dim} num_blocks=${num_blocks} max_samples=${max_samples} \
bash mapper/scripts/train_mapper.sh

source_code=exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt \
source_name=seed2026_crnet_transnet \
target_code=${target_code} mapper=${mapper} epochs=${epochs} gpu=${gpu} \
batch_size=${batch_size} workers=${workers} lr=${lr} weight_decay=${weight_decay} \
flow_blocks=${flow_blocks} flow_hidden_dim=${flow_hidden_dim} \
hidden_dim=${hidden_dim} num_blocks=${num_blocks} max_samples=${max_samples} \
bash mapper/scripts/train_mapper.sh

source_code=exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt \
source_name=seed2026_csinet_transnet \
target_code=${target_code} mapper=${mapper} epochs=${epochs} gpu=${gpu} \
batch_size=${batch_size} workers=${workers} lr=${lr} weight_decay=${weight_decay} \
flow_blocks=${flow_blocks} flow_hidden_dim=${flow_hidden_dim} \
hidden_dim=${hidden_dim} num_blocks=${num_blocks} max_samples=${max_samples} \
bash mapper/scripts/train_mapper.sh

