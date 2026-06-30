#!/bin/bash

# 跑 teacher-code adapter 跨架构实验需要的普通 baseline checkpoint。
#
# 目标输出：
#   exps/COST2100/in/seed2026/clnet_transnet/checkpoints/best_nmse.pth
#   exps/COST2100/in/seed2026/crnet_transnet/checkpoints/best_nmse.pth
#   exps/COST2100/in/seed2026/csinet_transnet/checkpoints/best_nmse.pth
#
# 这些是无 aux_pca、无 canonical、无 adapter 的普通 encoder/decoder 训练。
#
# 使用：
#   bash scripts/run.sh
#
# 常用覆盖：
#   seed=2026 bash scripts/run.sh
#   epochs=400 batch_size=256 lr_init=2e-4 gpu=4 bash scripts/run.sh

set -euo pipefail

seed=${seed:-2026}
decoder=${decoder:-transnet}
epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
# gpu=${gpu:-6}

encoder=clnet decoder=${decoder} seed=${seed} \
epochs=${epochs} batch_size=${batch_size} workers=${workers} \
lr_init=${lr_init} weight_decay=${weight_decay} gpu=1 \
exp_name=COST2100/in/seed${seed}/clnet_${decoder} \
bash scripts/train.sh

encoder=crnet decoder=${decoder} seed=${seed} \
epochs=${epochs} batch_size=${batch_size} workers=${workers} \
lr_init=${lr_init} weight_decay=${weight_decay} gpu=4 \
exp_name=COST2100/in/seed${seed}/crnet_${decoder} \
bash scripts/train.sh

encoder=csinet decoder=${decoder} seed=${seed} \
epochs=${epochs} batch_size=${batch_size} workers=${workers} \
lr_init=${lr_init} weight_decay=${weight_decay} gpu=6 \
exp_name=COST2100/in/seed${seed}/csinet_${decoder} \
bash scripts/train.sh
