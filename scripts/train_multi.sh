#!/bin/bash
#
# 多实验批量训练 — 一次读取 CSI，所有 worker 共享数据。
# 参数=值 形式传参，和 scripts/train.sh 风格一致。
#
# 用法：
#   seed_list="0,223" encoder_list="csinet,cnn" ... bash scripts/train_multi.sh
#
# 所有列表长度必须严格一致，长度 = 实验数。

set -euo pipefail

# ==============================================================================
# 1. 基础路径（带默认值）
# ==============================================================================
train_path=${train_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/cost2100/in_val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/cost2100/in_test.pt}

# ==============================================================================
# 2. 模型结构与数据维度参数
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
hidden=${hidden:-16}
num_blocks=${num_blocks:-2}
cr=${cr:-4}

# ==============================================================================
# 3. 训练超参数与实验列表
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}

# 实验列表（必须长度一致）
seed_list=${seed_list:-}
encoder_list=${encoder_list:-}
decoder_list=${decoder_list:-}
gpu_list=${gpu_list:-}

if [ -z "$seed_list" ] || [ -z "$encoder_list" ] || [ -z "$decoder_list" ] || [ -z "$gpu_list" ]; then
  echo "ERROR: seed_list, encoder_list, decoder_list, gpu_list must all be set" >&2
  exit 1
fi

# ==============================================================================
# 4. 运行 Python 脚本
# ==============================================================================
python -u main.py \
  --train_path "${train_path}" \
  --val_path "${val_path}" \
  --test_path "${test_path}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --epochs "${epochs}" \
  --cr "${cr}" \
  --d_model "${d_model}" \
  --nt "${nt}" \
  --nc "${nc}" \
  --dim_feedforward "${dim_feedforward}" \
  --hidden "${hidden}" \
  --num_blocks "${num_blocks}" \
  --scheduler "${scheduler}" \
  --lr_init "${lr_init}" \
  --weight_decay "${weight_decay}" \
  --seed_list "${seed_list}" \
  --encoder_list "${encoder_list}" \
  --decoder_list "${decoder_list}" \
  --gpu_list "${gpu_list}" \
  > /dev/null 2>&1 &
