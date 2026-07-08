#!/bin/bash

# 使用 seed42 transnet_transnet 的 decoder 作为预训练基座。
#
# 默认行为：
#   encoder 从随机初始化开始训练；
#   decoder 从 seed42 transnet_transnet checkpoint 加载；
#   decoder 中只有 --freeze_decoder 指定的层保持可训练，其余冻结。
#
# 示例：
#   seed=2026 gpu=0 encoder=transnet bash scripts/train_seed42_decoder_base.sh
#   seed=3407 gpu=1 encoder=cnn freeze_decoder=fc_decoder bash scripts/train_seed42_decoder_base.sh
#   seed=3407 gpu=1 encoder=transnet freeze_decoder=fc_decoder,ffn bash scripts/train_seed42_decoder_base.sh

set -euo pipefail

# ==============================================================================
# 1. 基础路径
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
encoder=${encoder:-transnet}
decoder=${decoder:-transnet}

# ==============================================================================
# 3. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-0}
seed=${seed:-2026}

# ==============================================================================
# 4. Decoder 基座与冻结配置
# ==============================================================================
pretrained_decoder=${pretrained_decoder:-exps/COST2100/in/base/seed42/transnet_transnet/checkpoints/best_nmse.pth}
freeze_decoder=${freeze_decoder:-fc_decoder,ffn}
freeze_tag=${freeze_decoder//,/_}

exp_name=${exp_name:-COST2100/in/seed42_decoder_base/seed${seed}/${encoder}_${decoder}_train_${freeze_tag}}

if [ ! -f "${pretrained_decoder}" ]; then
  echo "Missing pretrained_decoder: ${pretrained_decoder}" >&2
  exit 1
fi

if [ "${decoder}" != "transnet" ]; then
  echo "This script uses a transnet decoder checkpoint; decoder must be transnet." >&2
  echo "Got decoder=${decoder}" >&2
  exit 1
fi

echo "[RUN] seed=${seed} encoder=${encoder} decoder=${decoder} gpu=${gpu}"
echo "[RUN] pretrained_decoder=${pretrained_decoder}"
echo "[RUN] freeze_decoder=${freeze_decoder}"
echo "[RUN] exp_name=${exp_name}"

# ==============================================================================
# 5. 运行训练
# ==============================================================================
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
  --pretrained_decoder "${pretrained_decoder}" \
  --freeze_decoder "${freeze_decoder}" \
  > /dev/null 2>&1 &
