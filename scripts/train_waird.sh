#!/bin/bash

# encoder=transnet decoder=transnet batch_size=256 epochs=400 gpu=6 seed=42 bash scripts/train_waird.sh



# ==============================================================================
# 1. 基础路径与实验名称（环境变量传参，带默认值）
# ==============================================================================
train_path=${train_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/UniversalCSI/test.pt}

# ==============================================================================
# 2. 模型结构与数据维度参数
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
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
seed=${seed:-42}
exp_name=${exp_name:-WAIRD/seed${seed}/${encoder}_${decoder}}

# ==============================================================================
# 4. 运行 Python 脚本
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
  > /dev/null 2>&1 &
