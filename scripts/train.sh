#!/bin/bash

# ==============================================================================
# 1. 基础路径与实验名称（环境变量传参，带默认值）
# ==============================================================================
exp_name=${exp_name:-COST2100/out/seed3407/base}
train_path=${train_path:-/storage/hujiacong/zxd/datasets/cost2100/out_train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/cost2100/out_val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/cost2100/out_test.pt}

# ==============================================================================
# 2. 模型结构与数据维度参数
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}
encoder=${encoder:-transnet}
code_adapter=${code_adapter:-false}

# ==============================================================================
# 3. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-4}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-0}
seed=${seed:-3407}

# ==============================================================================
# 4. 运行 Python 脚本
# ==============================================================================
python ./main.py \
  --exp_name "${exp_name}" \
  --train_path "${train_path}" \
  --val_path "${val_path}" \
  --test_path "${test_path}" \
  --epochs "${epochs}" \
  --d_model "${d_model}" \
  --nt "${nt}" \
  --nc "${nc}" \
  --dim_feedforward "${dim_feedforward}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --cr "${cr}" \
  --encoder "${encoder}" \
  --scheduler "${scheduler}" \
  --lr_init "${lr_init}" \
  --weight_decay "${weight_decay}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  $( [ "${code_adapter}" = "true" ] && printf '%s' "--code_adapter" )
