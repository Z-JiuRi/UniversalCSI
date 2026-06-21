#!/bin/bash

# encoder=transnet decoder=hybrid gpu=2 encoder_seeds="1 2 3 42 3407" bash scripts/train_adapter.sh

# ==============================================================================
# 1. 基础路径与实验名称（环境变量传参，带默认值）
# ==============================================================================
train_path=${train_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/cost2100/in_val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/cost2100/in_test.pt}

# ==============================================================================
# 2. 模型结构、adapter 与数据维度参数
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
hidden=${hidden:-16}
num_blocks=${num_blocks:-2}
cr=${cr:-4}
encoder=${encoder:-transnet}
decoder=${decoder:-hybrid}
encoder_seeds=${encoder_seeds:-"1 2 3 42 3407"}
decoder_seed=${decoder_seed:-42}
pretrained_decoder=${pretrained_decoder:-exps/COST2100/in/seed42/transnet_hybrid/checkpoints/best_nmse.pth}

# ==============================================================================
# 3. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-2}
seed=${seed:-42}
exp_name=${exp_name:-COST2100/in/adapter/shared_adapter/decoder_seed${decoder_seed}/${encoder}_${decoder}}

# ==============================================================================
# 4. 运行 Python 脚本
# ==============================================================================
/home/hujiacong/zxd/.envs/miniconda3/envs/torch/bin/python ./main.py \
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
  --encoder_seeds ${encoder_seeds} \
  --decoder_seed "${decoder_seed}" \
  --pretrained_decoder "${pretrained_decoder}" \
  --scheduler "${scheduler}" \
  --lr_init "${lr_init}" \
  --weight_decay "${weight_decay}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  > /dev/null 2>&1 &

