#!/bin/bash

# 运行命令 (一行复制即可):
#   adapter=mlp gpu=2 seed=3407 bash scripts/train_adapter.sh
#   adapter=transformer adapter_hidden_dim=256 gpu=2 seed=3407 bash scripts/train_adapter.sh

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

# ==============================================================================
# 3. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-2}
seed=${seed:-42}

adapter=${adapter:-mlp}
adapter_hidden_dim=${adapter_hidden_dim:-}
pretrained_encoder=${pretrained_encoder:-exps/COST2100/in/seed${seed}/transnet_hybrid/checkpoints/best_nmse.pth}
pretrained_decoder=${pretrained_decoder:-exps/COST2100/in/seed42/transnet_hybrid/checkpoints/best_nmse.pth}

exp_name=${exp_name:-COST2100/in/adapter/${adapter}/seed${seed}}

extra_args=()
add_arg() { local flag=$1 val=$2; [ -n "$val" ] && extra_args+=("$flag" "$val"); }

add_arg --adapter             "$adapter"
add_arg --adapter_hidden_dim  "$adapter_hidden_dim"
add_arg --pretrained_encoder  "$pretrained_encoder"
add_arg --pretrained_decoder  "$pretrained_decoder"


# ==============================================================================
# 5. 运行 Python 脚本
# ==============================================================================
python -u main.py \
  --exp_name ${exp_name} \
  --train_path ${train_path} \
  --val_path ${val_path} \
  --test_path ${test_path} \
  --epochs ${epochs} \
  --d_model ${d_model} \
  --nt ${nt} \
  --nc ${nc} \
  --dim_feedforward ${dim_feedforward} \
  --hidden ${hidden} \
  --num_blocks ${num_blocks} \
  --batch_size ${batch_size} \
  --workers ${workers} \
  --cr ${cr} \
  --encoder ${encoder} \
  --decoder ${decoder} \
  --scheduler ${scheduler} \
  --lr_init ${lr_init} \
  --weight_decay ${weight_decay} \
  --gpu ${gpu} \
  --seed ${seed} \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &
