#!/bin/bash

# 运行命令:
#   adapter=mlp gpu=6 seed=3407 lambda_recon=0.0 lambda_code=1.0 bash scripts/train_adapter.sh
#   adapter=transformer adapter_hidden_dim=256 gpu=2 seed=3407 bash scripts/train_adapter.sh
#   adapter=lowrank_affine adapter_rank=32 gpu=2 seed=3407 bash scripts/train_adapter.sh
#   adapter=gated_lowrank_affine_mlp adapter_rank=32 adapter_gate_init=0.1 gpu=2 seed=3407 bash scripts/train_adapter.sh
#   adapter=gated_lowrank_affine_linear adapter_rank=32 adapter_gate_init=0.1 gpu=2 seed=3407 bash scripts/train_adapter.sh


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
gpu=${gpu:-2}
seed=${seed:-42}

adapter=${adapter:-mlp}
adapter_hidden_dim=${adapter_hidden_dim:-2048}
adapter_rank=${adapter_rank:-32}
adapter_gate_init=${adapter_gate_init:-0.1}
pretrained_encoder=${pretrained_encoder:-exps/COST2100/in/seed${seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth}
pretrained_decoder=${pretrained_decoder:-exps/COST2100/in/seed42/${encoder}_${decoder}/checkpoints/best_nmse.pth}

teacher_code=${teacher_code:-exps/COST2100/in/seed42/${encoder}_${decoder}/codewords/train_code.pt}
lambda_recon=${lambda_recon:-1.0}
lambda_code=${lambda_code:-0.0}
lambda_fc=${lambda_fc:-0.0}
lambda_recT=${lambda_recT:-0.0}

exp_name=${exp_name:-COST2100/in/adapter/${adapter}/seed${seed}_recon${lambda_recon}_code${lambda_code}_fc${lambda_fc}_recT${lambda_recT}_lr${lr_init}}

if [ ! -f "${teacher_code}" ]; then
  echo "Missing teacher_code: ${teacher_code}" >&2
  exit 1
fi

extra_args=()
add_arg() { local flag=$1 val=$2; [ -n "$val" ] && extra_args+=("$flag" "$val"); }

add_arg --adapter             "$adapter"
add_arg --adapter_hidden_dim  "$adapter_hidden_dim"
add_arg --adapter_rank        "$adapter_rank"
add_arg --adapter_gate_init   "$adapter_gate_init"
add_arg --pretrained_encoder  "$pretrained_encoder"
add_arg --pretrained_decoder  "$pretrained_decoder"
add_arg --lambda_recon        "$lambda_recon"
add_arg --lambda_code         "$lambda_code"
add_arg --lambda_fc           "$lambda_fc"
add_arg --lambda_recT         "$lambda_recT"
add_arg --teacher_code        "$teacher_code"


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
