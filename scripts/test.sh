#!/bin/bash

#    Usage:
#    seed=42 gpu=0 encoder=transnet decoder=transnet bash scripts/test.sh

train_path=${train_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/cost2100/in_val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/cost2100/in_test.pt}

d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
hidden=${hidden:-16}
num_blocks=${num_blocks:-2}
cr=${cr:-4}
encoder=${encoder:-transnet}
decoder=${decoder:-transnet}
code_adapter=${code_adapter:-false}

batch_size=${batch_size:-32}
workers=${workers:-4}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-0}
seed=${seed:-42}

pretrained=${pretrained:-exps/COST2100/in/seed${seed}/${encoder}_${decoder}/base/checkpoints/best_nmse.pth}
exp_name=${exp_name:-COST2100/in/test/seed${seed}/${encoder}_${decoder}}

python ./main.py \
  --exp_name "${exp_name}" \
  --train_path "${train_path}" \
  --val_path "${val_path}" \
  --test_path "${test_path}" \
  --epochs 1 \
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
  --pretrained "${pretrained}" \
  --evaluate \
  > /dev/null 2>&1 &

# sleep 5

# tail -f ${exp_name}/run.log