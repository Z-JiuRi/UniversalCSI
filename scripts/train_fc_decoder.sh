#!/bin/bash

# batch_size=200 epochs=400 seed=3407 encoder_seed=3407 decoder_seed=42 gpu=0 encoder=transnet decoder=transnet code_loss_only=true bash scripts/train_fc_decoder.sh

# ==============================================================================
# 1. 基础路径与实验名称（环境变量传参，带默认值）
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
# 3. 训练超参数、checkpoint 与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-200}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-0}
seed=${seed:-3407}
encoder_seed=${encoder_seed:-3407}
decoder_seed=${decoder_seed:-42}
pretrained_encoder=${pretrained_encoder:-exps/COST2100/in/seed${encoder_seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth}
pretrained_decoder=${pretrained_decoder:-exps/COST2100/in/seed${decoder_seed}/${encoder}_${decoder}/checkpoints/best_nmse.pth}
teacher_code=${teacher_code-exps/COST2100/in/seed${decoder_seed}/${encoder}_${decoder}/codewords/train_code.pt}
code_loss_lambda=${code_loss_lambda:-}
code_loss_only=${code_loss_only:-false}

if [ -z "${teacher_code}" ]; then
  loss_tag=recon_only
elif [ "${code_loss_only}" = "true" ]; then
  loss_tag=code_only
else
  loss_tag=lambda${code_loss_lambda}
fi
exp_name=${exp_name:-COST2100/in/fc_decoder/seed${seed}/${encoder}${encoder_seed}_${decoder}${decoder_seed}_${loss_tag}}

if [ "${decoder}" != "transnet" ]; then
  echo "train_fc_decoder.sh requires decoder=transnet because only TransNetDecoder has decoder.fc_decoder." >&2
  exit 1
fi

if [ -z "${pretrained_encoder}" ]; then
  echo "pretrained_encoder is required for fc_decoder training." >&2
  exit 1
fi

if [ -z "${pretrained_decoder}" ]; then
  echo "pretrained_decoder is required for fc_decoder training." >&2
  exit 1
fi

if [ ! -f "${pretrained_encoder}" ]; then
  echo "pretrained_encoder checkpoint not found: ${pretrained_encoder}" >&2
  exit 1
fi

if [ ! -f "${pretrained_decoder}" ]; then
  echo "pretrained_decoder checkpoint not found: ${pretrained_decoder}" >&2
  exit 1
fi

extra_args=()
if [ -n "${teacher_code}" ]; then
  if [ ! -f "${teacher_code}" ]; then
    echo "teacher_code not found: ${teacher_code}" >&2
    exit 1
  fi
  extra_args+=(--teacher_code "${teacher_code}")
  if [ -n "${code_loss_lambda}" ]; then
    extra_args+=(--code_loss_lambda "${code_loss_lambda}")
  fi
  if [ "${code_loss_only}" = "true" ]; then
    extra_args+=(--code_loss_only)
  fi
elif [ -n "${code_loss_lambda}" ]; then
  echo "code_loss_lambda=${code_loss_lambda} requires teacher_code." >&2
  exit 1
elif [ "${code_loss_only}" = "true" ]; then
  echo "code_loss_only=true requires teacher_code." >&2
  exit 1
fi

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
  --scheduler "${scheduler}" \
  --lr_init "${lr_init}" \
  --weight_decay "${weight_decay}" \
  --gpu "${gpu}" \
  --seed "${seed}" \
  --train_fc_decoder \
  --pretrained_encoder "${pretrained_encoder}" \
  --pretrained_decoder "${pretrained_decoder}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &

# sleep 5

# tail -f "exps/${exp_name}/run.log"
