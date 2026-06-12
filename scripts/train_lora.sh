#!/bin/bash

# lora_component=token_projection seed=2026 gpu=0 nt=32 nc=32 encoder=transnet decoder=hybrid batch_size=200 epochs=400 lora_rank=64 lora_alpha=128 pretrained_encoder=exps/COST2100/in/seed42/transnet_hybrid/base/checkpoints/best_nmse.pth pretrained_decoder=exps/COST2100/in/seed2026/transnet_hybrid/checkpoints/best_nmse.pth bash scripts/train_lora.sh

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
decoder=${decoder:-hybrid}
code_adapter=${code_adapter:-false}
lora_component=${lora_component:-token_projection}
lora_rank=${lora_rank:-8}
lora_alpha=${lora_alpha:-16}

# ==============================================================================
# 3. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-1024}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-0}
seed=${seed:-3407}
exp_name=${exp_name:-COST2100/in/lora/seed${seed}/${encoder}_${decoder}_${lora_component}_rank${lora_rank}_alpha${lora_alpha}}
pretrained=${pretrained:-}
pretrained_encoder=${pretrained_encoder:-}
pretrained_decoder=${pretrained_decoder:-}

if [ "${decoder}" != "hybrid" ]; then
  echo "train_lora.sh currently supports decoder=hybrid only because --lora_component token_projection is only implemented for HybridDecoder." >&2
  exit 1
fi

if [ "${lora_component}" != "token_projection" ]; then
  echo "Unsupported lora_component=${lora_component}; currently only token_projection is supported." >&2
  exit 1
fi

extra_args=()
if [ -n "${pretrained_encoder}" ] || [ -n "${pretrained_decoder}" ]; then
  if [ -n "${pretrained_encoder}" ] && [ ! -f "${pretrained_encoder}" ]; then
    echo "pretrained_encoder checkpoint not found: ${pretrained_encoder}" >&2
    exit 1
  fi
  if [ -n "${pretrained_decoder}" ] && [ ! -f "${pretrained_decoder}" ]; then
    echo "pretrained_decoder checkpoint not found: ${pretrained_decoder}" >&2
    exit 1
  fi
  if [ -n "${pretrained_encoder}" ]; then
    extra_args+=(--pretrained_encoder "${pretrained_encoder}")
  fi
  if [ -n "${pretrained_decoder}" ]; then
    extra_args+=(--pretrained_decoder "${pretrained_decoder}")
  fi
else
  if [ ! -f "${pretrained}" ]; then
    echo "pretrained checkpoint not found: ${pretrained}" >&2
    exit 1
  fi
  extra_args+=(--pretrained "${pretrained}")
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
  --lora_component "${lora_component}" \
  --lora_rank "${lora_rank}" \
  --lora_alpha "${lora_alpha}" \
  "${extra_args[@]}" \
  > /dev/null 2>&1 &

# sleep 5

# tail -f ${exp_name}/run.log
