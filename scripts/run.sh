#!/bin/bash

# 定义基础路径和 GPU 列表
BASE_PATH="exps/COST2100/in"
GPU_LIST=(4 5 6 7)
GPU_COUNT=${#GPU_LIST[@]}

# 计数器和用于去重的关联数组
COMMAND_COUNT=0
declare -A USED_SEEDS

echo "正在生成命令..."
echo "--------------------------------------"

# 循环直到成功生成 40 条唯一的命令
while [ $COMMAND_COUNT -lt 40 ]; do
    # 生成一个随机数（这里范围设为 1 到 100000，可根据需要调整）
    RAND_SEED=$((RANDOM % 100000 + 1))

    # 1. 检查当前脚本循环中是否已经生成过这个 seed
    if [[ -n "${USED_SEEDS[$RAND_SEED]}" ]]; then
        continue
    fi

    # 2. 检查目录是否存在，如果存在则跳过
    if [ -d "${BASE_PATH}/seed${RAND_SEED}" ]; then
        continue
    fi

    # 标记该 seed 已被使用
    USED_SEEDS[$RAND_SEED]=1

    # 轮转选择 GPU ID (0%4=4, 1%4=5, 2%4=6, 3%4=7, 4%4=4...)
    GPU_INDEX=$((COMMAND_COUNT % GPU_COUNT))
    GPU_ID=${GPU_LIST[$GPU_INDEX]}

    # 打印生成的命令
    echo "encoder=transnet decoder=hybrid batch_size=256 epochs=400 gpu=${GPU_ID} seed=${RAND_SEED} bash scripts/train.sh"
    echo "sleep 10s"
    encoder=transnet decoder=hybrid batch_size=256 epochs=400 gpu=${GPU_ID} seed=${RAND_SEED} bash scripts/train.sh
    sleep 10

    # 计数器加 1
    COMMAND_COUNT=$((COMMAND_COUNT + 1))
done

echo "--------------------------------------"
echo "成功生成 $COMMAND_COUNT 条命令。"