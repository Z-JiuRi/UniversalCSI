#!/bin/bash

# 定义列表
encoders=('csinet' 'cnn' 'cbam_cnn' 'crnet' 'clnet' 'transnet'
          'resnet' 'dscnn' 'convnext' 'mlp_mixer' 'attention_cnn'
          'swin' 'mlp_ae' 'sparse_resnet')
decoders=('transnet' 'hybrid')

gpu_count=6
index=0

# 嵌套循环
for encoder in "${encoders[@]}"; do
    for decoder in "${decoders[@]}"; do
        # 计算 GPU 编号（循环 0~5）
        gpu=$((index % gpu_count))
        
        # 日志文件名
        logfile="in_${encoder}_${decoder}_seed42.log"
        
        echo "Launching: encoder=${encoder}, decoder=${decoder}, gpu=${gpu}, log=${logfile}"
        
        # 启动训练任务（后台运行）
        exp_name="seed42/WAIRD/${encoder}_${decoder}/base" \
        train_path="/storage/hujiacong/zxd/datasets/WAIRD/data/base/train.pt" \
        val_path="/storage/hujiacong/zxd/datasets/WAIRD/data/base/val.pt" \
        test_path="/storage/hujiacong/zxd/datasets/WAIRD/data/base/test.pt" \
        encoder="$encoder" \
        decoder="$decoder" \
        code_adapter=false \
        nt=64 \
        nc=64 \
        batch_size=200 \
        epochs=400 \
        lr_init=2e-4 \
        gpu=$gpu \
        seed=42 \
        ./scripts/train.sh > "$logfile" 2>&1 &
        
        # 计数器递增
        index=$((index + 1))

        sleep 5

    done
done

echo "All jobs launched. Monitor with 'nvidia-smi' or 'ps'."