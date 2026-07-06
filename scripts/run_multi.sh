#!/bin/bash
#
# 批量编排多实验训练。
# 直接列出 seed_list / encoder_list / decoder_list / gpu_list，调用 train_multi.sh。
#
# 用法：
#   编辑下方列表，然后：
#     bash scripts/run_multi.sh

set -euo pipefail

seed_list="44,44,44,2025,2025,2025" \
encoder_list="csinet,cnn,clnet,csinet,cnn,clnet" \
decoder_list="transnet,transnet,transnet,transnet,transnet,transnet" \
gpu_list="0,0,0,7,7,7" \
bash scripts/train_multi.sh
