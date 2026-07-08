#!/bin/bash
#
# 批量编排多实验训练。
# 直接列出 seed_list / encoder_list / decoder_list / gpu_list，调用 train_multi.sh。
#
# 用法：
#   编辑下方列表，然后：
#     bash scripts/run_multi.sh

set -euo pipefail

num_workers=76

# ==================== 1. 动态随机生成不冲突的 seed ====================
# 从 exist.txt 读取已存在的 seed（跳过注释行）
declare -A used_seeds
while read -r s; do
  [[ -z "$s" || "$s" == \#* ]] && continue
  used_seeds["$s"]=1
done < exist.txt
echo "Loaded ${#used_seeds[@]} used seeds from exist.txt"

seed_arr=()
while [ "${#seed_arr[@]}" -lt $num_workers ]; do
  rand_seed=$(od -An -N4 -tu4 /dev/urandom | awk '{print $1 % 100001}')
  if [ -n "${used_seeds[$rand_seed]:-}" ]; then
    echo "冲突 seed: ${rand_seed}，跳过"
  else
    used_seeds["$rand_seed"]=1
    seed_arr+=("$rand_seed")
  fi
done

# 将 seed 数组转换成逗号分隔的字符串
seed_list=$(IFS=,; echo "${seed_arr[*]}")
echo "成功生成 $num_workers 个可用 seed"

# 将生成的 seed 追加写入 exist.txt（先写时间注释）
{
  printf '# %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  for s in "${seed_arr[@]}"; do printf '%s\n' "$s"; done
} >> exist.txt
echo "Seeds 已追加到 exist.txt"


# ==================== 2. 生成其他对应的参数列表 ====================
gpu_arr=()
for i in {1..38}; do gpu_arr+=("6"); done
for i in {1..38}; do gpu_arr+=("7"); done
# for ((i=1; i<=$num_workers; i++)); do gpu_arr+=("1"); done
gpu_list=$(IFS=,; echo "${gpu_arr[*]}")

enc_arr=()
for ((i=1; i<=$num_workers; i++)); do enc_arr+=("transnet"); done
encoder_list=$(IFS=,; echo "${enc_arr[*]}")

dec_arr=()
for ((i=1; i<=$num_workers; i++)); do dec_arr+=("transnet"); done
decoder_list=$(IFS=,; echo "${dec_arr[*]}")


# ==================== 3. 导出变量并调用训练脚本 ====================
export seed_list encoder_list decoder_list gpu_list

echo $seed_list
echo $encoder_list
echo $decoder_list
echo $gpu_list

bash scripts/train_multi.sh