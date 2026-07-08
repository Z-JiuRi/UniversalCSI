#!/bin/bash
# 暂停 0 和 7 号 GPU 上各一半的 main.py 进程（按 PID 排序）。
# 使用 SIGSTOP 暂停，SIGCONT 恢复。
#
# 用法：
#   bash scripts/suspend_half_gpu.sh          # 暂停
#   bash scripts/suspend_half_gpu.sh resume   # 恢复全部

set -euo pipefail

ACTION="${1:-suspend}"  # suspend 或 resume
GPUS=(0 7)

for gpu in "${GPUS[@]}"; do
  # 按 PID 升序排列的进程列表
  pids=$(ps aux | grep "python -u main.py --exp_name COST2100/in" \
    | grep "gpu ${gpu}\b" | grep -v grep \
    | awk '{print $2}' | sort -n)

  if [ -z "$pids" ]; then
    echo "[GPU ${gpu}] No processes found."
    continue
  fi

  pid_arr=($pids)
  total=${#pid_arr[@]}
  mid=$(( (total + 1) / 2 ))  # 上半部分的数量

  if [ "$ACTION" = "suspend" ]; then
    echo "[GPU ${gpu}] Found ${total} processes, suspending last ${mid} (pid ${pid_arr[$mid]}-${pid_arr[$total-1]})"
    for ((i = mid; i < total; i++)); do
      kill -STOP "${pid_arr[$i]}"
      echo "  STOP  pid=${pid_arr[$i]}"
    done
    echo "  => resume with: kill -CONT <pid>"
  elif [ "$ACTION" = "resume" ]; then
    echo "[GPU ${gpu}] Resuming all ${total} processes"
    for pid in "${pid_arr[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill -CONT "$pid"
        echo "  CONT  pid=${pid}"
      else
        echo "  SKIP  pid=${pid} (not running)"
      fi
    done
  fi
done
