#!/bin/bash

# 暂停所有正在运行的 flow_matching 训练进程，并保存 PID 列表。
#
# 用法：
#   bash flow_matching/scripts/pause_flow_matching.sh
#
# 默认保存：
#   flow_matching/tmp/paused_flow_matching_pids.txt
#
# 恢复：
#   bash flow_matching/scripts/resume_flow_matching.sh

set -euo pipefail

pid_file=${pid_file:-flow_matching/tmp/paused_flow_matching_pids.txt}
pattern=${pattern:-"python -u flow_matching/train_flow_matching.py"}

mkdir -p "$(dirname "${pid_file}")"
: > "${pid_file}"

mapfile -t lines < <(pgrep -af "${pattern}" || true)

if [ "${#lines[@]}" -eq 0 ]; then
  echo "No matching process found: ${pattern}"
  exit 0
fi

for line in "${lines[@]}"; do
  pid="${line%% *}"
  cmd="${line#* }"
  if [ "${pid}" = "$$" ]; then
    continue
  fi
  echo "${pid} ${cmd}" >> "${pid_file}"
  kill -STOP "${pid}"
  echo "[paused] pid=${pid}"
done

echo "Saved paused PIDs to ${pid_file}"
