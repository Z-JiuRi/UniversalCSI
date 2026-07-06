#!/bin/bash

# 恢复 pause_flow_matching.sh 暂停的 flow_matching 训练进程。
#
# 用法：
#   bash flow_matching/scripts/resume_flow_matching.sh
#
# 默认读取：
#   flow_matching/tmp/paused_flow_matching_pids.txt

set -euo pipefail

pid_file=${pid_file:-flow_matching/tmp/paused_flow_matching_pids.txt}

if [ ! -f "${pid_file}" ]; then
  echo "PID file not found: ${pid_file}" >&2
  exit 1
fi

resumed=0
while IFS= read -r line; do
  [ -n "${line}" ] || continue
  pid="${line%% *}"
  if kill -0 "${pid}" 2>/dev/null; then
    kill -CONT "${pid}"
    echo "[resumed] pid=${pid}"
    resumed=$((resumed + 1))
  else
    echo "[skip] pid=${pid} no longer exists"
  fi
done < "${pid_file}"

echo "Resumed ${resumed} process(es) from ${pid_file}"
