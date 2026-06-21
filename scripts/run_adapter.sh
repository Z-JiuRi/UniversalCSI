#!/bin/bash

poll_seconds=${poll_seconds:-1800}
gpu=${gpu:-2}
process_pattern=${process_pattern:-"python .*\\./main.py --exp_name [C]OST2100"}

while true; do
  running_count=$(pgrep -fc "${process_pattern}" || true)
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  if [ "${running_count}" -eq 0 ]; then
    echo "[${timestamp}] no COST2100 main.py process found; starting adapter training on GPU ${gpu}"
    encoder=transnet decoder=hybrid gpu=${gpu} encoder_seeds="0 223 314 404 424 424 520 644 796 1014 1024 1115 1234 1337 2026 2048 2718 3407 31415" bash scripts/train_adapter.sh
  else
    echo "[${timestamp}] found ${running_count} COST2100 main.py process(es); waiting for ${poll_seconds} seconds before checking again"
  fi

  sleep "${poll_seconds}"
done
