#!/usr/bin/env bash
set -euo pipefail

interval_seconds="${interval_seconds:-600}"
run_script="${run_script:-adapter/scripts/run_adapter.sh}"
process_pattern="${process_pattern:-[p]ython.*-u[[:space:]]+adapter/train_adapter.py}"

while true; do
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  if pgrep -af "${process_pattern}" >/dev/null; then
    echo "[${timestamp}] train_adapter.py is still running. Check again in ${interval_seconds}s."
    sleep "${interval_seconds}"
    continue
  fi

  echo "[${timestamp}] no train_adapter.py process found. Launching ${run_script}."
  bash "${run_script}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launched once. Exiting."
  exit 0
done
