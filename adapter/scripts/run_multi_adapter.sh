#!/usr/bin/env bash
set -euo pipefail

config_dir="${config_dir:-adapter/configs/exp6/2}"
start_gap_seconds="${start_gap_seconds:-2}"
torch_num_threads="${torch_num_threads:-2}"
python_bin="${python_bin:-python}"

"${python_bin}" adapter/train_multi_adapter.py \
  --config_dir "${config_dir}" \
  --start_gap_seconds "${start_gap_seconds}" \
  --torch_num_threads "${torch_num_threads}" \
  > /dev/null 2>&1 &
