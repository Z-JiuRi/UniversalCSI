#!/usr/bin/env bash
set -euo pipefail

start_gap_seconds="${start_gap_seconds:-2}"
torch_num_threads="${torch_num_threads:-2}"

config_dir=adapter/configs/exp7/0
python adapter/train_multi_adapter.py \
  --config_dir "${config_dir}" \
  --start_gap_seconds "${start_gap_seconds}" \
  --torch_num_threads "${torch_num_threads}" \
  > /dev/null 2>&1 &
echo "Started training for config_dir: ${config_dir}"
sleep 10

config_dir=adapter/configs/exp7/1
python adapter/train_multi_adapter.py \
  --config_dir "${config_dir}" \
  --start_gap_seconds "${start_gap_seconds}" \
  --torch_num_threads "${torch_num_threads}" \
  > /dev/null 2>&1 &
echo "Started training for config_dir: ${config_dir}"
sleep 10

config_dir=adapter/configs/exp7/4
python adapter/train_multi_adapter.py \
  --config_dir "${config_dir}" \
  --start_gap_seconds "${start_gap_seconds}" \
  --torch_num_threads "${torch_num_threads}" \
  > /dev/null 2>&1 &
echo "Started training for config_dir: ${config_dir}"
sleep 10

config_dir=adapter/configs/exp7/6
python adapter/train_multi_adapter.py \
  --config_dir "${config_dir}" \
  --start_gap_seconds "${start_gap_seconds}" \
  --torch_num_threads "${torch_num_threads}" \
  > /dev/null 2>&1 &
echo "Started training for config_dir: ${config_dir}"
sleep 10

config_dir=adapter/configs/exp7/7
python adapter/train_multi_adapter.py \
  --config_dir "${config_dir}" \
  --start_gap_seconds "${start_gap_seconds}" \
  --torch_num_threads "${torch_num_threads}" \
  > /dev/null 2>&1 &
echo "Started training for config_dir: ${config_dir}"
