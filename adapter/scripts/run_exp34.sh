#!/usr/bin/env bash
# Run the all-split affine oracle diagnostic. Result is not a valid held-out test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python adapter/configs/exp34_oracle_affine/generate_exp34_configs.py
python adapter/train_multi_adapter.py \
  --config_dir adapter/configs/exp34_oracle_affine/branches \
  --torch_num_threads "${torch_num_threads:-8}"
