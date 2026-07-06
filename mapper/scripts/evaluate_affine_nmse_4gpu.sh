#!/bin/bash

# 用 0/4/6/7 四张 GPU 分片测试 mapper 下所有 alignaffine 实验保存码字的真实 NMSE。
# 每个分片独立写入 mapper/reports/affine_true_nmse/shards/，最后合并为：
#   mapper/reports/affine_true_nmse/affine_code_nmse.json
#   mapper/reports/affine_true_nmse/affine_code_nmse.csv

set -euo pipefail

gpus=(${gpus:-0 4 6 7})
batch_size=${batch_size:-2048}
workers=${workers:-0}
root=${root:-mapper}
decoder_checkpoint=${decoder_checkpoint:-exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth}
decoder_args_json=${decoder_args_json:-exps/COST2100/in/seed42/transnet_transnet/args.json}
data_path=${data_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
out_dir=${out_dir:-mapper/reports/affine_true_nmse}
shard_dir="${out_dir}/shards"
mkdir -p "${shard_dir}"

pids=()
num_shards=${#gpus[@]}
for shard_id in "${!gpus[@]}"; do
  gpu="${gpus[$shard_id]}"
  output_json="${shard_dir}/affine_code_nmse_shard${shard_id}.json"
  output_csv="${shard_dir}/affine_code_nmse_shard${shard_id}.csv"
  log_path="${shard_dir}/affine_code_nmse_shard${shard_id}.log"
  echo "start shard=${shard_id}/${num_shards} gpu=${gpu}"
  python mapper/evaluate_affine_nmse.py \
    --root "${root}" \
    --decoder_checkpoint "${decoder_checkpoint}" \
    --decoder_args_json "${decoder_args_json}" \
    --data_path "${data_path}" \
    --batch_size "${batch_size}" \
    --workers "${workers}" \
    --gpu "${gpu}" \
    --num_shards "${num_shards}" \
    --shard_id "${shard_id}" \
    --output_json "${output_json}" \
    --output_csv "${output_csv}" \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

python - <<'PY'
import csv
import json
from pathlib import Path

out_dir = Path("mapper/reports/affine_true_nmse")
shard_dir = out_dir / "shards"
rows = []
for path in sorted(shard_dir.glob("affine_code_nmse_shard*.json")):
    rows.extend(json.loads(path.read_text()))
rows.sort(key=lambda row: row.get("relative_path", ""))
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "affine_code_nmse.json").write_text(
    json.dumps(rows, indent=2, ensure_ascii=False),
    encoding="utf-8")
if rows:
    keys = list(rows[0].keys())
    with (out_dir / "affine_code_nmse.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    ok = [row for row in rows if not row.get("error")]
    if ok:
        best = min(ok, key=lambda row: row["nmse_db"])
        print(f"best_nmse={best['nmse_db']:.3f}dB path={best['relative_path']}")
print(f"merged_rows={len(rows)}")
print("saved mapper/reports/affine_true_nmse/affine_code_nmse.json")
print("saved mapper/reports/affine_true_nmse/affine_code_nmse.csv")
PY
