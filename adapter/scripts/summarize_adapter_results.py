#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def find_test_at_epoch(history, epoch):
    if not isinstance(history, list):
        return {}
    for record in history:
        if record.get("epoch") == epoch:
            return record.get("test", {}) or {}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="adapter/exps/affine_residual_mlp/seed1014/transnet",
        help="experiment root containing adapter runs",
    )
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = load_json(metrics_path)
        if not metrics:
            continue
        best = metrics.get("val_decoder_nmse", {})
        best_metrics = best.get("metrics", {}) or {}
        epoch = best.get("epoch")
        if epoch in (None, 0):
            continue
        history = load_json(metrics_path.with_name("history.json"))
        test_metrics = find_test_at_epoch(history, epoch)
        exp_dir = metrics_path.parent
        rows.append({
            "val_nmse": best_metrics.get("decoder_nmse"),
            "test_nmse": test_metrics.get("decoder_nmse"),
            "val_code_nmse": best_metrics.get("code_nmse"),
            "test_code_nmse": test_metrics.get("code_nmse"),
            "epoch": epoch,
            "path": str(exp_dir),
        })

    rows.sort(key=lambda row: row["val_nmse"] if row["val_nmse"] is not None else float("inf"))
    if not rows:
        print(f"No finished adapter results found under {root}")
        return

    print("rank\tval_nmse\ttest_nmse\tval_code_nmse\ttest_code_nmse\tepoch\tpath")
    for idx, row in enumerate(rows[:args.topk], 1):
        print(
            f"{idx}\t"
            f"{fmt(row['val_nmse'])}\t"
            f"{fmt(row['test_nmse'])}\t"
            f"{fmt(row['val_code_nmse'])}\t"
            f"{fmt(row['test_code_nmse'])}\t"
            f"{row['epoch']}\t"
            f"{row['path']}"
        )


if __name__ == "__main__":
    main()
