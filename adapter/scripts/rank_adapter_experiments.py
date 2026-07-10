#!/usr/bin/env python
import argparse
import json
import math
import subprocess
import time
from pathlib import Path


PROCESS_PATTERN = r"adapter/(train_multi_adapter|train_adapter.py)"


def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def bool_int(value):
    return int(bool(value))


def mapper_trainable_params(args):
    channel = int(args.get("channel", 2))
    nt = int(args.get("nt", 32))
    nc = int(args.get("nc", 32))
    cr = int(args.get("cr", 4))
    dim = channel * nt * nc // cr
    hidden_dim = int(args.get("hidden_dim", 1024))
    lowrank_rank = int(args.get("lowrank_rank", 64))
    num_blocks = int(args.get("num_blocks", 4))
    mapper_type = args.get("mapper_type", "affine_residual_mlp")
    train_affine = bool(args.get("train_affine", False))
    learnable_gate = bool(args.get("learnable_residual_gate", False))
    total = 0

    if train_affine:
        total += dim * dim + dim

    if mapper_type == "affine_residual_mlp":
        per_block = dim * hidden_dim + hidden_dim
        per_block += hidden_dim * dim + dim
        if not bool(args.get("no_block_norm", False)):
            per_block += 2 * dim
        if learnable_gate:
            per_block += dim
        total += num_blocks * per_block
        if bool(args.get("use_final_norm", False)):
            total += 2 * dim
    elif mapper_type == "affine_lowrank_residual":
        per_block = dim * lowrank_rank + lowrank_rank
        per_block += lowrank_rank * dim + dim
        if not bool(args.get("no_block_norm", False)):
            per_block += 2 * dim
        if learnable_gate:
            per_block += dim
        total += num_blocks * per_block
        if bool(args.get("use_final_norm", False)):
            total += 2 * dim
    elif mapper_type == "affine_linear":
        total += num_blocks * (dim * dim + dim)
    elif mapper_type == "direct_mlp":
        in_dim = dim
        for _ in range(num_blocks):
            total += in_dim * hidden_dim + hidden_dim
            in_dim = hidden_dim
        total += in_dim * dim + dim
    return total


def find_test_at_epoch(history, epoch):
    if not isinstance(history, list):
        return {}
    for record in history:
        if record.get("epoch") == epoch:
            return record.get("test", {}) or {}
    return {}


def process_running():
    result = subprocess.run(
        ["pgrep", "-af", PROCESS_PATTERN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def collect_rows(root):
    rows = []
    incomplete = []
    for args_path in sorted(Path(root).rglob("args.json")):
        exp_dir = args_path.parent
        args = load_json(args_path) or {}
        metrics = load_json(exp_dir / "metrics.json")
        history = load_json(exp_dir / "history.json")
        trainable_params = mapper_trainable_params(args)
        row = {
            "path": str(exp_dir),
            "name": exp_dir.name,
            "mapper_type": args.get("mapper_type", "affine_residual_mlp"),
            "hidden_dim": args.get("hidden_dim"),
            "lowrank_rank": args.get("lowrank_rank"),
            "params_m": trainable_params / 1e6,
            "lambda_code": args.get("lambda_code"),
            "lambda_recon": args.get("lambda_recon"),
            "code_loss_type": args.get("code_loss_type", "mse"),
            "residual_scale": args.get("residual_scale"),
            "learnable_residual_gate": bool(args.get("learnable_residual_gate", False)),
            "gate_max": args.get("gate_max"),
            "block_norm": not bool(args.get("no_block_norm", False)),
            "align_ridge": args.get("align_ridge"),
            "train_affine": bool(args.get("train_affine", False)),
            "epochs": args.get("epochs"),
            "best_epoch": None,
            "val_nmse": None,
            "test_nmse": None,
            "val_code_nmse": None,
            "test_code_nmse": None,
            "priority_score": None,
        }
        if not metrics:
            incomplete.append(row)
            continue
        best = metrics.get("val_decoder_nmse", {})
        best_metrics = best.get("metrics", {}) or {}
        best_epoch = best.get("epoch")
        test_metrics = find_test_at_epoch(history, best_epoch)
        row.update({
            "best_epoch": best_epoch,
            "val_nmse": best_metrics.get("decoder_nmse"),
            "test_nmse": test_metrics.get("decoder_nmse"),
            "val_code_nmse": best_metrics.get("code_nmse"),
            "test_code_nmse": test_metrics.get("code_nmse"),
        })
        if row["val_nmse"] is None:
            incomplete.append(row)
            continue
        # Lower is better. Parameter penalty is deliberately mild: 0.02 dB per M trainable params.
        row["priority_score"] = row["val_nmse"] + 0.02 * row["params_m"]
        rows.append(row)
    rows.sort(key=lambda row: (
        row["priority_score"],
        row["val_nmse"],
        row["params_m"],
        row["test_nmse"] if row["test_nmse"] is not None else math.inf,
    ))
    return rows, incomplete


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(rows, incomplete, output_path, topk):
    lines = []
    lines.append("# Adapter Experiment Priority Ranking")
    lines.append("")
    lines.append("排序规则：优先按验证集 decoder NMSE，加入轻量参数惩罚 `0.02 dB / M trainable params`。分数越低越优先。")
    lines.append("")
    lines.append("| rank | score | val_nmse | test_nmse | params_M | mapper | hidden/rank | loss | recon | ridge | gate | block_norm | train_affine | rs | best_epoch | name |")
    lines.append("|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for idx, row in enumerate(rows[:topk], 1):
        size = row["lowrank_rank"] if row["mapper_type"] == "affine_lowrank_residual" else row["hidden_dim"]
        lines.append(
            f"| {idx} | {fmt(row['priority_score'])} | {fmt(row['val_nmse'])} | "
            f"{fmt(row['test_nmse'])} | {fmt(row['params_m'])} | "
            f"`{row['mapper_type']}` | {fmt(size, 0)} | `{row['code_loss_type']}` | "
            f"{fmt(row['lambda_recon'])} | {fmt(row['align_ridge'])} | "
            f"{fmt(row['learnable_residual_gate'])} | {fmt(row['block_norm'])} | "
            f"{fmt(row['train_affine'])} | {fmt(row['residual_scale'])} | "
            f"{fmt(row['best_epoch'], 0)} | `{row['name']}` |"
        )
    if incomplete:
        lines.append("")
        lines.append("## Incomplete")
        for row in incomplete:
            lines.append(f"- `{row['path']}`")
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="adapter/exps",
        help="experiment root",
    )
    parser.add_argument("--output", default="adapter/exps/adapter_priority_report.md")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--interval_seconds", type=int, default=600)
    args = parser.parse_args()

    if args.wait:
        while process_running():
            print(f"adapter training is still running; sleep {args.interval_seconds}s", flush=True)
            time.sleep(args.interval_seconds)

    rows, incomplete = collect_rows(args.root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(rows, incomplete, output_path, args.topk)
    print(f"saved report: {output_path}")
    print("rank\tscore\tval_nmse\ttest_nmse\tparams_M\tmapper\thidden_or_rank\tloss\trecon\tridge\tgate\tblock_norm\ttrain_affine\trs\tbest_epoch\tname")
    for idx, row in enumerate(rows[:args.topk], 1):
        size = row["lowrank_rank"] if row["mapper_type"] == "affine_lowrank_residual" else row["hidden_dim"]
        print(
            f"{idx}\t{fmt(row['priority_score'])}\t{fmt(row['val_nmse'])}\t"
            f"{fmt(row['test_nmse'])}\t{fmt(row['params_m'])}\t"
            f"{row['mapper_type']}\t{fmt(size, 0)}\t{row['code_loss_type']}\t"
            f"{fmt(row['lambda_recon'])}\t{fmt(row['align_ridge'])}\t"
            f"{fmt(row['learnable_residual_gate'])}\t{fmt(row['block_norm'])}\t"
            f"{fmt(row['train_affine'])}\t{fmt(row['residual_scale'])}\t"
            f"{fmt(row['best_epoch'], 0)}\t{row['name']}"
        )
    if incomplete:
        print(f"incomplete: {len(incomplete)}")


if __name__ == "__main__":
    main()
