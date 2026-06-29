#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (discover_experiments, ensure_dir, infer_family,
                    infer_scheme, parse_seed_arch, read_json, safe_float)


FLOAT_RE = r"[-+]?(?:\d+\.\d+|\d+)(?:e[-+]?\d+)?"


def parse_metric_pairs(line):
    metrics = {}
    patterns = {
        "train_recon_loss": r"Train\s+Recon loss:\s*(" + FLOAT_RE + ")",
        "val_recon_loss": r"Val\s+Recon loss:\s*(" + FLOAT_RE + ")",
        "code_loss": r"Code loss:\s*(" + FLOAT_RE + ")",
        "fc_loss": r"FC loss:\s*(" + FLOAT_RE + ")",
        "teacher_recon_loss": r"Teacher recon loss:\s*(" + FLOAT_RE + ")",
        "anchor_loss": r"Anchor loss:\s*(" + FLOAT_RE + ")",
        "code_reg_loss": r"Code reg loss:\s*(" + FLOAT_RE + ")",
        "total_loss": r"Total:\s*(" + FLOAT_RE + ")",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            metrics[key] = float(match.group(1))
    for key, value in re.findall(r"adapter/([a-zA-Z0-9_]+)=(" + FLOAT_RE + ")", line):
        metrics[f"adapter_{key}"] = float(value)
    return metrics


def parse_log(log_path):
    text = Path(log_path).read_text(errors="ignore")
    result = {
        "log_lines": text.count("\n") + 1,
        "crashed": "Traceback (most recent call last)" in text,
        "has_runtime_error": "RuntimeError:" in text,
        "loaded_best_before_export": "Loading best checkpoint before codeword export" in text,
        "saved_codewords": "Saved index-aligned codewords" in text,
    }

    best_nmse = None
    best_epoch = None
    for match in re.finditer(
            r"Best NMSE:\s*(" + FLOAT_RE + r")\s*\(epoch=(\d+)\)", text):
        best_nmse = float(match.group(1))
        best_epoch = int(match.group(2))
    result["best_nmse"] = best_nmse
    result["best_epoch"] = best_epoch

    latest_test_nmse = None
    test_epochs = []
    for match in re.finditer(r"=>\s*Test NMSE:\s*(" + FLOAT_RE + ")", text):
        latest_test_nmse = float(match.group(1))
    result["latest_test_nmse"] = latest_test_nmse

    final_test_loss = None
    final_test_nmse = None
    match = re.search(
        r"Final test loss:\s*(" + FLOAT_RE + r").*?test NMSE:\s*(" + FLOAT_RE + ")",
        text,
        flags=re.S,
    )
    if match:
        final_test_loss = float(match.group(1))
        final_test_nmse = float(match.group(2))
    result["final_test_loss"] = final_test_loss
    result["final_test_nmse"] = final_test_nmse
    result["complete"] = final_test_nmse is not None

    first_test = None
    before_train = text.split("Define the training pipeline")[0]
    for match in re.finditer(r"=>\s*Test NMSE:\s*(" + FLOAT_RE + ")", before_train):
        first_test = float(match.group(1))
    result["before_train_test_nmse"] = first_test

    last_train = {}
    last_val = {}
    for line in text.splitlines():
        if "=> Train  Recon loss:" in line:
            last_train = parse_metric_pairs(line)
        elif "=> Val  Recon loss:" in line:
            last_val = parse_metric_pairs(line)
        epoch_match = re.search(r"Epoch:\s*\[(\d+)/(\d+)\]", line)
        if epoch_match:
            result["last_seen_epoch"] = int(epoch_match.group(1))
            result["target_epochs"] = int(epoch_match.group(2))

    for key, value in last_train.items():
        result[f"last_{key}"] = value
    for key, value in last_val.items():
        result[f"last_{key}"] = value

    if "last_seen_epoch" not in result:
        result["last_seen_epoch"] = np.nan
    if "target_epochs" not in result:
        result["target_epochs"] = np.nan
    return result


def build_row(exp_dir, root):
    exp_dir = Path(exp_dir)
    args = read_json(exp_dir / "args.json")
    log_metrics = parse_log(exp_dir / "run.log")
    rel = str(exp_dir.relative_to(root))
    seed_from_dir, enc_from_dir, dec_from_dir = parse_seed_arch(exp_dir.name)

    row = {
        "rel_path": rel,
        "exp_dir": str(exp_dir),
        "scheme": infer_scheme(exp_dir, root),
        "family": infer_family(exp_dir, root),
        "seed": args.get("seed", seed_from_dir),
        "encoder": args.get("encoder", enc_from_dir),
        "decoder": args.get("decoder", dec_from_dir),
        "adapter": args.get("adapter"),
        "epochs": args.get("epochs"),
        "batch_size": args.get("batch_size"),
        "lr_init": args.get("lr_init"),
        "weight_decay": args.get("weight_decay"),
        "pretrained_encoder": args.get("pretrained_encoder"),
        "pretrained_decoder": args.get("pretrained_decoder"),
        "teacher_code": args.get("teacher_code"),
        "lambda_recon": args.get("lambda_recon"),
        "lambda_code": args.get("lambda_code"),
        "lambda_fc": args.get("lambda_fc"),
        "lambda_recT": args.get("lambda_recT"),
        "anchor_target": args.get("anchor_target"),
        "lambda_anchor": args.get("lambda_anchor"),
        "anchor_loss": args.get("anchor_loss"),
        "lambda_code_mean": args.get("lambda_code_mean"),
        "lambda_code_var": args.get("lambda_code_var"),
        "lambda_code_cov": args.get("lambda_code_cov"),
        "lambda_code_l1": args.get("lambda_code_l1"),
        "canonical_head": args.get("canonical_head"),
        "codewords_path": str(exp_dir / "codewords" / "train_code.pt"),
        "codewords_exists": (exp_dir / "codewords" / "train_code.pt").exists(),
        "best_ckpt_exists": (exp_dir / "checkpoints" / "best_nmse.pth").exists(),
    }
    row.update(log_metrics)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="exps/COST2100/in/encoder_canonical")
    parser.add_argument("--out-dir", default="analysis_outputs/encoder_canonical")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = ensure_dir(args.out_dir)
    rows = [build_row(exp_dir, root) for exp_dir in discover_experiments(root)]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No experiments found under {root}")

    numeric_cols = [
        "best_nmse", "final_test_nmse", "latest_test_nmse",
        "before_train_test_nmse", "last_seen_epoch", "target_epochs",
        "last_train_recon_loss", "last_code_loss", "last_fc_loss",
        "last_anchor_loss", "last_code_reg_loss", "last_total_loss",
        "last_adapter_delta_ratio", "last_adapter_gate_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(safe_float)

    df.to_csv(out_dir / "experiment_log_summary.csv", index=False)

    group_cols = ["family", "scheme"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("rel_path", "count"),
            complete=("complete", "sum"),
            codewords=("codewords_exists", "sum"),
            crashed=("crashed", "sum"),
            best_nmse_mean=("best_nmse", "mean"),
            best_nmse_min=("best_nmse", "min"),
            best_nmse_max=("best_nmse", "max"),
            final_nmse_mean=("final_test_nmse", "mean"),
        )
        .reset_index()
        .sort_values(["family", "best_nmse_min"], na_position="last")
    )
    summary.to_csv(out_dir / "scheme_log_summary.csv", index=False)

    print(f"Wrote {len(df)} experiment rows to {out_dir}")
    print(out_dir / "experiment_log_summary.csv")
    print(out_dir / "scheme_log_summary.csv")


if __name__ == "__main__":
    main()

