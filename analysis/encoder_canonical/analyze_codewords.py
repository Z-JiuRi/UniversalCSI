#!/usr/bin/env python
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ensure_dir, parse_seed_arch


def resolve_device(device_arg, gpu):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu}")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def load_codes(path, sample_size=None, device=torch.device("cpu")):
    codes = torch.load(path, weights_only=True, map_location="cpu").float()
    if codes.ndim != 2:
        raise ValueError(f"{path} should be 2D, got {tuple(codes.shape)}")
    if sample_size and codes.size(0) > sample_size:
        codes = codes[:sample_size].contiguous()
    return codes.to(device, non_blocking=True)


def effective_rank(cov):
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    return (eig.sum().pow(2) / (eig.pow(2).sum() + 1e-12)).item()


def code_stats(codes):
    centered = codes - codes.mean(dim=0, keepdim=True)
    denom = max(codes.size(0) - 1, 1)
    cov = centered.T @ centered / denom
    diag = torch.diag(cov)
    offdiag = cov - torch.diag(diag)
    offdiag_ratio = (
        offdiag.pow(2).sum().sqrt() / (diag.pow(2).sum().sqrt() + 1e-12)
    ).item()
    dim_var = codes.var(dim=0, unbiased=False)
    return {
        "num_samples": int(codes.size(0)),
        "code_dim": int(codes.size(1)),
        "global_mean": codes.mean().item(),
        "global_std": codes.std(unbiased=False).item(),
        "global_rms": codes.pow(2).mean().sqrt().item(),
        "sample_norm_mean": codes.norm(dim=1).mean().item(),
        "sample_norm_std": codes.norm(dim=1).std(unbiased=False).item(),
        "dim_mean_abs": codes.mean(dim=0).abs().mean().item(),
        "dim_var_mean": dim_var.mean().item(),
        "dim_var_std": dim_var.std(unbiased=False).item(),
        "dim_var_cv": (dim_var.std(unbiased=False) / (dim_var.mean() + 1e-12)).item(),
        "cov_offdiag_ratio": offdiag_ratio,
        "effective_rank": effective_rank(cov),
    }


def normalized(codes):
    return torch.nn.functional.normalize(codes, dim=1)


def compare_codes(a, b):
    n = min(a.size(0), b.size(0))
    a = a[:n]
    b = b[:n]
    an = normalized(a)
    bn = normalized(b)
    cos = (an * bn).sum(dim=1)
    return {
        "num_samples": int(n),
        "cos_mean": cos.mean().item(),
        "cos_std": cos.std(unbiased=False).item(),
        "cos_min": cos.min().item(),
        "cos_p05": torch.quantile(cos, 0.05).item(),
        "cos_p50": torch.quantile(cos, 0.50).item(),
        "cos_p95": torch.quantile(cos, 0.95).item(),
        "mse": (a - b).pow(2).mean().item(),
        "l2_mean": (a - b).norm(dim=1).mean().item(),
    }


def load_log_summary(path):
    df = pd.read_csv(path)
    df = df[df["codewords_exists"] == True].copy()  # noqa: E712
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-summary",
                        default="analysis_outputs/encoder_canonical/experiment_log_summary.csv")
    parser.add_argument("--out-dir", default="analysis_outputs/encoder_canonical")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--max-pair-per-family", type=int, default=2000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for codeword covariance/pairwise tensor ops.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index used when --device is auto/cuda.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    device = resolve_device(args.device, args.gpu)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device for codeword analysis: {device}")
    df = load_log_summary(args.log_summary)
    if df.empty:
        raise SystemExit("No codeword files listed in log summary")

    cache = {}
    stats_rows = []
    for _, row in df.iterrows():
        path = Path(row["codewords_path"])
        if not path.exists():
            continue
        try:
            codes = load_codes(path, args.sample_size, device)
            stats = code_stats(codes)
        except Exception as exc:
            stats_rows.append({
                "rel_path": row["rel_path"],
                "load_error": str(exc),
            })
            continue
        cache[row["rel_path"]] = codes
        stats.update({
            "rel_path": row["rel_path"],
            "family": row["family"],
            "scheme": row["scheme"],
            "seed": row.get("seed"),
            "encoder": row.get("encoder"),
            "decoder": row.get("decoder"),
            "adapter": row.get("adapter"),
            "best_nmse": row.get("best_nmse"),
            "final_test_nmse": row.get("final_test_nmse"),
            "complete": row.get("complete"),
        })
        stats_rows.append(stats)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "codeword_stats.csv", index=False)

    pair_rows = []
    by_family = defaultdict(list)
    for _, row in df.iterrows():
        if row["rel_path"] in cache:
            by_family[(row["family"], row["scheme"])].append(row)

    for (family, scheme), rows in by_family.items():
        rows = list(rows)
        pair_count = 0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if pair_count >= args.max_pair_per_family:
                    break
                a = rows[i]
                b = rows[j]
                ca = cache.get(a["rel_path"])
                cb = cache.get(b["rel_path"])
                if ca is None or cb is None:
                    continue
                cmp_stats = compare_codes(ca, cb)
                cmp_stats.update({
                    "family": family,
                    "scheme": scheme,
                    "rel_a": a["rel_path"],
                    "rel_b": b["rel_path"],
                    "seed_a": a.get("seed"),
                    "seed_b": b.get("seed"),
                    "encoder_a": a.get("encoder"),
                    "encoder_b": b.get("encoder"),
                    "decoder_a": a.get("decoder"),
                    "decoder_b": b.get("decoder"),
                    "adapter_a": a.get("adapter"),
                    "adapter_b": b.get("adapter"),
                })
                pair_rows.append(cmp_stats)
                pair_count += 1
            if pair_count >= args.max_pair_per_family:
                break

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "codeword_pairwise.csv", index=False)

    print(f"Wrote codeword stats: {out_dir / 'codeword_stats.csv'}")
    print(f"Wrote pairwise stats: {out_dir / 'codeword_pairwise.csv'}")
    print(f"Loaded {len(cache)} codeword tensors with sample_size={args.sample_size}")


if __name__ == "__main__":
    main()
