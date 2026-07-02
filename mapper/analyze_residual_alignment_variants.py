#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

TEACHER_CODE = ROOT / "exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt"
SOURCES = {
    "seed2026_transnet": ROOT / "exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt",
    "seed3407_transnet": ROOT / "exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt",
    "seed2026_clnet": ROOT / "exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt",
    "seed2026_crnet": ROOT / "exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt",
    "seed2026_csinet": ROOT / "exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt",
}


def load_code(path):
    x = torch.load(path, map_location="cpu", weights_only=True).float()
    if x.ndim != 2:
        raise ValueError(f"{path} should be 2D, got {tuple(x.shape)}")
    return x


def covariance_eig(x):
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.t().matmul(x) / max(x.size(0) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
    return eigvals


def effective_rank(eigvals):
    return float(eigvals.sum().pow(2) / eigvals.pow(2).sum().clamp_min(1e-12))


def fit_procrustes(source, target):
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    xs = source - source_mean
    yt = target - target_mean
    cross = xs.t().matmul(yt)
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    rotation = u.matmul(vh)
    aligned = xs.matmul(rotation) + target_mean
    return aligned


def fit_affine(source, target, ridge):
    ones = torch.ones(source.size(0), 1, device=source.device, dtype=source.dtype)
    x_aug = torch.cat([source, ones], dim=1)
    xtx = x_aug.t().matmul(x_aug) / source.size(0)
    xty = x_aug.t().matmul(target) / source.size(0)
    eye = torch.eye(xtx.size(0), device=source.device, dtype=source.dtype)
    eye[-1, -1] = 0.0
    weight = torch.linalg.solve(xtx + ridge * eye, xty)
    aligned = x_aug.matmul(weight)
    return aligned


def distribution_fit_stats(values):
    values = values.detach().float().cpu()
    if values.numel() > 1_000_000:
        step = max(1, values.numel() // 1_000_000)
        values = values[::step][:1_000_000]
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(1e-12)
    centered = values - mean
    kurtosis = float(centered.pow(4).mean() / std.pow(4) - 3.0)
    median = values.median()
    laplace_b = (values - median).abs().mean().clamp_min(1e-12)
    normal_nll = float(
        0.5 * math.log(2.0 * math.pi)
        + torch.log(std)
        + centered.pow(2).mean() / (2.0 * std.pow(2))
    )
    laplace_nll = float(
        math.log(2.0)
        + torch.log(laplace_b)
        + (values - median).abs().mean() / laplace_b
    )
    return kurtosis, normal_nll, laplace_nll


def residual_stats(source_name, variant, pred, target):
    residual = target - pred
    per_sample = residual.pow(2).mean(dim=1)
    per_dim_rmse = residual.pow(2).mean(dim=0).sqrt()
    eigvals = covariance_eig(residual)
    cumulative = torch.cumsum(eigvals, dim=0) / eigvals.sum().clamp_min(1e-12)
    flat = residual.reshape(-1)
    kurtosis, normal_nll, laplace_nll = distribution_fit_stats(flat)
    cos = F.cosine_similarity(pred, target, dim=1)
    return {
        "source": source_name,
        "variant": variant,
        "mse": float(per_sample.mean().detach().cpu()),
        "rmse": float(per_sample.mean().sqrt().detach().cpu()),
        "mae": float(residual.abs().mean().detach().cpu()),
        "cos_mean": float(cos.mean().detach().cpu()),
        "sample_mse_p50": float(torch.quantile(per_sample, 0.50).detach().cpu()),
        "sample_mse_p90": float(torch.quantile(per_sample, 0.90).detach().cpu()),
        "sample_mse_p95": float(torch.quantile(per_sample, 0.95).detach().cpu()),
        "sample_mse_p99": float(torch.quantile(per_sample, 0.99).detach().cpu()),
        "dim_rmse_mean": float(per_dim_rmse.mean().detach().cpu()),
        "dim_rmse_p95": float(torch.quantile(per_dim_rmse, 0.95).detach().cpu()),
        "dim_rmse_max": float(per_dim_rmse.max().detach().cpu()),
        "residual_mean_norm": float(residual.mean(dim=0).norm().detach().cpu()),
        "residual_effective_rank": effective_rank(eigvals.detach().cpu()),
        "residual_top10_energy": float(cumulative[9].detach().cpu()),
        "residual_top50_energy": float(cumulative[49].detach().cpu()),
        "residual_top100_energy": float(cumulative[99].detach().cpu()),
        "residual_kurtosis": kurtosis,
        "normal_nll": normal_nll,
        "laplace_nll": laplace_nll,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_source_bars(rows, out_dir):
    sources = sorted({row["source"] for row in rows})
    variants = ["raw", "procrustes", "affine"]
    colors = {
        "raw": "#4C78A8",
        "procrustes": "#F58518",
        "affine": "#E45756",
    }
    by = {(row["source"], row["variant"]): row for row in rows}
    x = np.arange(len(sources))
    width = 0.25

    for key, ylabel, title, filename, logy in [
        ("mse", "residual MSE to teacher", "Residual MSE after alignment", "residual_mse_by_source.png", True),
        ("sample_mse_p95", "sample p95 MSE", "Residual sample tail after alignment", "residual_sample_p95_by_source.png", True),
        ("dim_rmse_max", "max dim RMSE", "Worst residual dimension after alignment", "residual_dim_max_by_source.png", False),
        ("residual_effective_rank", "effective rank", "Residual covariance effective rank", "residual_effective_rank_by_source.png", False),
    ]:
        plt.figure(figsize=(10.5, 4.8))
        for i, variant in enumerate(variants):
            vals = [by[(source, variant)][key] for source in sources]
            plt.bar(x + (i - 1) * width, vals, width=width,
                    label=variant, color=colors[variant])
        if logy:
            plt.yscale("log")
        plt.xticks(x, sources, rotation=25, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=ROOT / "mapper/reports/generative_code_mapping_feasibility/residual_alignment_analysis",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    target = load_code(TEACHER_CODE).to(device)
    rows = []
    for source_name, source_path in SOURCES.items():
        source = load_code(source_path).to(device)
        if source.shape != target.shape:
            raise ValueError(f"{source_name} shape mismatch: {source.shape} vs {target.shape}")

        raw_pred = source
        procrustes_pred = fit_procrustes(source, target)
        affine_pred = fit_affine(source, target, args.ridge)

        rows.append(residual_stats(source_name, "raw", raw_pred, target))
        rows.append(residual_stats(source_name, "procrustes", procrustes_pred, target))
        rows.append(residual_stats(source_name, "affine", affine_pred, target))

    write_csv(out_dir / "residual_alignment_stats.csv", rows)
    plot_source_bars(rows, fig_dir)

    summary = {
        "device": str(device),
        "ridge": args.ridge,
        "teacher_code": str(TEACHER_CODE.relative_to(ROOT)),
        "rows": rows,
        "figures": [
            str(path.relative_to(out_dir))
            for path in sorted(fig_dir.glob("*.png"))
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
