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
import matplotlib.font_manager as fm


ROOT = Path(__file__).resolve().parents[1]
SONGTI_PATH = Path("/home/hujiacong/zxd/.envs/SongTi.ttf")

DEFAULT_CSI = "/storage/hujiacong/zxd/datasets/cost2100/in_train.pt"
TEACHER_CODE = "exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt"
SOURCE_CODES = {
    "seed2026_transnet": "exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt",
    "seed3407_transnet": "exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt",
    "seed2026_clnet": "exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt",
    "seed2026_crnet": "exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt",
    "seed2026_csinet": "exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt",
}
MAPPED_CODES = {
    "old_hybrid_seed2026_transnet": "mapper/exps/hybrid/seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400/mapped_code.pt",
    "smooth_tail_white_seed2026_transnet": "mapper/exps/hybrid/smooth_tail_white/seed2026_transnet_transnet_to_seed42_transnet_lr5e-4_ep400/codewords/mapped_code.pt",
    "smoothl1_seed2026_clnet": "mapper/exps/hybrid/smoothl1_0.5/seed2026_clnet_transnet_to_seed42_transnet_lr5e-4_ep400/codewords/mapped_code.pt",
}


def setup_fonts():
    if SONGTI_PATH.exists():
        fm.fontManager.addfont(str(SONGTI_PATH))
        font = fm.FontProperties(fname=str(SONGTI_PATH))
        plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def rel(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_tensor(path):
    return torch.load(rel(path), map_location="cpu", weights_only=True).float()


def quantiles(values, qs=(0.01, 0.05, 0.5, 0.9, 0.95, 0.99)):
    t = values.detach().float().cpu()
    return {f"p{int(q * 100):02d}": float(torch.quantile(t, q)) for q in qs}


def pearson(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(a.dot(b) / denom)


def effective_rank(eigvals):
    eigvals = eigvals.clamp_min(0)
    return float(eigvals.sum().pow(2) / eigvals.pow(2).sum().clamp_min(1e-12))


def covariance_eig(x, device):
    x = x.to(device)
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.t().matmul(x) / max(x.size(0) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return eigvals[order].clamp_min(0), eigvecs[:, order]


def csi_analysis(csi, device, out_dir, pca_samples, kmeans_k):
    fig_dir = out_dir / "figures"
    n = csi.size(0)
    flat = csi.reshape(n, -1)
    channel_stats = []
    for ch in range(csi.size(1)):
        x = csi[:, ch]
        channel_stats.append({
            "channel": ch,
            "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)),
            "rms": float(x.pow(2).mean().sqrt()),
            "abs_mean": float(x.abs().mean()),
        })

    power = flat.pow(2).sum(dim=1)
    power_stats = {
        "n": n,
        "shape": list(csi.shape),
        "global_mean": float(csi.mean()),
        "global_std": float(csi.std(unbiased=False)),
        "global_rms": float(csi.pow(2).mean().sqrt()),
        "sample_power_mean": float(power.mean()),
        "sample_power_std": float(power.std(unbiased=False)),
        "sample_power_min": float(power.min()),
        "sample_power_max": float(power.max()),
        **{f"sample_power_{k}": v for k, v in quantiles(power).items()},
    }

    energy = csi.pow(2).sum(dim=1).reshape(n, -1)
    sorted_energy, _ = torch.sort(energy, dim=1, descending=True)
    total_energy = sorted_energy.sum(dim=1).clamp_min(1e-12)
    bins = energy.size(1)
    top_counts = {
        "top1pct": max(1, math.ceil(bins * 0.01)),
        "top5pct": max(1, math.ceil(bins * 0.05)),
        "top10pct": max(1, math.ceil(bins * 0.10)),
    }
    sparsity = {}
    for name, count in top_counts.items():
        frac = sorted_energy[:, :count].sum(dim=1) / total_energy
        sparsity[name + "_energy_frac_mean"] = float(frac.mean())
        sparsity[name + "_energy_frac_p50"] = float(torch.quantile(frac, 0.50))
        sparsity[name + "_energy_frac_p95"] = float(torch.quantile(frac, 0.95))
    cum_energy = torch.cumsum(sorted_energy, dim=1) / total_energy[:, None]
    needed_bins = {}
    for threshold in [0.90, 0.95, 0.99]:
        needed = (cum_energy < threshold).sum(dim=1) + 1
        needed_bins[f"bins_for_{int(threshold * 100)}_mean"] = float(needed.float().mean())
        needed_bins[f"bins_for_{int(threshold * 100)}_p50"] = float(torch.quantile(needed.float(), 0.50))
        needed_bins[f"bins_for_{int(threshold * 100)}_p95"] = float(torch.quantile(needed.float(), 0.95))
    top5_frac = sorted_energy[:, :top_counts["top5pct"]].sum(dim=1) / total_energy

    pca_n = min(pca_samples, n)
    pca_idx = torch.linspace(0, n - 1, steps=pca_n).long()
    x_sample = flat.index_select(0, pca_idx).to(device)
    eigvals, eigvecs = covariance_eig(x_sample, device)
    eig_cpu = eigvals.detach().cpu()
    csi_pca = {
        "sample_size": pca_n,
        "dim": flat.size(1),
        "effective_rank": effective_rank(eig_cpu),
    }
    cumulative = torch.cumsum(eig_cpu, dim=0) / eig_cpu.sum().clamp_min(1e-12)
    for r in [10, 32, 64, 128, 256, 512]:
        csi_pca[f"top{r}_energy"] = float(cumulative[min(r, len(cumulative)) - 1])
    for target in [0.90, 0.95, 0.99]:
        csi_pca[f"rank_{int(target * 100)}"] = int((cumulative < target).sum().item() + 1)

    plt.figure(figsize=(6.8, 4.8))
    plt.hist(power.numpy(), bins=120, color="#4C78A8", alpha=0.85)
    plt.xlabel("sample CSI power")
    plt.ylabel("count")
    plt.title("CSI sample power distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "csi_power_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    labels = ["top1%", "top5%", "top10%"]
    vals = [
        sparsity["top1pct_energy_frac_mean"],
        sparsity["top5pct_energy_frac_mean"],
        sparsity["top10pct_energy_frac_mean"],
    ]
    plt.bar(labels, vals, color="#F58518")
    plt.ylim(0, 1)
    plt.ylabel("mean energy fraction")
    plt.title("Angular-delay energy concentration")
    plt.tight_layout()
    plt.savefig(fig_dir / "csi_energy_concentration.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative.numpy(),
             color="#4C78A8")
    plt.axhline(0.90, color="#54A24B", linestyle="--", linewidth=1)
    plt.axhline(0.95, color="#F58518", linestyle="--", linewidth=1)
    plt.axhline(0.99, color="#E45756", linestyle="--", linewidth=1)
    plt.xlabel("PCA components")
    plt.ylabel("cumulative energy")
    plt.title("Raw CSI PCA spectrum")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "csi_pca_cumulative.png", dpi=180)
    plt.close()

    # K-means on CSI PCA features for cluster-level hard sample analysis.
    pcs = (x_sample - x_sample.mean(dim=0, keepdim=True)).matmul(eigvecs[:, :16])
    pcs = pcs.float()
    centroids = pcs[torch.linspace(0, pcs.size(0) - 1, steps=kmeans_k).long()].clone()
    for _ in range(30):
        dist = torch.cdist(pcs, centroids)
        labels_gpu = dist.argmin(dim=1)
        new_centroids = []
        for k in range(kmeans_k):
            mask = labels_gpu == k
            if bool(mask.any()):
                new_centroids.append(pcs[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = torch.stack(new_centroids, dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-5, rtol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids
    labels_cpu = labels_gpu.detach().cpu()
    cluster_rows = []
    sample_power = power.index_select(0, pca_idx)
    sample_top5 = top5_frac.index_select(0, pca_idx)
    for k in range(kmeans_k):
        mask = labels_cpu == k
        count = int(mask.sum())
        if count == 0:
            continue
        cluster_rows.append({
            "cluster": k,
            "n": count,
            "power_mean": float(sample_power[mask].mean()),
            "power_p95": float(torch.quantile(sample_power[mask], 0.95)),
            "top5_energy_frac_mean": float(sample_top5[mask].mean()),
        })

    return {
        "basic": power_stats,
        "channel_stats": channel_stats,
        "sparsity": {**sparsity, **needed_bins},
        "pca": csi_pca,
        "power": power,
        "top5_frac": top5_frac,
        "pca_idx": pca_idx,
        "cluster_labels": labels_cpu,
        "cluster_rows": cluster_rows,
    }


def code_stats(name, x, device):
    xg = x.to(device)
    eigvals, _ = covariance_eig(xg, device)
    eig_cpu = eigvals.detach().cpu()
    cumulative = torch.cumsum(eig_cpu, dim=0) / eig_cpu.sum().clamp_min(1e-12)
    dim_var = x.var(dim=0, unbiased=False)
    norm = x.norm(dim=1)
    return {
        "name": name,
        "n": int(x.size(0)),
        "dim": int(x.size(1)),
        "global_mean": float(x.mean()),
        "global_std": float(x.std(unbiased=False)),
        "norm_mean": float(norm.mean()),
        "norm_p95": float(torch.quantile(norm, 0.95)),
        "dim_var_mean": float(dim_var.mean()),
        "dim_var_cv": float(dim_var.std(unbiased=False) / dim_var.mean().clamp_min(1e-12)),
        "effective_rank": effective_rank(eig_cpu),
        "top10_energy": float(cumulative[9]),
        "top50_energy": float(cumulative[49]),
        "top100_energy": float(cumulative[99]),
    }


def pair_stats(name, source, target, device):
    source_g = source.to(device)
    target_g = target.to(device)
    diff = source_g - target_g
    per_sample = diff.pow(2).mean(dim=1)
    per_dim = diff.pow(2).mean(dim=0).sqrt()
    cos = F.cosine_similarity(source_g, target_g, dim=1)
    eigvals, _ = covariance_eig(diff, device)
    eig_cpu = eigvals.detach().cpu()
    residual_flat = diff.detach().cpu().reshape(-1)
    if residual_flat.numel() > 1_000_000:
        step = max(1, residual_flat.numel() // 1_000_000)
        idx = torch.arange(0, residual_flat.numel(), step=step)[:1_000_000]
        residual_flat = residual_flat.index_select(0, idx)
    mu = residual_flat.mean()
    sigma = residual_flat.std(unbiased=False).clamp_min(1e-12)
    centered = residual_flat - mu
    kurtosis = float((centered.pow(4).mean() / sigma.pow(4)).item() - 3.0)
    median = torch.median(residual_flat)
    laplace_b = (residual_flat - median).abs().mean().clamp_min(1e-12)
    normal_nll = float((0.5 * math.log(2 * math.pi) + torch.log(sigma) +
                        centered.pow(2).mean() / (2 * sigma.pow(2))).item())
    laplace_nll = float((math.log(2) + torch.log(laplace_b) +
                         (residual_flat - median).abs().mean() / laplace_b).item())
    cumulative = torch.cumsum(eig_cpu, dim=0) / eig_cpu.sum().clamp_min(1e-12)
    return {
        "name": name,
        "mse": float(per_sample.mean().detach().cpu()),
        "rmse": float(per_sample.mean().sqrt().detach().cpu()),
        "mae": float(diff.abs().mean().detach().cpu()),
        "cos_mean": float(cos.mean().detach().cpu()),
        "sample_mse_p50": float(torch.quantile(per_sample, 0.50).detach().cpu()),
        "sample_mse_p95": float(torch.quantile(per_sample, 0.95).detach().cpu()),
        "sample_mse_p99": float(torch.quantile(per_sample, 0.99).detach().cpu()),
        "dim_rmse_mean": float(per_dim.mean().detach().cpu()),
        "dim_rmse_max": float(per_dim.max().detach().cpu()),
        "residual_effective_rank": effective_rank(eig_cpu),
        "residual_top10_energy": float(cumulative[9]),
        "residual_top50_energy": float(cumulative[49]),
        "residual_kurtosis": kurtosis,
        "normal_nll": normal_nll,
        "laplace_nll": laplace_nll,
    }, per_sample.detach().cpu()


def affine_fit_stats(name, source, target, device, ridge):
    x = source.to(device)
    y = target.to(device)
    ones = torch.ones(x.size(0), 1, device=device, dtype=x.dtype)
    x_aug = torch.cat([x, ones], dim=1)
    xtx = x_aug.t().matmul(x_aug) / x_aug.size(0)
    xty = x_aug.t().matmul(y) / x_aug.size(0)
    eye = torch.eye(xtx.size(0), device=device, dtype=xtx.dtype)
    eye[-1, -1] = 0.0
    w = torch.linalg.solve(xtx + ridge * eye, xty)
    pred = x_aug.matmul(w)
    affine_mse = float((pred - y).pow(2).mean().detach().cpu())
    affine_cos = float(F.cosine_similarity(pred, y, dim=1).mean().detach().cpu())

    x_center = x - x.mean(dim=0, keepdim=True)
    y_center = y - y.mean(dim=0, keepdim=True)
    cross = x_center.t().matmul(y_center)
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    r = u.matmul(vh)
    pred_p = x_center.matmul(r) + y.mean(dim=0, keepdim=True)
    procrustes_mse = float((pred_p - y).pow(2).mean().detach().cpu())
    procrustes_cos = float(F.cosine_similarity(pred_p, y, dim=1).mean().detach().cpu())
    raw_mse = float((x - y).pow(2).mean().detach().cpu())
    raw_cos = float(F.cosine_similarity(x, y, dim=1).mean().detach().cpu())
    return {
        "name": name,
        "raw_mse": raw_mse,
        "raw_cos": raw_cos,
        "procrustes_mse": procrustes_mse,
        "procrustes_cos": procrustes_cos,
        "affine_mse": affine_mse,
        "affine_cos": affine_cos,
    }


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_code_figures(code_rows, pair_rows, linear_rows, out_dir):
    fig_dir = out_dir / "figures"
    names = [row["name"] for row in pair_rows]
    mse = [row["mse"] for row in pair_rows]
    cos = [row["cos_mean"] for row in pair_rows]
    x = np.arange(len(names))
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x, mse, color="#4C78A8")
    plt.yscale("log")
    plt.xticks(x, names, rotation=25, ha="right", fontsize=8)
    plt.ylabel("MSE to teacher (log)")
    plt.title("Code pair MSE to teacher")
    plt.tight_layout()
    plt.savefig(fig_dir / "code_pair_mse.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x, cos, color="#54A24B")
    plt.xticks(x, names, rotation=25, ha="right", fontsize=8)
    plt.ylabel("mean cosine")
    plt.title("Code pair cosine to teacher")
    plt.tight_layout()
    plt.savefig(fig_dir / "code_pair_cosine.png", dpi=180)
    plt.close()

    lin_names = [row["name"] for row in linear_rows]
    raw = [row["raw_mse"] for row in linear_rows]
    proc = [row["procrustes_mse"] for row in linear_rows]
    aff = [row["affine_mse"] for row in linear_rows]
    x = np.arange(len(lin_names))
    width = 0.25
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(x - width, raw, width=width, label="raw", color="#4C78A8")
    plt.bar(x, proc, width=width, label="orthogonal", color="#F58518")
    plt.bar(x + width, aff, width=width, label="affine", color="#E45756")
    plt.yscale("log")
    plt.xticks(x, lin_names, rotation=25, ha="right", fontsize=8)
    plt.ylabel("MSE to teacher (log)")
    plt.title("Linear alignability of source codes")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "code_linear_alignment.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    for row in code_rows:
        if row["name"] in {
            "teacher",
            "seed2026_transnet",
            "old_hybrid_seed2026_transnet",
            "smooth_tail_white_seed2026_transnet",
        }:
            plt.scatter(row["effective_rank"], row["top50_energy"],
                        label=row["name"], s=55)
    plt.xlabel("effective rank")
    plt.ylabel("top50 PCA energy")
    plt.title("Code distribution rank summary")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "code_rank_summary.png", dpi=180)
    plt.close()


def plot_cluster_hardness(cluster_rows, out_dir):
    fig_dir = out_dir / "figures"
    rows = sorted(cluster_rows, key=lambda r: r.get("new_mapped_mse_mean", 0), reverse=True)
    labels = [str(r["cluster"]) for r in rows]
    x = np.arange(len(rows))
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(x, [r.get("raw_source_mse_mean", 0) for r in rows],
            label="raw source", color="#4C78A8", alpha=0.7)
    plt.bar(x, [r.get("new_mapped_mse_mean", 0) for r in rows],
            label="smooth+tail+white", color="#E45756", alpha=0.7)
    plt.yscale("log")
    plt.xticks(x, labels)
    plt.xlabel("CSI PCA cluster")
    plt.ylabel("code MSE to teacher (log)")
    plt.title("CSI clusters and code alignment hardness")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "csi_cluster_code_hardness.png", dpi=180)
    plt.close()


def main():
    setup_fonts()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--csi_path", default=DEFAULT_CSI)
    parser.add_argument("--out_dir", type=Path,
                        default=ROOT / "mapper/reports/generative_code_mapping_feasibility/csi_code_deep_dive")
    parser.add_argument("--pca_samples", type=int, default=20000)
    parser.add_argument("--kmeans_k", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = args.out_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    csi = load_tensor(args.csi_path)
    if csi.ndim == 2:
        csi = csi.view(csi.size(0), 2, 32, 32)
    csi_result = csi_analysis(csi, device, out_dir, args.pca_samples, args.kmeans_k)

    teacher = load_tensor(TEACHER_CODE)
    code_rows = [code_stats("teacher", teacher, device)]
    pair_rows = []
    linear_rows = []
    per_sample_maps = {}

    for name, path in SOURCE_CODES.items():
        if not rel(path).exists():
            continue
        source = load_tensor(path)
        code_rows.append(code_stats(name, source, device))
        pair, per_sample = pair_stats(name, source, teacher, device)
        pair_rows.append(pair)
        per_sample_maps[name] = per_sample
        linear_rows.append(affine_fit_stats(name, source, teacher, device, args.ridge))

    for name, path in MAPPED_CODES.items():
        if not rel(path).exists():
            continue
        mapped = load_tensor(path)
        code_rows.append(code_stats(name, mapped, device))
        pair, per_sample = pair_stats(name, mapped, teacher, device)
        pair_rows.append(pair)
        per_sample_maps[name] = per_sample

    # Cluster-level hardness uses the same CSI PCA subset.
    pca_idx = csi_result["pca_idx"]
    labels = csi_result["cluster_labels"]
    cluster_rows = csi_result["cluster_rows"]
    for row in cluster_rows:
        mask = labels == row["cluster"]
        for metric_name, samples in [
            ("raw_source_mse_mean", per_sample_maps.get("seed2026_transnet")),
            ("old_mapped_mse_mean", per_sample_maps.get("old_hybrid_seed2026_transnet")),
            ("new_mapped_mse_mean", per_sample_maps.get("smooth_tail_white_seed2026_transnet")),
        ]:
            if samples is not None:
                values = samples.index_select(0, pca_idx)[mask]
                row[metric_name] = float(values.mean())

    correlations = {}
    for name, samples in per_sample_maps.items():
        correlations[name] = {
            "corr_with_csi_power": pearson(samples, csi_result["power"]),
            "corr_with_top5_energy_frac": pearson(samples, csi_result["top5_frac"]),
        }

    write_csv(out_dir / "csi_channel_stats.csv", csi_result["channel_stats"])
    write_csv(out_dir / "csi_cluster_summary.csv", cluster_rows)
    write_csv(out_dir / "code_stats_full.csv", code_rows)
    write_csv(out_dir / "code_pair_stats_full.csv", pair_rows)
    write_csv(out_dir / "code_linear_alignment.csv", linear_rows)
    plot_code_figures(code_rows, pair_rows, linear_rows, out_dir)
    plot_cluster_hardness(cluster_rows, out_dir)

    summary = {
        "device": str(device),
        "csi": {
            "basic": csi_result["basic"],
            "sparsity": csi_result["sparsity"],
            "pca": csi_result["pca"],
        },
        "code": {
            "code_stats": code_rows,
            "pair_stats": pair_rows,
            "linear_alignment": linear_rows,
            "correlations": correlations,
        },
        "cluster_summary": cluster_rows,
        "figures": [
            str(path.relative_to(out_dir))
            for path in sorted((out_dir / "figures").glob("*.png"))
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
