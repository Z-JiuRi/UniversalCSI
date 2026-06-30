#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_TEACHER = "exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt"
DEFAULT_OUT = "mapper/reports/mapper_distribution_analysis"
FONT_PATHS = [
    "/home/hujiacong/zxd/.envs/SongTi.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def setup_matplotlib():
    for path in FONT_PATHS:
        if Path(path).exists():
            matplotlib.font_manager.fontManager.addfont(path)
            prop = matplotlib.font_manager.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def load_code(path):
    code = torch.load(path, weights_only=True, map_location="cpu")
    if code.ndim != 2:
        raise ValueError(f"{path} should be 2D, got {tuple(code.shape)}")
    return code.to(torch.float32)


def source_name(path):
    match = re.search(
        r"exps/COST2100/in/seed(\d+)/([^/]+)/codewords/train_code.pt",
        str(path))
    if match:
        return f"seed{match.group(1)}_{match.group(2)}"
    return Path(path).parent.parent.name


def normal_pdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi))


def laplace_pdf(x, loc, scale):
    scale = max(float(scale), 1e-12)
    return np.exp(-np.abs(x - loc) / scale) / (2.0 * scale)


def vector_stats(x):
    x = x.reshape(-1).to(torch.float64)
    mean = x.mean()
    centered = x - mean
    std = centered.pow(2).mean().sqrt().clamp_min(1e-12)
    skew = centered.pow(3).mean() / std.pow(3)
    kurt = centered.pow(4).mean() / std.pow(4)
    median = x.median()
    mad = (x - median).abs().median()
    laplace_b = (x - median).abs().mean().clamp_min(1e-12)
    normal_nll = 0.5 * math.log(2.0 * math.pi) + torch.log(std) + 0.5
    laplace_nll = math.log(2.0) + torch.log(laplace_b) + (
        (x - median).abs().mean() / laplace_b)
    tail3 = (x.abs() > 3.0 * std).to(torch.float32).mean()
    tail5 = (x.abs() > 5.0 * std).to(torch.float32).mean()
    return {
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "mad": float(mad),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "laplace_b": float(laplace_b),
        "normal_nll": float(normal_nll),
        "laplace_nll": float(laplace_nll),
        "tail_abs_gt_3std": float(tail3),
        "tail_abs_gt_5std": float(tail5),
    }


def pair_stats(pred, target):
    diff = pred - target
    mse = diff.pow(2).mean()
    rmse_dim = diff.pow(2).mean(dim=0).sqrt()
    sample_l2 = diff.pow(2).sum(dim=1).sqrt()
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    return {
        "mse": float(mse),
        "rmse": float(mse.sqrt()),
        "cos_mean": float(cos.mean()),
        "cos_p05": float(cos.quantile(0.05)),
        "cos_p50": float(cos.quantile(0.50)),
        "cos_p95": float(cos.quantile(0.95)),
        "sample_l2_mean": float(sample_l2.mean()),
        "sample_l2_p95": float(sample_l2.quantile(0.95)),
        "dim_rmse_mean": float(rmse_dim.mean()),
        "dim_rmse_p95": float(rmse_dim.quantile(0.95)),
        "dim_rmse_max": float(rmse_dim.max()),
    }


def covariance_eigs(x):
    x = x.to(torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.t().matmul(x) / max(x.size(0) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
    total = eig.sum().clamp_min(1e-12)
    p = eig / total
    eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum())
    return eig.to(torch.float32), float(eff_rank)


def sample_flat(x, max_points, rng):
    flat = x.reshape(-1)
    n = flat.numel()
    if n <= max_points:
        return flat.numpy()
    idx = torch.from_numpy(rng.choice(n, size=max_points, replace=False))
    return flat[idx].numpy()


def plot_value_distribution(out_path, source, mapped, teacher, title, max_points, rng):
    arrays = {
        "teacher": sample_flat(teacher, max_points, rng),
        "source/raw": sample_flat(source, max_points, rng),
        "mapped": sample_flat(mapped, max_points, rng),
    }
    low = np.quantile(np.concatenate(list(arrays.values())), 0.001)
    high = np.quantile(np.concatenate(list(arrays.values())), 0.999)
    bins = np.linspace(low, high, 180)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, values in arrays.items():
        ax.hist(values, bins=bins, density=True, alpha=0.28, label=name)
    ax.set_title(title)
    ax.set_xlabel("code value")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_residual_fit(out_path, raw_diff, mapped_diff, title, max_points, rng):
    raw = sample_flat(raw_diff, max_points, rng)
    mapped = sample_flat(mapped_diff, max_points, rng)
    low = np.quantile(np.concatenate([raw, mapped]), 0.001)
    high = np.quantile(np.concatenate([raw, mapped]), 0.999)
    bins = np.linspace(low, high, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, name, values in [
        (axes[0], "raw - teacher", raw),
        (axes[1], "mapped - teacher", mapped),
    ]:
        mu = values.mean()
        sigma = values.std()
        loc = np.median(values)
        scale = np.mean(np.abs(values - loc))
        xs = np.linspace(np.quantile(values, 0.001),
                         np.quantile(values, 0.999), 500)
        ax.hist(values, bins=bins, density=True, alpha=0.35,
                label="empirical")
        ax.plot(xs, normal_pdf(xs, mu, sigma), label="Normal fit", lw=1.7)
        ax.plot(xs, laplace_pdf(xs, loc, scale), label="Laplace fit", lw=1.7)
        ax.set_title(name)
        ax.set_xlabel("residual")
        ax.set_ylabel("density")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_dim_rmse(out_path, raw_diff, mapped_diff, title):
    raw_rmse = raw_diff.pow(2).mean(dim=0).sqrt().sort(descending=True).values
    mapped_rmse = mapped_diff.pow(2).mean(dim=0).sqrt().sort(descending=True).values
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(raw_rmse.numpy(), label="raw residual dim RMSE")
    ax.plot(mapped_rmse.numpy(), label="mapped residual dim RMSE")
    ax.set_title(title)
    ax.set_xlabel("dimension sorted by RMSE")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pca_spectrum(out_path, eigs, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, eig in eigs.items():
        y = eig.numpy()
        y = y / max(y.sum(), 1e-12)
        axes[0].semilogy(y[:128], label=name)
        axes[1].plot(np.cumsum(y)[:128], label=name)
    axes[0].set_title("归一化 eigen spectrum")
    axes[0].set_xlabel("PC index")
    axes[0].set_ylabel("eigenvalue ratio")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("累计方差占比")
    axes[1].set_xlabel("PC index")
    axes[1].set_ylabel("cumulative ratio")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scatter_summary(out_path, rows):
    rows = [r for r in rows if r.get("decoder_nmse_db") is not None]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    markers = {"mlp": "o", "hybrid": "s", "flow": "^"}
    for row in rows:
        ax.scatter(row["mapped_mse"], row["decoder_nmse_db"],
                   marker=markers.get(row["mapper"], "o"), s=55,
                   label=row["mapper"])
        ax.annotate(row["source"].replace("seed2026_", "").replace("_transnet", ""),
                    (row["mapped_mse"], row["decoder_nmse_db"]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys())
    ax.set_xscale("log")
    ax.set_xlabel("mapped code MSE")
    ax.set_ylabel("fixed decoder NMSE (dB)")
    ax.set_title("code MSE 与 fixed decoder NMSE 的关系")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_decoder_nmse():
    result = {}
    root = Path("mapper/reports/decoder_nmse")
    if not root.exists():
        return result
    for path in root.glob("*.json"):
        data = json.loads(path.read_text())
        result[str(data["code_path"])] = data
    return result


def read_exp_rows():
    rows = []
    for args_path in sorted(Path("mapper/exps").glob("*/*/args.json")):
        exp_dir = args_path.parent
        mapped_path = exp_dir / "mapped_code.pt"
        metrics_path = exp_dir / "metrics.json"
        if not mapped_path.exists() or not metrics_path.exists():
            continue
        args = json.loads(args_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        rows.append({
            "mapper": exp_dir.parent.name,
            "exp": exp_dir.name,
            "exp_dir": exp_dir,
            "source_path": args["source_code"],
            "source": source_name(args["source_code"]),
            "mapped_path": str(mapped_path),
            "metrics": metrics,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_code", default=DEFAULT_TEACHER)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--max_hist_points", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    setup_matplotlib()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_code(args.teacher_code)
    teacher_eig, teacher_eff_rank = covariance_eigs(teacher)
    decoder_nmse = load_decoder_nmse()
    rows = read_exp_rows()

    summary = []
    seen_mapped_hash = {}
    for row in rows:
        exp_dir = row["exp_dir"]
        source = load_code(row["source_path"])
        mapped = load_code(row["mapped_path"])
        if source.shape != teacher.shape or mapped.shape != teacher.shape:
            raise ValueError(f"shape mismatch in {exp_dir}")

        raw_diff = source - teacher
        mapped_diff = mapped - teacher
        raw_pair = pair_stats(source, teacher)
        mapped_pair = pair_stats(mapped, teacher)
        raw_dist = vector_stats(raw_diff)
        mapped_dist = vector_stats(mapped_diff)
        source_val_dist = vector_stats(source)
        mapped_val_dist = vector_stats(mapped)
        teacher_val_dist = vector_stats(teacher)
        source_eig, source_eff_rank = covariance_eigs(source)
        mapped_eig, mapped_eff_rank = covariance_eigs(mapped)

        fig_prefix = f"{row['mapper']}_{row['exp']}"
        plot_value_distribution(
            fig_dir / f"{fig_prefix}_value_distribution.png",
            source, mapped, teacher,
            f"{row['mapper']} / {row['source']} code value distribution",
            args.max_hist_points, rng)
        plot_residual_fit(
            fig_dir / f"{fig_prefix}_residual_fit.png",
            raw_diff, mapped_diff,
            f"{row['mapper']} / {row['source']} residual distribution fit",
            args.max_hist_points, rng)
        plot_dim_rmse(
            fig_dir / f"{fig_prefix}_dim_rmse.png",
            raw_diff, mapped_diff,
            f"{row['mapper']} / {row['source']} residual RMSE by dimension")
        plot_pca_spectrum(
            fig_dir / f"{fig_prefix}_pca_spectrum.png",
            {
                "teacher": teacher_eig,
                "source/raw": source_eig,
                "mapped": mapped_eig,
            },
            f"{row['mapper']} / {row['source']} covariance spectrum")

        dec = decoder_nmse.get(row["mapped_path"], {})
        item = {
            "mapper": row["mapper"],
            "exp": row["exp"],
            "source": row["source"],
            "source_path": row["source_path"],
            "mapped_path": row["mapped_path"],
            "raw_mse": raw_pair["mse"],
            "raw_cos": raw_pair["cos_mean"],
            "mapped_mse": mapped_pair["mse"],
            "mapped_cos": mapped_pair["cos_mean"],
            "mapped_dim_rmse_p95": mapped_pair["dim_rmse_p95"],
            "mapped_dim_rmse_max": mapped_pair["dim_rmse_max"],
            "raw_residual_std": raw_dist["std"],
            "raw_residual_kurtosis": raw_dist["kurtosis"],
            "raw_residual_normal_nll": raw_dist["normal_nll"],
            "raw_residual_laplace_nll": raw_dist["laplace_nll"],
            "mapped_residual_std": mapped_dist["std"],
            "mapped_residual_kurtosis": mapped_dist["kurtosis"],
            "mapped_residual_normal_nll": mapped_dist["normal_nll"],
            "mapped_residual_laplace_nll": mapped_dist["laplace_nll"],
            "mapped_residual_tail3": mapped_dist["tail_abs_gt_3std"],
            "mapped_residual_tail5": mapped_dist["tail_abs_gt_5std"],
            "teacher_value_std": teacher_val_dist["std"],
            "source_value_std": source_val_dist["std"],
            "mapped_value_std": mapped_val_dist["std"],
            "teacher_eff_rank": teacher_eff_rank,
            "source_eff_rank": source_eff_rank,
            "mapped_eff_rank": mapped_eff_rank,
            "decoder_nmse_db": dec.get("nmse_db"),
            "decoder_mse_loss": dec.get("mse_loss"),
            "value_distribution_png": str(fig_dir / f"{fig_prefix}_value_distribution.png"),
            "residual_fit_png": str(fig_dir / f"{fig_prefix}_residual_fit.png"),
            "dim_rmse_png": str(fig_dir / f"{fig_prefix}_dim_rmse.png"),
            "pca_spectrum_png": str(fig_dir / f"{fig_prefix}_pca_spectrum.png"),
        }
        summary.append(item)

        if row["mapper"] == "hybrid" and "seed3407" in row["exp"]:
            seen_mapped_hash[row["exp"]] = mapped

    plot_scatter_summary(fig_dir / "mapped_mse_vs_decoder_nmse.png", summary)

    csv_path = out_dir / "distribution_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    json_path = out_dir / "distribution_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved figures: {fig_dir}")


if __name__ == "__main__":
    main()
