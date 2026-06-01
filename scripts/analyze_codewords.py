import argparse
import csv
import json
import math
from pathlib import Path

import torch


def load_code(path, max_samples=None):
    code = torch.load(path, weights_only=True, map_location="cpu")
    code = code.float()
    if max_samples is not None and code.size(0) > max_samples:
        idx = torch.linspace(0, code.size(0) - 1, steps=max_samples).long()
        code = code[idx]
    return code


def safe_float(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def summarize_code(code):
    norms = torch.linalg.vector_norm(code, dim=1)
    dim_mean = code.mean(dim=0)
    dim_std = code.std(dim=0, unbiased=False)
    centered = code - dim_mean
    sample = centered
    if sample.size(0) > 4096:
        idx = torch.linspace(0, sample.size(0) - 1, steps=4096).long()
        sample = sample[idx]
    _, singular_values, _ = torch.pca_lowrank(sample, q=min(32, sample.size(1)))
    var = singular_values.square()
    var_ratio = var / var.sum().clamp_min(1e-12)
    entropy = -(var_ratio * torch.log(var_ratio.clamp_min(1e-12))).sum()
    effective_rank = torch.exp(entropy)
    return {
        "num_samples": code.size(0),
        "code_dim": code.size(1),
        "mean": safe_float(code.mean()),
        "std": safe_float(code.std(unbiased=False)),
        "min": safe_float(code.min()),
        "max": safe_float(code.max()),
        "abs_mean": safe_float(code.abs().mean()),
        "l2_norm_mean": safe_float(norms.mean()),
        "l2_norm_std": safe_float(norms.std(unbiased=False)),
        "near_zero_1e-3": safe_float((code.abs() < 1e-3).float().mean()),
        "dim_mean_abs_mean": safe_float(dim_mean.abs().mean()),
        "dim_std_mean": safe_float(dim_std.mean()),
        "dim_std_std": safe_float(dim_std.std(unbiased=False)),
        "pca_top1_ratio": safe_float(var_ratio[0]),
        "pca_top5_ratio": safe_float(var_ratio[:5].sum()),
        "pca_top10_ratio": safe_float(var_ratio[:10].sum()),
        "effective_rank_32": safe_float(effective_rank),
    }


def discover_runs(exp_root, split):
    runs = []
    exp_root = Path(exp_root)
    for code_path in sorted(exp_root.glob(f"**/codewords/{split}_code.pt")):
        name = code_path.relative_to(exp_root).parts[0]
        if name.endswith("_cnn_residual"):
            encoder = name[:-len("_cnn_residual")]
            decoder = "cnn_residual"
        elif name.endswith("_transnet"):
            encoder = name[:-len("_transnet")]
            decoder = "transnet"
        elif name.endswith("_hybrid"):
            encoder = name[:-len("_hybrid")]
            decoder = "hybrid"
        else:
            continue
        runs.append({
            "name": name,
            "encoder": encoder,
            "decoder": decoder,
            "path": code_path,
        })
    return runs


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pairwise_rows(samples):
    rows = []
    names = sorted(samples)
    for i, left in enumerate(names):
        a = samples[left]
        a_mean = a.mean(dim=0)
        a_std = a.std(dim=0, unbiased=False)
        for right in names[i + 1:]:
            b = samples[right]
            b_mean = b.mean(dim=0)
            b_std = b.std(dim=0, unbiased=False)
            centroid_l2 = torch.linalg.vector_norm(a_mean - b_mean)
            centroid_cos = torch.nn.functional.cosine_similarity(
                a_mean.unsqueeze(0), b_mean.unsqueeze(0)).item()
            std_l2 = torch.linalg.vector_norm(a_std - b_std)
            std_cos = torch.nn.functional.cosine_similarity(
                a_std.unsqueeze(0), b_std.unsqueeze(0)).item()
            rows.append({
                "left": left,
                "right": right,
                "centroid_l2": safe_float(centroid_l2),
                "centroid_cosine": safe_float(centroid_cos),
                "std_profile_l2": safe_float(std_l2),
                "std_profile_cosine": safe_float(std_cos),
            })
    return rows


def plot_heatmap(pairwise, out_path, metric):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = sorted({row["left"] for row in pairwise} |
                   {row["right"] for row in pairwise})
    index = {name: idx for idx, name in enumerate(names)}
    matrix = torch.zeros(len(names), len(names))
    for row in pairwise:
        i = index[row["left"]]
        j = index[row["right"]]
        value = float(row[metric])
        matrix[i, j] = value
        matrix[j, i] = value
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35),
                                    max(7, len(names) * 0.35)))
    im = ax.imshow(matrix.numpy(), cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title(metric)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def plot_pca(samples, out_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = sorted(samples)
    chunks = []
    labels = []
    for name in names:
        code = samples[name]
        chunks.append(code)
        labels.extend([name] * code.size(0))
    data = torch.cat(chunks, dim=0)
    mean = data.mean(dim=0)
    centered = data - mean
    _, _, v = torch.pca_lowrank(centered, q=2)
    coords = centered @ v[:, :2]
    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 0
    for name in names:
        count = samples[name].size(0)
        xy = coords[offset:offset + count]
        ax.scatter(xy[:, 0].numpy(), xy[:, 1].numpy(), s=4, alpha=0.45,
                   label=name)
        offset += count
    ax.set_title("Codeword PCA projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=5, ncol=2, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def render_markdown(out_dir, split, summary_rows, pairwise, plots):
    lines = [
        "# Codeword Architecture Analysis",
        "",
        f"Split: `{split}`",
        f"Runs analyzed: `{len(summary_rows)}`",
        "",
        "## What Was Measured",
        "",
        "- Global code distribution: mean, std, min/max, absolute mean.",
        "- Per-sample code norm: average and spread of L2 norms.",
        "- Sparsity proxy: fraction of values with `abs(code) < 1e-3`.",
        "- Dimension usage: mean per-dimension std and its spread.",
        "- PCA concentration: top-1/top-5/top-10 variance ratios and effective rank.",
        "- Pairwise architecture distance: centroid L2/cosine and std-profile L2/cosine.",
        "",
        "## Summary Table",
        "",
        "| model | encoder | decoder | samples | dim | mean | std | l2_norm | near_zero | top5_pca | eff_rank32 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['name']} | {row['encoder']} | {row['decoder']} | "
            f"{row['num_samples']} | {row['code_dim']} | "
            f"{row['mean']:.4e} | {row['std']:.4e} | "
            f"{row['l2_norm_mean']:.4e} | {row['near_zero_1e-3']:.4e} | "
            f"{row['pca_top5_ratio']:.4e} | {row['effective_rank_32']:.4e} |"
        )
    lines.extend(["", "## Largest Pairwise Centroid Distances", ""])
    top = sorted(pairwise, key=lambda row: row["centroid_l2"], reverse=True)[:20]
    lines.extend([
        "| left | right | centroid_l2 | centroid_cosine | std_profile_l2 |",
        "|---|---|---:|---:|---:|",
    ])
    for row in top:
        lines.append(
            f"| {row['left']} | {row['right']} | "
            f"{row['centroid_l2']:.4e} | {row['centroid_cosine']:.4e} | "
            f"{row['std_profile_l2']:.4e} |"
        )
    lines.extend(["", "## Generated Plots", ""])
    for plot in plots:
        lines.append(f"- [{plot.name}]({plot.name})")
    lines.extend([
        "",
        "## Files",
        "",
        "- `summary.csv`: per-model distribution and PCA metrics.",
        "- `pairwise_distances.csv`: pairwise code distribution distances.",
        "- `analysis_config.json`: input settings for this report.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze UniversalCSI codewords across model architectures.")
    parser.add_argument("--exp_root", default="exps/real_matrix_2epoch")
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--out_dir", default="exps/real_matrix_2epoch/codeword_analysis")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--plot_samples", type=int, default=400)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.exp_root, args.split)
    if not runs:
        raise SystemExit(f"No codewords found under {args.exp_root}")

    summary_rows = []
    samples = {}
    for run in runs:
        code = load_code(run["path"], max_samples=args.max_samples)
        summary = summarize_code(code)
        row = {
            "name": run["name"],
            "encoder": run["encoder"],
            "decoder": run["decoder"],
            "path": str(run["path"]),
            **summary,
        }
        summary_rows.append(row)
        samples[run["name"]] = load_code(run["path"],
                                         max_samples=args.plot_samples)

    summary_rows = sorted(summary_rows, key=lambda row: row["name"])
    summary_fields = list(summary_rows[0].keys())
    write_csv(out_dir / "summary.csv", summary_rows, summary_fields)

    pairwise = pairwise_rows(samples)
    pairwise_fields = list(pairwise[0].keys()) if pairwise else []
    if pairwise:
        write_csv(out_dir / "pairwise_distances.csv", pairwise, pairwise_fields)

    plots = []
    heatmap = out_dir / "centroid_l2_heatmap.png"
    if pairwise and plot_heatmap(pairwise, heatmap, "centroid_l2"):
        plots.append(heatmap)
    pca_plot = out_dir / "pca_projection.png"
    if plot_pca(samples, pca_plot):
        plots.append(pca_plot)

    config = vars(args)
    config["runs"] = [
        {key: str(value) for key, value in run.items()}
        for run in runs
    ]
    (out_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False))
    render_markdown(out_dir, args.split, summary_rows, pairwise, plots)
    print(f"wrote analysis to {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
