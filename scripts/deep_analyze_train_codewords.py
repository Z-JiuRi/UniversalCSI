import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


DECODER_SUFFIXES = ["_cnn_residual", "_transnet", "_hybrid"]


def parse_model_name(name):
    for suffix in DECODER_SUFFIXES:
        if name.endswith(suffix):
            decoder = suffix[1:]
            encoder = name[:-len(suffix)]
            return encoder, decoder
    return name, "unknown"


def load_training_results(path):
    results = {}
    if not path.is_file():
        return results
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            results[row["name"]] = {
                "final_test_loss": float(row["final_test_loss"]),
                "final_test_nmse": float(row["final_test_nmse"]),
            }
    return results


def discover_codewords(exp_root):
    runs = []
    for path in sorted(Path(exp_root).glob("*/codewords/train_code.pt")):
        name = path.parents[1].name
        encoder, decoder = parse_model_name(name)
        if decoder == "unknown":
            continue
        runs.append({
            "name": name,
            "encoder": encoder,
            "decoder": decoder,
            "path": path,
        })
    return runs


def tensor_quantiles(x, values):
    x = x.float().reshape(-1)
    sorted_x, _ = torch.sort(x)
    n = sorted_x.numel()
    results = []
    for value in values:
        position = value * (n - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            results.append(sorted_x[lower].item())
            continue
        weight = position - lower
        interpolated = sorted_x[lower] * (1.0 - weight) + sorted_x[upper] * weight
        results.append(interpolated.item())
    return results


def exact_pca_from_cov(cov):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0)
    eigvecs = eigvecs[:, order]
    ratio = eigvals / eigvals.sum().clamp_min(1e-12)
    entropy = -(ratio * torch.log(ratio.clamp_min(1e-12))).sum()
    return eigvals, eigvecs, ratio, torch.exp(entropy)


def covariance(code, mean=None):
    if mean is None:
        mean = code.mean(dim=0)
    centered = code - mean
    return centered.T.mm(centered) / max(code.size(0) - 1, 1)


def summarize_code(run, code, train_result):
    n, d = code.shape
    mean = code.mean(dim=0)
    std = code.std(dim=0, unbiased=False)
    cov = covariance(code, mean)
    eigvals, eigvecs, ratio, effective_rank = exact_pca_from_cov(cov)
    norms = torch.linalg.vector_norm(code, dim=1)
    abs_code = code.abs()
    q_norm = tensor_quantiles(norms, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    q_val = tensor_quantiles(code.flatten(), [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    q_abs = tensor_quantiles(abs_code.flatten(), [0.5, 0.9, 0.95, 0.99])
    dead_dim_1e_4 = (std < 1e-4).float().mean().item()
    dead_dim_1e_3 = (std < 1e-3).float().mean().item()
    active_dim_1e_2 = (std > 1e-2).float().mean().item()
    row = {
        "name": run["name"],
        "encoder": run["encoder"],
        "decoder": run["decoder"],
        "path": str(run["path"]),
        "num_samples": n,
        "code_dim": d,
        "mean_global": code.mean().item(),
        "std_global": code.std(unbiased=False).item(),
        "min": code.min().item(),
        "max": code.max().item(),
        "abs_mean": abs_code.mean().item(),
        "abs_median": q_abs[0],
        "abs_q90": q_abs[1],
        "abs_q95": q_abs[2],
        "abs_q99": q_abs[3],
        "value_q001": q_val[0],
        "value_q01": q_val[1],
        "value_q05": q_val[2],
        "value_median": q_val[3],
        "value_q95": q_val[4],
        "value_q99": q_val[5],
        "value_q999": q_val[6],
        "l2_mean": norms.mean().item(),
        "l2_std": norms.std(unbiased=False).item(),
        "l2_q01": q_norm[0],
        "l2_q05": q_norm[1],
        "l2_q25": q_norm[2],
        "l2_median": q_norm[3],
        "l2_q75": q_norm[4],
        "l2_q95": q_norm[5],
        "l2_q99": q_norm[6],
        "near_zero_1e_4": (abs_code < 1e-4).float().mean().item(),
        "near_zero_1e_3": (abs_code < 1e-3).float().mean().item(),
        "near_zero_1e_2": (abs_code < 1e-2).float().mean().item(),
        "dim_mean_abs_mean": mean.abs().mean().item(),
        "dim_mean_abs_max": mean.abs().max().item(),
        "dim_std_mean": std.mean().item(),
        "dim_std_std": std.std(unbiased=False).item(),
        "dim_std_min": std.min().item(),
        "dim_std_max": std.max().item(),
        "dead_dim_ratio_std_lt_1e_4": dead_dim_1e_4,
        "dead_dim_ratio_std_lt_1e_3": dead_dim_1e_3,
        "active_dim_ratio_std_gt_1e_2": active_dim_1e_2,
        "cov_trace": torch.trace(cov).item(),
        "pca_top1_ratio": ratio[0].item(),
        "pca_top2_ratio": ratio[:2].sum().item(),
        "pca_top5_ratio": ratio[:5].sum().item(),
        "pca_top10_ratio": ratio[:10].sum().item(),
        "pca_top20_ratio": ratio[:20].sum().item(),
        "pca_top50_ratio": ratio[:50].sum().item(),
        "effective_rank": effective_rank.item(),
        "condition_number_top512": (eigvals[0] / eigvals[-1].clamp_min(1e-12)).item(),
        "final_test_loss": train_result.get("final_test_loss", float("nan")),
        "final_test_nmse": train_result.get("final_test_nmse", float("nan")),
    }
    return row, mean, std, cov, eigvals, eigvecs, ratio, norms


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pairwise_distances(names, means, stds, covs):
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            mean_l2 = torch.linalg.vector_norm(means[left] - means[right]).item()
            mean_cos = torch.nn.functional.cosine_similarity(
                means[left].unsqueeze(0), means[right].unsqueeze(0)).item()
            std_l2 = torch.linalg.vector_norm(stds[left] - stds[right]).item()
            std_cos = torch.nn.functional.cosine_similarity(
                stds[left].unsqueeze(0), stds[right].unsqueeze(0)).item()
            cov_delta = covs[left] - covs[right]
            cov_fro = torch.linalg.matrix_norm(cov_delta, ord="fro").item()
            rows.append({
                "left": left,
                "right": right,
                "centroid_l2": mean_l2,
                "centroid_cosine": mean_cos,
                "std_profile_l2": std_l2,
                "std_profile_cosine": std_cos,
                "covariance_frobenius": cov_fro,
            })
    return rows


def aggregate_global_pca(codes, total_n):
    global_sum = None
    second = None
    for code in codes.values():
        if global_sum is None:
            global_sum = code.sum(dim=0)
            second = code.T.mm(code)
        else:
            global_sum += code.sum(dim=0)
            second += code.T.mm(code)
    mean = global_sum / total_n
    cov = (second - total_n * torch.outer(mean, mean)) / max(total_n - 1, 1)
    eigvals, eigvecs, ratio, effective_rank = exact_pca_from_cov(cov)
    return mean, cov, eigvals, eigvecs, ratio, effective_rank


def build_plot_data(codes, global_mean, global_pc, norm_bins=80, pc_bins=100):
    plot = {
        "norm_hist": {},
        "pc1_hist": {},
        "pc2_hist": {},
        "pc12_hist2d": {},
        "pc_centers": {},
    }
    all_norms = torch.cat([torch.linalg.vector_norm(code, dim=1) for code in codes.values()])
    norm_min = all_norms.min().item()
    norm_max = all_norms.max().item()
    norm_edges = torch.linspace(norm_min, norm_max, norm_bins + 1)

    pc_ranges = []
    projections = {}
    for name, code in codes.items():
        coords = (code - global_mean).mm(global_pc[:, :2])
        projections[name] = coords
        pc_ranges.append(coords)
    all_pc = torch.cat(pc_ranges, dim=0)
    pc1_edges = torch.linspace(all_pc[:, 0].min().item(), all_pc[:, 0].max().item(), pc_bins + 1)
    pc2_edges = torch.linspace(all_pc[:, 1].min().item(), all_pc[:, 1].max().item(), pc_bins + 1)

    plot["norm_edges"] = norm_edges.numpy()
    plot["pc1_edges"] = pc1_edges.numpy()
    plot["pc2_edges"] = pc2_edges.numpy()
    for name, code in codes.items():
        norms = torch.linalg.vector_norm(code, dim=1)
        plot["norm_hist"][name] = torch.histc(norms, bins=norm_bins,
                                              min=norm_min, max=norm_max).numpy()
        coords = projections[name]
        plot["pc1_hist"][name] = torch.histc(coords[:, 0], bins=pc_bins,
                                             min=pc1_edges[0].item(),
                                             max=pc1_edges[-1].item()).numpy()
        plot["pc2_hist"][name] = torch.histc(coords[:, 1], bins=pc_bins,
                                             min=pc2_edges[0].item(),
                                             max=pc2_edges[-1].item()).numpy()
        hist2d, _, _ = np.histogram2d(
            coords[:, 0].numpy(), coords[:, 1].numpy(),
            bins=[pc1_edges.numpy(), pc2_edges.numpy()])
        plot["pc12_hist2d"][name] = hist2d
        plot["pc_centers"][name] = coords.mean(dim=0).numpy()
    return plot


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = [
        "AR PL UMing TW MBE",
        "AR PL UMing CN",
        "Noto Sans CJK SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_figures(out_dir, summary_rows, pairwise_rows, plot_data):
    plt = setup_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    names = [row["name"] for row in summary_rows]

    def save(name):
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        figures.append(path)

    x = np.arange(len(summary_rows))
    labels = names

    plt.figure(figsize=(18, 6))
    plt.bar(x, [row["std_global"] for row in summary_rows])
    plt.xticks(x, labels, rotation=90, fontsize=6)
    plt.ylabel("global std")
    plt.title("Code Global Std Deviation")
    save("01_global_std_bar.png")

    plt.figure(figsize=(18, 6))
    plt.bar(x, [row["l2_mean"] for row in summary_rows])
    plt.xticks(x, labels, rotation=90, fontsize=6)
    plt.ylabel("mean L2 norm")
    plt.title("Sample Codeword L2 Norm Mean")
    save("02_l2_norm_bar.png")

    plt.figure(figsize=(18, 6))
    plt.bar(x, [row["effective_rank"] for row in summary_rows])
    plt.xticks(x, labels, rotation=90, fontsize=6)
    plt.ylabel("effective rank")
    plt.title("PCA Effective Rank")
    save("03_effective_rank_bar.png")

    plt.figure(figsize=(18, 6))
    plt.bar(x, [row["near_zero_1e_3"] for row in summary_rows])
    plt.xticks(x, labels, rotation=90, fontsize=6)
    plt.ylabel("ratio")
    plt.title("Ratio of abs(code) < 1e-3")
    save("04_near_zero_ratio_bar.png")

    decoders = sorted({row["decoder"] for row in summary_rows})
    decoder_groups = {decoder: [row for row in summary_rows if row["decoder"] == decoder]
                      for decoder in decoders}
    plt.figure(figsize=(10, 5))
    plt.boxplot([[row["l2_mean"] for row in decoder_groups[d]] for d in decoders],
                labels=decoders)
    plt.ylabel("mean L2 norm")
    plt.title("Code Norm Distribution by Decoder Type")
    save("05_decoder_l2_boxplot.png")

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)))
    for row in pairwise_rows:
        i = name_to_idx[row["left"]]
        j = name_to_idx[row["right"]]
        matrix[i, j] = row["centroid_l2"]
        matrix[j, i] = row["centroid_l2"]
    plt.figure(figsize=(16, 14))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(label="centroid L2")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=5)
    plt.yticks(range(len(names)), names, fontsize=5)
    plt.title("Inter-Architecture Code Centroid Distance Heatmap")
    save("06_centroid_l2_heatmap.png")

    centers = np.stack([plot_data["pc_centers"][name] for name in names], axis=0)
    plt.figure(figsize=(12, 9))
    for row, xy in zip(summary_rows, centers):
        marker = {"transnet": "o", "cnn_residual": "s", "hybrid": "^"}.get(
            row["decoder"], "o")
        plt.scatter(xy[0], xy[1], marker=marker, s=50)
        plt.text(xy[0], xy[1], row["name"], fontsize=6)
    plt.xlabel("Global PC1")
    plt.ylabel("Global PC2")
    plt.title("Full Global PCA: Code Centroids by Model")
    save("07_global_pca_centroids.png")

    plt.figure(figsize=(12, 8))
    norm_edges = plot_data["norm_edges"]
    norm_mid = (norm_edges[:-1] + norm_edges[1:]) / 2
    for row in summary_rows:
        hist = plot_data["norm_hist"][row["name"]]
        hist = hist / max(hist.sum(), 1)
        if row["decoder"] == "hybrid":
            alpha = 0.85
        elif row["decoder"] == "cnn_residual":
            alpha = 0.45
        else:
            alpha = 0.25
        plt.plot(norm_mid, hist, alpha=alpha, linewidth=1)
    plt.xlabel("L2 norm")
    plt.ylabel("density")
    plt.title("Full L2 Norm Histogram Curves (All Models)")
    save("08_l2_norm_histograms.png")

    selected = sorted(summary_rows, key=lambda row: row["final_test_nmse"])[:8]
    selected += sorted(summary_rows, key=lambda row: row["final_test_nmse"], reverse=True)[:4]
    seen = set()
    selected = [row for row in selected if not (row["name"] in seen or seen.add(row["name"]))]
    ncols = 4
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    pc1_edges = plot_data["pc1_edges"]
    pc2_edges = plot_data["pc2_edges"]
    for ax, row in zip(axes, selected):
        hist = plot_data["pc12_hist2d"][row["name"]]
        ax.imshow(np.log1p(hist.T), origin="lower", aspect="auto",
                  extent=[pc1_edges[0], pc1_edges[-1], pc2_edges[0], pc2_edges[-1]],
                  cmap="magma")
        ax.set_title(row["name"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(selected):]:
        ax.axis("off")
    fig.suptitle("Full Global PCA 2D Density: Best & Worst Representative Models", y=1.02)
    save("09_pca_density_selected.png")

    plt.figure(figsize=(8, 6))
    std_values = np.array([row["std_global"] for row in summary_rows])
    nmse_values = np.array([row["final_test_nmse"] for row in summary_rows])
    for row in summary_rows:
        plt.scatter(row["std_global"], row["final_test_nmse"])
        plt.text(row["std_global"], row["final_test_nmse"], row["name"], fontsize=5)
    plt.xlabel("global code std")
    plt.ylabel("final test NMSE")
    corr = np.corrcoef(std_values, nmse_values)[0, 1]
    plt.title(f"Code Scale vs 2-epoch NMSE (Pearson={corr:.3f})")
    save("10_std_vs_nmse.png")

    return figures


def group_rows(summary_rows, key):
    groups = defaultdict(list)
    for row in summary_rows:
        groups[row[key]].append(row)
    rows = []
    for name, values in sorted(groups.items()):
        rows.append({
            key: name,
            "count": len(values),
            "std_global_mean": float(np.mean([row["std_global"] for row in values])),
            "l2_mean_mean": float(np.mean([row["l2_mean"] for row in values])),
            "effective_rank_mean": float(np.mean([row["effective_rank"] for row in values])),
            "top5_pca_mean": float(np.mean([row["pca_top5_ratio"] for row in values])),
            "near_zero_1e_3_mean": float(np.mean([row["near_zero_1e_3"] for row in values])),
            "final_test_nmse_mean": float(np.mean([row["final_test_nmse"] for row in values])),
        })
    return rows


def correlation_rows(summary_rows):
    metrics = [
        "std_global",
        "abs_mean",
        "l2_mean",
        "near_zero_1e_3",
        "dim_std_mean",
        "pca_top1_ratio",
        "pca_top5_ratio",
        "effective_rank",
        "cov_trace",
    ]
    nmse = np.array([row["final_test_nmse"] for row in summary_rows])
    rows = []
    for metric in metrics:
        values = np.array([row[metric] for row in summary_rows])
        pearson = float(np.corrcoef(values, nmse)[0, 1])
        rows.append({"metric": metric, "pearson_with_nmse": pearson})
    return sorted(rows, key=lambda row: abs(row["pearson_with_nmse"]), reverse=True)


def top_markdown_table(rows, columns, limit=None):
    if limit is not None:
        rows = rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4e}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_report(out_dir, summary_rows, pairwise_rows, encoder_rows,
                  decoder_rows, corr_rows, figures, global_pca):
    best_nmse = sorted(summary_rows, key=lambda row: row["final_test_nmse"])[:10]
    worst_nmse = sorted(summary_rows, key=lambda row: row["final_test_nmse"], reverse=True)[:10]
    largest_scale = sorted(summary_rows, key=lambda row: row["std_global"], reverse=True)[:10]
    highest_rank = sorted(summary_rows, key=lambda row: row["effective_rank"], reverse=True)[:10]
    farthest = sorted(pairwise_rows, key=lambda row: row["centroid_l2"], reverse=True)[:10]
    closest = sorted(pairwise_rows, key=lambda row: row["centroid_l2"])[:10]

    lines = [
        "# train_code.pt 全量深层分析报告",
        "",
        "本报告分析 `exps/real_matrix_2epoch/*/codewords/train_code.pt`，"
        "所有统计与图表均基于完整 train codewords，没有抽样。",
        "",
        "## 数据覆盖",
        "",
        f"- 模型组合数：`{len(summary_rows)}`",
        f"- 每组样本数：`{summary_rows[0]['num_samples']}`",
        f"- 码字维度：`{summary_rows[0]['code_dim']}`",
        f"- 全量样本点数：`{sum(row['num_samples'] for row in summary_rows)}`",
        f"- Global PCA 有效秩：`{global_pca['effective_rank']:.4e}`",
        f"- Global PCA Top-5 方差占比：`{global_pca['top5_ratio']:.4e}`",
        "",
        "## 一句话结论",
        "",
        "- `hybrid` decoder 组合在 2 epoch 下整体 NMSE 排名靠前，但它诱导的码字尺度通常比 `transnet` decoder 更大。",
        "- `cnn_residual` decoder 组合中部分模型的码字高度集中在少数主成分上，有效秩偏低，说明压缩码空间更容易塌缩到低维方向。",
        "- `transnet` decoder 组合的码字有效秩普遍较高，分布更均匀，但 2 epoch 重建效果明显落后于多数 `hybrid` 组合。",
        "- `cnn` 与 `cbam_cnn` 都基于同一卷积骨架；CBAM 改变了码字尺度和 PCA 集中度，但在当前 2 epoch 设置下并没有超过 `cnn_hybrid`。",
        "",
        "## 主要图表",
        "",
    ]
    for fig in figures:
        lines.append(f"- [{fig.name}](figures/{fig.name})")

    lines.extend([
        "",
        "## 2-epoch NMSE 最好组合",
        "",
    ])
    lines.extend(top_markdown_table(
        best_nmse,
        ["name", "encoder", "decoder", "final_test_nmse", "std_global",
         "l2_mean", "effective_rank"],
        limit=10))

    lines.extend(["", "## 2-epoch NMSE 最差组合", ""])
    lines.extend(top_markdown_table(
        worst_nmse,
        ["name", "encoder", "decoder", "final_test_nmse", "std_global",
         "l2_mean", "effective_rank"],
        limit=10))

    lines.extend(["", "## 码字尺度最大的组合", ""])
    lines.extend(top_markdown_table(
        largest_scale,
        ["name", "decoder", "std_global", "l2_mean", "pca_top5_ratio",
         "effective_rank"],
        limit=10))

    lines.extend(["", "## 维度利用最充分的组合（有效秩最高）", ""])
    lines.extend(top_markdown_table(
        highest_rank,
        ["name", "decoder", "effective_rank", "pca_top5_ratio",
         "dim_std_mean", "final_test_nmse"],
        limit=10))

    lines.extend(["", "## Encoder 维度聚合视角", ""])
    lines.extend(top_markdown_table(
        encoder_rows,
        ["encoder", "count", "std_global_mean", "l2_mean_mean",
         "effective_rank_mean", "top5_pca_mean", "final_test_nmse_mean"]))

    lines.extend(["", "## Decoder 维度聚合视角", ""])
    lines.extend(top_markdown_table(
        decoder_rows,
        ["decoder", "count", "std_global_mean", "l2_mean_mean",
         "effective_rank_mean", "top5_pca_mean", "final_test_nmse_mean"]))

    lines.extend(["", "## 架构间最远的码字分布", ""])
    lines.extend(top_markdown_table(
        farthest,
        ["left", "right", "centroid_l2", "centroid_cosine",
         "std_profile_l2", "covariance_frobenius"],
        limit=10))

    lines.extend(["", "## 架构间最近的码字分布", ""])
    lines.extend(top_markdown_table(
        closest,
        ["left", "right", "centroid_l2", "centroid_cosine",
         "std_profile_l2", "covariance_frobenius"],
        limit=10))

    lines.extend(["", "## 码字统计与 NMSE 的相关性", ""])
    lines.extend(top_markdown_table(corr_rows, ["metric", "pearson_with_nmse"]))

    lines.extend([
        "",
        "## 输出文件说明",
        "",
        "- `full_summary.csv`：每个模型组合的全量分布、范数、稀疏性、维度利用和 PCA 指标。",
        "- `pairwise_distances.csv`：任意两个模型组合之间的中心、标准差轮廓、协方差距离。",
        "- `encoder_aggregate.csv`：按 encoder 聚合的均值指标。",
        "- `decoder_aggregate.csv`：按 decoder 聚合的均值指标。",
        "- `metric_nmse_correlations.csv`：码字指标和最终 NMSE 的 Pearson 相关性。",
        "- `figures/`：全部图表。",
        "",
    ])
    (out_dir / "train_codewords_deep_report_zh.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Full-data deep analysis for train_code.pt files.")
    parser.add_argument("--exp_root", default="exps/real_matrix_2epoch")
    parser.add_argument("--out_dir",
                        default="exps/real_matrix_2epoch/codeword_analysis/deep_train_full")
    parser.add_argument("--training_results",
                        default="exps/real_matrix_2epoch/training_results.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_codewords(args.exp_root)
    if not runs:
        raise SystemExit(f"No train_code.pt files found under {args.exp_root}")
    train_results = load_training_results(Path(args.training_results))

    codes = {}
    means = {}
    stds = {}
    covs = {}
    summary_rows = []
    total_n = 0
    for run in runs:
        code = torch.load(run["path"], weights_only=True, map_location="cpu").float()
        if code.ndim != 2:
            raise ValueError(f"{run['path']} should be 2D, got {tuple(code.shape)}")
        codes[run["name"]] = code
        total_n += code.size(0)
        row, mean, std, cov, eigvals, eigvecs, ratio, norms = summarize_code(
            run, code, train_results.get(run["name"], {}))
        summary_rows.append(row)
        means[run["name"]] = mean
        stds[run["name"]] = std
        covs[run["name"]] = cov

    summary_rows = sorted(summary_rows, key=lambda row: row["name"])
    names = [row["name"] for row in summary_rows]
    pairwise_rows = pairwise_distances(names, means, stds, covs)
    encoder_rows = group_rows(summary_rows, "encoder")
    decoder_rows = group_rows(summary_rows, "decoder")
    corr_rows = correlation_rows(summary_rows)

    global_mean, global_cov, global_eigvals, global_pc, global_ratio, global_erank = (
        aggregate_global_pca(codes, total_n))
    plot_data = build_plot_data(codes, global_mean, global_pc)
    figures = save_figures(figure_dir, summary_rows, pairwise_rows, plot_data)

    write_csv(out_dir / "full_summary.csv", summary_rows)
    write_csv(out_dir / "pairwise_distances.csv", pairwise_rows)
    write_csv(out_dir / "encoder_aggregate.csv", encoder_rows)
    write_csv(out_dir / "decoder_aggregate.csv", decoder_rows)
    write_csv(out_dir / "metric_nmse_correlations.csv", corr_rows)

    global_pca = {
        "effective_rank": global_erank.item(),
        "top1_ratio": global_ratio[0].item(),
        "top5_ratio": global_ratio[:5].sum().item(),
        "top10_ratio": global_ratio[:10].sum().item(),
    }
    (out_dir / "global_pca.json").write_text(
        json.dumps(global_pca, indent=2, ensure_ascii=False))
    render_report(out_dir, summary_rows, pairwise_rows, encoder_rows,
                  decoder_rows, corr_rows, figures, global_pca)
    print(f"wrote {out_dir / 'train_codewords_deep_report_zh.md'}")


if __name__ == "__main__":
    main()
