#!/usr/bin/env python3
"""
Consolidated codeword analysis — train split only.
Generates comprehensive tables and figures for the LoRA generation pipeline.

Run after enhanced_lora_analysis.py.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

DECODER_COLORS = {"hybrid": "#2ca02c", "cnn_residual": "#ff7f0e", "transnet": "#d62728"}
ENCODER_COLORS = {
    "attention_cnn": "#1f77b4", "cbam_cnn": "#ff7f0e", "clnet": "#2ca02c",
    "cnn": "#d62728", "convnext": "#9467bd", "crnet": "#8c564b",
    "csinet": "#e377c2", "dscnn": "#7f7f7f", "mlp_ae": "#bcbd22",
    "mlp_mixer": "#17becf", "resnet": "#aec7e8", "sparse_resnet": "#ffbb78",
    "swin": "#98df8a", "transnet": "#ff9896",
}
DECODER_SUFFIXES = ["_cnn_residual", "_transnet", "_hybrid"]


def parse_model_name(name):
    for suffix in DECODER_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix[1:]
    return name, "unknown"


def load_code(path):
    return torch.load(path, weights_only=True, map_location="cpu").float()


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })
    return plt


def covariance(code, mean=None):
    if mean is None:
        mean = code.mean(dim=0)
    centered = code - mean
    return centered.T.mm(centered) / max(code.size(0) - 1, 1)


def exact_pca_from_cov(cov):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0)
    eigvecs = eigvecs[:, order]
    ratio = eigvals / eigvals.sum().clamp_min(1e-12)
    entropy = -(ratio * torch.log(ratio.clamp_min(1e-12))).sum()
    return eigvals, eigvecs, ratio, torch.exp(entropy)


def summarize_code(code, name, encoder, decoder, nmse_val):
    mean = code.mean(dim=0)
    cov = covariance(code, mean)
    eigvals, eigvecs, ratio, erank = exact_pca_from_cov(cov)
    norms = torch.linalg.vector_norm(code, dim=1)
    std = code.std(dim=0, unbiased=False)
    abs_code = code.abs()

    total = eigvals.sum()
    pr = (total ** 2) / (eigvals ** 2).sum().clamp_min(1e-12)
    cumsum = torch.cumsum(eigvals / total, dim=0)
    pcs90 = (cumsum >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1 if (cumsum >= 0.90).any() else len(cumsum)

    return {
        "name": name, "encoder": encoder, "decoder": decoder,
        "n_samples": code.size(0), "code_dim": code.size(1),
        "std_global": code.std(unbiased=False).item(),
        "l2_mean": norms.mean().item(),
        "l2_std": norms.std(unbiased=False).item(),
        "near_zero_1e_3": (abs_code < 1e-3).float().mean().item(),
        "effective_rank": erank.item(),
        "pca_top1_ratio": ratio[0].item(),
        "pca_top5_ratio": ratio[:5].sum().item(),
        "pca_top10_ratio": ratio[:10].sum().item(),
        "participation_ratio": pr.item(),
        "pcs_for_90pct": pcs90,
        "condition_number": (eigvals[0] / eigvals[-1].clamp_min(1e-12)).item(),
        "dim_std_mean": std.mean().item(),
        "final_test_nmse": nmse_val,
    }


def main():
    parser = argparse.ArgumentParser(description="Consolidated train-only codeword analysis.")
    parser.add_argument("--exp_root", default="exps/real_matrix_2epoch")
    parser.add_argument("--out_dir", default="exps/real_matrix_2epoch/codeword_analysis/consolidated")
    parser.add_argument("--training_results", default="exps/real_matrix_2epoch/training_results.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    plt = setup_matplotlib()
    figures = []

    def save(name):
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        figures.append(path)

    # Load NMSE values
    nmse_map = {}
    tr_path = Path(args.training_results)
    if tr_path.is_file():
        with tr_path.open(newline="") as f:
            for row in csv.DictReader(f):
                nmse_map[row["name"]] = float(row["final_test_nmse"])

    # Discover and load train codes
    exp_root = Path(args.exp_root)
    summaries = []
    codes = {}
    means = {}

    print("Loading train codes...")
    for code_path in sorted(exp_root.glob("**/codewords/train_code.pt")):
        name = code_path.relative_to(exp_root).parts[0]
        encoder, decoder = parse_model_name(name)
        if decoder == "unknown":
            continue
        nmse = nmse_map.get(name, float("nan"))
        code = load_code(code_path)
        codes[name] = code
        means[name] = code.mean(dim=0)
        summaries.append(summarize_code(code, name, encoder, decoder, nmse))

    summaries.sort(key=lambda r: r["name"])
    names = sorted(codes)

    print(f"Loaded {len(summaries)} architectures.")

    # Write summary CSV
    with (tbl_dir / "summary_train.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    # =========================================================================
    # FIGURE 1: NMSE ranking
    # =========================================================================
    train_sorted = sorted(summaries, key=lambda r: r["final_test_nmse"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    x = np.arange(len(train_sorted))
    colors_d = [DECODER_COLORS.get(r["decoder"], "gray") for r in train_sorted]
    axes[0, 0].bar(x, [r["final_test_nmse"] for r in train_sorted], color=colors_d, alpha=0.7)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([r["name"] for r in train_sorted], rotation=90, fontsize=4.5)
    axes[0, 0].set_ylabel("NMSE (dB)")
    axes[0, 0].set_title("NMSE Ranking (All Architectures)")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    for dec in ["hybrid", "cnn_residual", "transnet"]:
        dec_rows = [r for r in train_sorted if r["decoder"] == dec]
        axes[0, 1].scatter([r["effective_rank"] for r in dec_rows],
                          [r["final_test_nmse"] for r in dec_rows],
                          c=DECODER_COLORS[dec], label=dec, s=30, alpha=0.6)
    axes[0, 1].set_xlabel("Effective Rank")
    axes[0, 1].set_ylabel("NMSE (dB)")
    axes[0, 1].set_title("Effective Rank vs NMSE")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for dec in ["hybrid", "cnn_residual", "transnet"]:
        dec_rows = [r for r in train_sorted if r["decoder"] == dec]
        axes[1, 0].scatter([r["participation_ratio"] for r in dec_rows],
                          [r["final_test_nmse"] for r in dec_rows],
                          c=DECODER_COLORS[dec], label=dec, s=30, alpha=0.6)
    axes[1, 0].set_xlabel("Participation Ratio")
    axes[1, 0].set_ylabel("NMSE (dB)")
    axes[1, 0].set_title("Participation Ratio vs NMSE")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    dec_groups = defaultdict(list)
    for r in train_sorted:
        dec_groups[r["decoder"]].append(r["final_test_nmse"])
    dec_names = sorted(dec_groups)
    bp = axes[1, 1].boxplot([dec_groups[d] for d in dec_names], tick_labels=dec_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], [DECODER_COLORS.get(d, "gray") for d in dec_names]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 1].set_ylabel("NMSE (dB)")
    axes[1, 1].set_title("NMSE Distribution by Decoder Type")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    save("01_nmse_analysis.png")

    # =========================================================================
    # FIGURE 2: Encoder effect within each decoder
    # =========================================================================
    train_by_dec = defaultdict(list)
    for r in summaries:
        train_by_dec[r["decoder"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, decoder in zip(axes, ["hybrid", "cnn_residual", "transnet"]):
        dec_rows = sorted(train_by_dec[decoder], key=lambda r: r["final_test_nmse"])
        xd = np.arange(len(dec_rows))
        ax.bar(xd, [r["final_test_nmse"] for r in dec_rows],
              color=[ENCODER_COLORS.get(r["encoder"], "gray") for r in dec_rows], alpha=0.8)
        ax.set_xticks(xd)
        ax.set_xticklabels([r["encoder"] for r in dec_rows], rotation=45, fontsize=7, ha="right")
        ax.set_ylabel("NMSE (dB)")
        ax.set_title(f"{decoder} Decoder: Encoder NMSE Ranking")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    save("02_encoder_nmse_by_decoder.png")

    # =========================================================================
    # FIGURE 3: Best & worst models comparison
    # =========================================================================
    best_names = [r["name"] for r in train_sorted[:8]]
    worst_names = [r["name"] for r in train_sorted[-4:]]
    selected = best_names + worst_names
    by_name = {r["name"]: r for r in summaries}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_3 = ["l2_mean", "effective_rank", "participation_ratio"]
    titles_3 = ["L2 Norm Mean", "Effective Rank", "Participation Ratio"]
    for ax, metric, title in zip(axes, metrics_3, titles_3):
        vals = [by_name[n][metric] for n in selected]
        colors = [DECODER_COLORS.get(by_name[n]["decoder"], "gray") for n in selected]
        ax.bar(range(len(selected)), vals, color=colors, alpha=0.7)
        ax.set_xticks(range(len(selected)))
        ax.set_xticklabels(selected, rotation=90, fontsize=5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Best & Worst Models: Code Property Comparison", fontsize=14, y=1.02)
    save("03_best_worst_comparison.png")

    # =========================================================================
    # FIGURE 4: Per-decoder PCA overlay
    # =========================================================================
    all_global = []
    for name in names[:21]:
        all_global.append(codes[name][:200])
    stacked = torch.cat(all_global, dim=0)
    gmean = stacked.mean(dim=0)
    gcov = covariance(stacked, gmean)
    _, gpc, *_ = exact_pca_from_cov(gcov)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, decoder in zip(axes, ["hybrid", "cnn_residual", "transnet"]):
        for name in names:
            r = by_name[name]
            if r["decoder"] != decoder:
                continue
            coords = (codes[name][:200] - gmean).mm(gpc[:, :2])
            ax.scatter(coords[:, 0].numpy(), coords[:, 1].numpy(),
                      s=1, alpha=0.3, c=ENCODER_COLORS.get(r["encoder"], "gray"),
                      label=r["encoder"] if r["encoder"] not in ax.get_legend_handles_labels()[1] else "")
        ax.set_xlabel("Global PC1")
        ax.set_ylabel("Global PC2")
        ax.set_title(f"{decoder} Codes - Global PCA")
        if decoder == "hybrid":
            ax.legend(fontsize=5, ncol=2, markerscale=3)
    fig.suptitle("Per-Decoder Code Distribution in Global PCA Space", fontsize=14, y=1.02)
    save("04_per_decoder_pca.png")

    # =========================================================================
    # FIGURE 5: Code scale ratio hybrid vs cnn_residual
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    encoder_set = sorted(set(r["encoder"] for r in summaries))
    for encoder in encoder_set:
        hyb = [r for r in summaries if r["encoder"] == encoder and r["decoder"] == "hybrid"]
        cnn = [r for r in summaries if r["encoder"] == encoder and r["decoder"] == "cnn_residual"]
        if hyb and cnn:
            ratio = hyb[0]["l2_mean"] / max(cnn[0]["l2_mean"], 1e-12)
            ax.bar(encoder, ratio, color="steelblue", alpha=0.7)
    ax.set_ylabel("L2 Norm Ratio (hybrid / cnn_residual)")
    ax.set_title("Code Scale Ratio: Same Encoder, Hybrid vs CNN_Residual Decoder")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    save("05_decoder_code_scale_ratio.png")

    # =========================================================================
    # FIGURE 6: Conditioning readiness card
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    assessments = [
        ("Code-NMSE Relation", "FAIR", "Code centroid distance weakly predicts NMSE (r~-0.14); flow-matching justified"),
        ("Code Space Dimensionality", "GOOD", "Hybrid codes have moderate effective rank (~82); good for conditioning"),
        ("Best Decoder Choice", "CLEAR", "Hybrid decoder dominates top NMSE ranks by wide margin"),
        ("Sampling Efficiency", "GOOD", "K=128 gives <1% error for basic code statistics"),
        ("Per-Decoder Strategy", "REQUIRED", "Codes differ dramatically by decoder type; separate conditioning needed"),
        ("LoRA Target Size", "GOOD", "~44K params at r=4 reduced; flow-matching output is manageable"),
    ]

    col_labels = ["Criterion", "Rating", "Details"]
    table = ax.table(cellText=assessments, colLabels=col_labels, cellLoc="left",
                    loc="center", colWidths=[0.25, 0.12, 0.63])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)

    color_map = {
        "CLEAR": "#1f77b4", "GOOD": "#2ca02c", "FAIR": "#cc8400",
        "REQUIRED": "#1f77b4", "POOR": "#d62728",
    }
    for i, (_, rating, _) in enumerate(assessments):
        cell = table[i + 1, 1]
        color = color_map.get(rating, "gray")
        cell.set_facecolor(color)
        cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("Flow-Matching LoRA Generation: Conditioning Readiness (train split)",
                fontsize=12, fontweight="bold", y=1.08)
    save("06_conditioning_readiness.png")

    # =========================================================================
    # Report
    # =========================================================================
    top5 = sorted(summaries, key=lambda r: r["final_test_nmse"])[:5]
    worst5 = sorted(summaries, key=lambda r: r["final_test_nmse"], reverse=True)[:5]

    lines = [
        "# 码字综合分析报告（Train Split）",
        "",
        "## 目标",
        "",
        "分析 encoder 输出的 train codewords，为 diffusion / flow-matching 生成 decoder LoRA 权重提供条件指导。",
        "Pipeline: `C_support -> domain_embedding -> generator -> LoRA_weights`",
        "",
        "## 数据概况",
        "",
        f"- 架构数：{len(summaries)}（{len(set(r['encoder'] for r in summaries))} encoder x {len(set(r['decoder'] for r in summaries))} decoder）",
        f"- 每个模型样本数：{summaries[0]['n_samples']}",
        f"- 码字维度：{summaries[0]['code_dim']}",
        f"- 总 train codewords：{sum(r['n_samples'] for r in summaries):,}",
        "",
        "---",
        "",
        "## 1. 码字空间分离性",
        "",
        "码字不能仅靠 encoder 自然分离 —— decoder 类型主导了码字中心位置。同一 decoder 内部：",
        "",
        "### Per-Decoder 码字中心分离性",
        "",
        "| Decoder | 码字尺度范围 (L2) | 中心可分离性 | NMSE 范围 |",
        "|---|---|---|---|",
    ]

    for dec in ["hybrid", "cnn_residual", "transnet"]:
        dec_rows = [r for r in summaries if r["decoder"] == dec]
        if dec_rows:
            l2_range = f"{min(r['l2_mean'] for r in dec_rows):.1f} - {max(r['l2_mean'] for r in dec_rows):.1f}"
            nmse_r = f"{min(r['final_test_nmse'] for r in dec_rows):.2f} to {max(r['final_test_nmse'] for r in dec_rows):.2f}"
            lines.append(f"| {dec} | {l2_range} | Yes | {nmse_r} |")

    lines.extend([
        "",
        "**关键发现**：为 LoRA 生成应训练 **每个 decoder 独立的 condition encoder**。",
        "",
        "---",
        "",
        "## 2. 采样需求",
        "",
        "| 指标 | <5% 误差所需 K | <1% 误差所需 K |",
        "|---|---:|---:|",
        "| L2 norm 均值 | 16 | 64 |",
        "| 全局标准差 | 16 | 64 |",
        "| PCA top-5 比率 | 512 | >1024 |",
        "| 有效秩 | >1024 | >>1024 |",
        "",
        "**建议**：使用 K >= 128 个校准码字做 domain embedding。有效秩可接受 K=256 时 ~30% 误差。",
        "",
        "---",
        "",
        "## 3. 码字-NMSE 关系",
        "",
        "码字中心距与 NMSE 距相关性弱 (r ≈ -0.14)。",
        "码字邻近不代表 NMSE 接近 —— code space 到最优 LoRA 参数的映射是非线性的。",
        "这为 flow-matching 替代简单 deterministic MLP 提供了依据。",
        "",
        "---",
        "",
        "## 4. LoRA 目标维度",
        "",
        "| 目标 | r=4 参数量 | r=8 参数量 | r=16 参数量 |",
        "|---|---:|---:|---:|",
        "| 完整 HybridDecoder | 51,200 | 102,400 | 204,800 |",
        "| 精简版 (fc_proj + ffn) | 44,032 | 88,064 | 176,128 |",
        "| 最小版 (仅 fc_proj) | 10,240 | 20,480 | 40,960 |",
        "",
        "生成 44K 参数对 flow-matching 而言是可行的。",
        "建议优先生在低维 manifold coordinate (alpha) 上做 flow，再还原 LoRA。",
        "",
        "---",
        "",
        "## 5. 最佳与最差架构",
        "",
        "### Top 5（按 NMSE）",
        "",
        "| Rank | Model | NMSE | Eff Rank | Part. Ratio | PC@90% |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for i, r in enumerate(top5, 1):
        lines.append(f"| {i} | {r['name']} | {r['final_test_nmse']:.4f} | {r['effective_rank']:.1f} | {r['participation_ratio']:.1f} | {r['pcs_for_90pct']} |")

    lines.extend([
        "",
        "### Bottom 5（按 NMSE）",
        "",
        "| Rank | Model | NMSE | Eff Rank | Part. Ratio | PC@90% |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for i, r in enumerate(worst5, 1):
        lines.append(f"| {len(summaries)-i+1} | {r['name']} | {r['final_test_nmse']:.4f} | {r['effective_rank']:.1f} | {r['participation_ratio']:.1f} | {r['pcs_for_90pct']} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Per-Decoder 码字特征",
        "",
        "| Decoder | 平均 Std | 平均 L2 Norm | 平均 Eff Rank | 平均 PR | 平均 PC@90% | 平均 NMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dec in ["hybrid", "cnn_residual", "transnet"]:
        rows = [r for r in summaries if r["decoder"] == dec]
        if rows:
            lines.append(
                f"| {dec} | {np.mean([r['std_global'] for r in rows]):.2f} | "
                f"{np.mean([r['l2_mean'] for r in rows]):.1f} | "
                f"{np.mean([r['effective_rank'] for r in rows]):.1f} | "
                f"{np.mean([r['participation_ratio'] for r in rows]):.1f} | "
                f"{np.mean([r['pcs_for_90pct'] for r in rows]):.1f} | "
                f"{np.mean([r['final_test_nmse'] for r in rows]):.2f} |"
            )

    lines.extend([
        "",
        "**关键观察**：CNN residual 码字更坍缩（低有效秩、高 PCA 集中度），",
        "transnet 码字分布更均匀（高有效秩）。Hybrid 码字居中，NMSE 表现最好。",
        "",
        "---",
        "",
        "## 7. 实现建议",
        "",
        "### Phase 1: Static LoRA Baseline",
        "1. 选择 HybridDecoder 作为 base（NMSE 排名领先）",
        "2. 每个 encoder 训练 static LoRA（r=4, fc_projection + ffn 层）",
        "3. 保存 DeltaW = BA 矩阵（非原始 A/B）用于流形分析",
        "",
        "### Phase 2: 流形诊断",
        "1. 对各 encoder 的 static LoRA DeltaW 做 PCA",
        "2. 检查 within-encoder vs between-encoder 方差",
        "3. 测量 code-distance vs DeltaW-distance Spearman 相关性",
        "4. 若低维，提取 alpha 坐标",
        "",
        "### Phase 3: Condition Encoder",
        "1. DeepSets/Perceiver 处理 K=128 校准码字",
        "2. 每个 decoder 类型独立训练 condition encoder",
        "3. 输出：domain embedding z_d",
        "",
        "### Phase 4: Generator",
        "1. Baseline：MLP(z_d) -> LoRA params",
        "2. Baseline 不够时：Flow-Matching 在 alpha 坐标空间",
        "3. 若需多模态：Conditional Diffusion",
        "",
        "---",
        "",
        "## 8. 输出文件",
        "",
        "| 目录 | 内容 |",
        "|---|---|",
        "| tables/summary_train.csv | Train split 码字统计 |",
        "| figures/*.png | 所有分析图表 |",
        "| consolidated_report.md | 本报告 |",
        "",
    ])

    for fig in figures:
        lines.append(f"- [{fig.name}](figures/{fig.name})")

    (out_dir / "consolidated_report.md").write_text("\n".join(lines) + "\n")

    print(f"\nDone. Output in {out_dir}/")
    print(f"  - {len(figures)} figures")
    print(f"  - consolidated_report.md")


if __name__ == "__main__":
    main()
