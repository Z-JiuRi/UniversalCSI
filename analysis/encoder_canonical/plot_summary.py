#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common import ensure_dir, setup_matplotlib_fonts, safe_float


def savefig(path):
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(path)


def read_csv_if_exists(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def best_metric(df):
    values = df["final_test_nmse"].copy() if "final_test_nmse" in df else np.nan
    values = values.fillna(df["best_nmse"])
    return values


def plot_scheme_nmse(log_df, fig_dir):
    df = log_df.copy()
    df["plot_nmse"] = best_metric(df)
    df = df[np.isfinite(df["plot_nmse"])].copy()
    if df.empty:
        return
    summary = (
        df.groupby(["family", "scheme"], dropna=False)
        .agg(best=("plot_nmse", "min"), mean=("plot_nmse", "mean"),
             n=("plot_nmse", "count"), complete=("complete", "sum"))
        .reset_index()
        .sort_values("best")
    )
    summary["label"] = summary["scheme"].astype(str)
    plt.figure(figsize=(12, max(5, 0.32 * len(summary))))
    ax = sns.barplot(data=summary, y="label", x="best", hue="family",
                     dodge=False, palette="tab10")
    ax.set_title("各方案 best/final NMSE 排名")
    ax.set_xlabel("NMSE (dB, 越低越好)")
    ax.set_ylabel("方案")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    savefig(fig_dir / "scheme_nmse_ranking.png")


def plot_completion(log_df, fig_dir):
    df = log_df.copy()
    summary = (
        df.groupby(["family", "scheme"], dropna=False)
        .agg(total=("rel_path", "count"),
             complete=("complete", "sum"),
             codewords=("codewords_exists", "sum"),
             crashed=("crashed", "sum"))
        .reset_index()
        .sort_values(["family", "scheme"])
    )
    if summary.empty:
        return
    plot_df = summary.melt(
        id_vars=["family", "scheme"],
        value_vars=["total", "complete", "codewords", "crashed"],
        var_name="status",
        value_name="count",
    )
    plot_df["label"] = plot_df["scheme"]
    plt.figure(figsize=(13, max(5, 0.32 * summary.shape[0])))
    ax = sns.barplot(data=plot_df, y="label", x="count", hue="status")
    ax.set_title("实验完成状态统计")
    ax.set_xlabel("数量")
    ax.set_ylabel("方案")
    savefig(fig_dir / "experiment_completion_status.png")


def plot_encoder_heatmap(log_df, fig_dir):
    df = log_df[(log_df["family"] == "encoder")].copy()
    if df.empty:
        return
    df["plot_nmse"] = best_metric(df)
    table = df.pivot_table(index="scheme", columns="seed", values="plot_nmse",
                           aggfunc="min")
    if table.empty:
        return
    table = table.loc[table.min(axis=1).sort_values().index]
    plt.figure(figsize=(max(7, 0.7 * table.shape[1]), max(5, 0.35 * table.shape[0])))
    ax = sns.heatmap(table, annot=True, fmt=".2f", cmap="viridis_r",
                     cbar_kws={"label": "NMSE (dB)"})
    ax.set_title("Encoder canonical 自编码 NMSE 热力图")
    ax.set_xlabel("seed")
    ax.set_ylabel("scheme")
    savefig(fig_dir / "encoder_nmse_heatmap.png")


def plot_adapter_summary(log_df, fig_dir):
    df = log_df[log_df["family"].astype(str).str.startswith("adapter")].copy()
    if df.empty:
        return
    df["plot_nmse"] = best_metric(df)
    df = df[np.isfinite(df["plot_nmse"])].copy()
    if df.empty:
        return
    df["short_name"] = df["rel_path"].apply(lambda x: "/".join(str(x).split("/")[-3:]))
    top = df.sort_values("plot_nmse").head(40).copy()
    plt.figure(figsize=(12, max(5, 0.34 * len(top))))
    ax = sns.barplot(data=top, y="short_name", x="plot_nmse", hue="adapter",
                     dodge=False)
    ax.set_title("Adapter 实验 NMSE Top 排名")
    ax.set_xlabel("NMSE (dB, 越低越好)")
    ax.set_ylabel("实验")
    savefig(fig_dir / "adapter_nmse_top.png")

    cols = [c for c in [
        "last_code_loss", "last_fc_loss", "last_adapter_delta_ratio",
        "last_adapter_gate_mean", "plot_nmse"
    ] if c in df.columns]
    if len(cols) >= 3:
        plot_df = df[cols + ["adapter"]].dropna()
        if not plot_df.empty:
            sns.pairplot(plot_df, hue="adapter", corner=True,
                         plot_kws={"s": 24, "alpha": 0.75})
            plt.suptitle("Adapter 指标关系图", y=1.02)
            savefig(fig_dir / "adapter_metric_pairplot.png")


def plot_codeword_stats(code_df, pair_df, fig_dir):
    if not code_df.empty:
        code_df = code_df.copy()
        for col in ["best_nmse", "final_test_nmse", "effective_rank",
                    "cov_offdiag_ratio", "global_std", "sample_norm_mean"]:
            if col in code_df.columns:
                code_df[col] = code_df[col].map(safe_float)
        code_df["plot_nmse"] = code_df["final_test_nmse"].fillna(code_df["best_nmse"])

        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=code_df,
            x="effective_rank",
            y="plot_nmse",
            hue="family",
            style="adapter",
            s=70,
        )
        ax.set_title("码字有效秩与 NMSE")
        ax.set_xlabel("effective rank")
        ax.set_ylabel("NMSE (dB)")
        savefig(fig_dir / "code_effective_rank_vs_nmse.png")

        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(
            data=code_df,
            x="cov_offdiag_ratio",
            y="plot_nmse",
            hue="family",
            style="adapter",
            s=70,
        )
        ax.set_title("码字协方差非对角强度与 NMSE")
        ax.set_xlabel("offdiag / diag Frobenius")
        ax.set_ylabel("NMSE (dB)")
        ax.set_xscale("symlog")
        savefig(fig_dir / "code_cov_offdiag_vs_nmse.png")

        summary = (
            code_df.groupby(["family", "scheme"], dropna=False)
            .agg(effective_rank=("effective_rank", "mean"),
                 offdiag=("cov_offdiag_ratio", "mean"),
                 std=("global_std", "mean"),
                 n=("rel_path", "count"))
            .reset_index()
            .sort_values("effective_rank", ascending=False)
        )
        plt.figure(figsize=(12, max(5, 0.32 * len(summary))))
        ax = sns.barplot(data=summary, y="scheme", x="effective_rank",
                         hue="family", dodge=False)
        ax.set_title("各方案码字平均有效秩")
        ax.set_xlabel("effective rank")
        ax.set_ylabel("方案")
        savefig(fig_dir / "scheme_effective_rank.png")

    if not pair_df.empty:
        pair_df = pair_df.copy()
        pair_df["cos_mean"] = pair_df["cos_mean"].map(safe_float)
        summary = (
            pair_df.groupby(["family", "scheme"], dropna=False)
            .agg(cos_mean=("cos_mean", "mean"),
                 cos_min=("cos_mean", "min"),
                 cos_max=("cos_mean", "max"),
                 pairs=("cos_mean", "count"))
            .reset_index()
            .sort_values("cos_mean", ascending=False)
        )
        plt.figure(figsize=(12, max(5, 0.32 * len(summary))))
        ax = sns.barplot(data=summary, y="scheme", x="cos_mean",
                         hue="family", dodge=False)
        ax.set_title("同方案码字 pairwise cosine 平均值")
        ax.set_xlabel("mean same-sample cosine")
        ax.set_ylabel("方案")
        ax.set_xlim(-0.1, 1.02)
        savefig(fig_dir / "scheme_pairwise_code_cosine.png")


def write_markdown_stub(log_df, code_df, pair_df, out_dir):
    md_path = out_dir / "report_auto_summary.md"
    completed = int(log_df["complete"].sum()) if "complete" in log_df else 0
    total = len(log_df)
    best_rows = log_df.copy()
    best_rows["plot_nmse"] = best_metric(best_rows)
    best_rows = best_rows[np.isfinite(best_rows["plot_nmse"])].sort_values("plot_nmse").head(15)

    lines = [
        "# Encoder Canonical 自动分析摘要",
        "",
        f"- 已发现实验数：{total}",
        f"- 已完成 final test：{completed}",
        f"- 已有码字统计实验数：{len(code_df)}",
        "",
        "## 当前 NMSE Top 15",
        "",
        "| 排名 | NMSE | family | scheme | 实验 |",
        "|---:|---:|---|---|---|",
    ]
    for idx, row in enumerate(best_rows.itertuples(), 1):
        lines.append(
            f"| {idx} | {row.plot_nmse:.3f} | {row.family} | {row.scheme} | `{row.rel_path}` |"
        )
    lines.extend([
        "",
        "## 主要图表",
        "",
        "- `figures/scheme_nmse_ranking.png`",
        "- `figures/encoder_nmse_heatmap.png`",
        "- `figures/adapter_nmse_top.png`",
        "- `figures/code_effective_rank_vs_nmse.png`",
        "- `figures/code_cov_offdiag_vs_nmse.png`",
        "- `figures/scheme_pairwise_code_cosine.png`",
        "",
        "说明：该文件由脚本自动生成，实验未跑完时只代表当前已完成/已有日志的结果。",
    ])
    md_path.write_text("\n".join(lines))
    print(md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="analysis_outputs/encoder_canonical")
    parser.add_argument("--fig-dir", default="analysis_outputs/encoder_canonical/figures")
    args = parser.parse_args()

    setup_matplotlib_fonts()
    sns.set_theme(style="whitegrid")
    setup_matplotlib_fonts()

    in_dir = Path(args.in_dir)
    fig_dir = ensure_dir(args.fig_dir)
    log_df = read_csv_if_exists(in_dir / "experiment_log_summary.csv")
    code_df = read_csv_if_exists(in_dir / "codeword_stats.csv")
    pair_df = read_csv_if_exists(in_dir / "codeword_pairwise.csv")
    if log_df.empty:
        raise SystemExit(f"Missing or empty {in_dir / 'experiment_log_summary.csv'}")

    plot_scheme_nmse(log_df, fig_dir)
    plot_completion(log_df, fig_dir)
    plot_encoder_heatmap(log_df, fig_dir)
    plot_adapter_summary(log_df, fig_dir)
    plot_codeword_stats(code_df, pair_df, fig_dir)
    write_markdown_stub(log_df, code_df, pair_df, in_dir)


if __name__ == "__main__":
    main()

