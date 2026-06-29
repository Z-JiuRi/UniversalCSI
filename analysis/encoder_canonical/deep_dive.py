#!/usr/bin/env python
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import ensure_dir, setup_matplotlib_fonts


def savefig(path):
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def nmse_gain(new, ref):
    if pd.isna(new) or pd.isna(ref):
        return np.nan
    return new - ref


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def format_cell(value, floatfmt=".4f"):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), floatfmt)
    return str(value)


def markdown_table(df, columns=None, max_rows=None, floatfmt=".4f"):
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return ""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_cell(row[c], floatfmt) for c in df.columns) + " |")
    return "\n".join([header, sep] + rows)


def encoder_rows(log_df):
    df = log_df[log_df["family"] == "encoder"].copy()
    df["nmse"] = df["best_nmse"].where(df["best_nmse"].notna(), df["final_test_nmse"])
    return df


def adapter_rows(log_df):
    df = log_df[log_df["family"].astype(str).str.startswith("adapter/")].copy()
    df["nmse"] = df["best_nmse"].where(df["best_nmse"].notna(), df["final_test_nmse"])
    return df


def build_encoder_scheme_summary(enc):
    group = enc.groupby("scheme", dropna=False)
    rows = []
    for scheme, g in group:
        rows.append({
            "scheme": scheme,
            "n": len(g),
            "complete": int(g["complete"].sum()),
            "nmse_mean": g["nmse"].mean(),
            "nmse_std": g["nmse"].std(),
            "nmse_min": g["nmse"].min(),
            "nmse_max": g["nmse"].max(),
            "seed_std_transnet": g[g["encoder"] == "transnet"]["nmse"].std(),
            "seed_range_transnet": (
                g[g["encoder"] == "transnet"]["nmse"].max()
                - g[g["encoder"] == "transnet"]["nmse"].min()
            ),
            "lambda_anchor": g["lambda_anchor"].dropna().iloc[0]
            if g["lambda_anchor"].notna().any() else np.nan,
            "lambda_code_mean": g["lambda_code_mean"].dropna().iloc[0]
            if g["lambda_code_mean"].notna().any() else np.nan,
            "lambda_code_cov": g["lambda_code_cov"].dropna().iloc[0]
            if g["lambda_code_cov"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("nmse_mean")


def build_adapter_summary(adapter, enc):
    decoder42 = enc[
        (enc["scheme"].notna())
        & (enc["seed"].astype(str) == "42")
        & (enc["encoder"] == "transnet")
        & (enc["decoder"] == "transnet")
    ][["scheme", "nmse"]].rename(columns={"nmse": "decoder42_self_nmse"})

    rows = []
    for _, row in adapter.iterrows():
        scheme_enc = enc[
            (enc["scheme"] == row["scheme"])
            & (enc["seed"].astype(str) == str(row["seed"]))
            & (enc["encoder"] == row["encoder"])
            & (enc["decoder"] == row["decoder"])
        ]
        source_nmse = scheme_enc["nmse"].iloc[0] if not scheme_enc.empty else np.nan
        rows.append({
            "rel_path": row["rel_path"],
            "scheme": row["scheme"],
            "adapter": row["adapter"],
            "seed": row["seed"],
            "encoder": row["encoder"],
            "decoder": row["decoder"],
            "epochs": row["epochs"],
            "nmse": row["nmse"],
            "source_self_nmse": source_nmse,
            "loss_code": row.get("last_code_loss", np.nan),
            "loss_fc": row.get("last_fc_loss", np.nan),
            "delta_ratio": row.get("last_adapter_delta_ratio", np.nan),
            "gate_mean": row.get("last_adapter_gate_mean", np.nan),
            "gate_std": row.get("last_adapter_gate_std", np.nan),
            "lowrank_ratio": row.get("last_adapter_lowrank_ratio", np.nan),
            "mlp_ratio": row.get("last_adapter_mlp_ratio", np.nan),
        })
    out = pd.DataFrame(rows)
    out = out.merge(decoder42, on="scheme", how="left")
    out["adapter_vs_source_nmse_delta"] = out.apply(
        lambda r: nmse_gain(r["nmse"], r["source_self_nmse"]), axis=1)
    out["adapter_vs_decoder42_nmse_delta"] = out.apply(
        lambda r: nmse_gain(r["nmse"], r["decoder42_self_nmse"]), axis=1)
    return out.sort_values("nmse")


def build_code_join(enc, code_stats):
    if code_stats.empty:
        return pd.DataFrame()
    code = code_stats.merge(
        enc[["rel_path", "nmse", "scheme", "seed", "encoder", "decoder"]],
        on="rel_path",
        how="left",
        suffixes=("", "_log"),
    )
    for col in ["scheme", "seed", "encoder", "decoder"]:
        log_col = f"{col}_log"
        if log_col in code.columns:
            code[col] = code[col].where(code[col].notna(), code[log_col])
            code = code.drop(columns=[log_col])
    return code


def build_pair_summary(pair_df):
    if pair_df.empty:
        return pd.DataFrame()
    rows = []
    for scheme, g in pair_df.groupby("scheme", dropna=False):
        rows.append({
            "scheme": scheme,
            "n_pairs": len(g),
            "cos_mean": g["cos_mean"].mean(),
            "cos_std": g["cos_mean"].std(),
            "cos_min": g["cos_mean"].min(),
            "cos_max": g["cos_mean"].max(),
            "mse_mean": g["mse"].mean(),
            "l2_mean": g["l2_mean"].mean(),
        })
    return pd.DataFrame(rows).sort_values("cos_mean", ascending=False)


def plot_encoder_summary(summary, fig_dir):
    top = summary.sort_values("nmse_mean").copy()
    plt.figure(figsize=(12, max(5, len(top) * 0.35)))
    sns.barplot(data=top, y="scheme", x="nmse_mean", color="#4C78A8")
    plt.xlabel("平均 best NMSE (dB, 越低越好)")
    plt.ylabel("scheme")
    plt.title("Canonical encoder 方案平均重建性能")
    savefig(fig_dir / "deep_encoder_scheme_nmse.png")


def plot_adapter_summary(adapter_summary, fig_dir):
    if adapter_summary.empty:
        return
    top = adapter_summary.sort_values("nmse").head(30)
    plt.figure(figsize=(12, max(6, len(top) * 0.35)))
    sns.barplot(data=top, y="rel_path", x="nmse", hue="adapter", dodge=False)
    plt.xlabel("adapter best NMSE (dB, 越低越好)")
    plt.ylabel("实验")
    plt.title("Adapter 实验性能排名 Top 30")
    plt.legend(loc="lower right", fontsize=8)
    savefig(fig_dir / "deep_adapter_top30.png")

    numeric = [
        "nmse", "loss_code", "loss_fc", "delta_ratio", "gate_mean",
        "lowrank_ratio", "mlp_ratio", "adapter_vs_source_nmse_delta",
    ]
    cols = [c for c in numeric if c in adapter_summary and adapter_summary[c].notna().sum() > 3]
    corr = adapter_summary[cols].corr(numeric_only=True)
    if not corr.empty:
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Adapter 指标相关性")
        savefig(fig_dir / "deep_adapter_metric_corr.png")


def plot_code_summary(code_join, pair_summary, fig_dir):
    if not code_join.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=code_join,
            x="cov_offdiag_ratio",
            y="nmse",
            hue="scheme",
            legend=False,
        )
        plt.xlabel("cov offdiag ratio")
        plt.ylabel("best NMSE (dB, 越低越好)")
        plt.title("码字去相关程度与重建性能")
        savefig(fig_dir / "deep_code_cov_vs_nmse.png")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=code_join,
            x="effective_rank",
            y="nmse",
            hue="scheme",
            legend=False,
        )
        plt.xlabel("effective rank")
        plt.ylabel("best NMSE (dB, 越低越好)")
        plt.title("码字有效秩与重建性能")
        savefig(fig_dir / "deep_code_rank_vs_nmse.png")

    if not pair_summary.empty:
        top = pair_summary.sort_values("cos_mean", ascending=False)
        plt.figure(figsize=(12, max(5, len(top) * 0.35)))
        sns.barplot(data=top, y="scheme", x="cos_mean", color="#59A14F")
        plt.xlabel("pairwise cosine mean (越高表示码字坐标越接近)")
        plt.ylabel("scheme")
        plt.title("不同实验码字对齐程度")
        savefig(fig_dir / "deep_pairwise_cosine_by_scheme.png")


def write_markdown(out_dir, enc_summary, adapter_summary, code_join, pair_summary):
    lines = []
    lines.append("# encoder_canonical 全量实验深度汇总")
    lines.append("")
    lines.append("本文件由 `analysis/encoder_canonical/deep_dive.py` 自动生成，作为最终中文分析文档的数据底稿。")
    lines.append("")

    if not enc_summary.empty:
        lines.append("## Encoder 方案排名")
        lines.append("")
        lines.append(markdown_table(enc_summary, max_rows=20))
        lines.append("")

    if not adapter_summary.empty:
        lines.append("## Adapter 性能排名")
        lines.append("")
        cols = [
            "scheme", "adapter", "seed", "encoder", "epochs", "nmse",
            "source_self_nmse", "decoder42_self_nmse",
            "adapter_vs_source_nmse_delta", "delta_ratio", "gate_mean",
        ]
        cols = [c for c in cols if c in adapter_summary]
        lines.append(markdown_table(adapter_summary, columns=cols, max_rows=30))
        lines.append("")

    if not code_join.empty:
        lines.append("## Codeword 统计摘要")
        lines.append("")
        code_scheme = (
            code_join.groupby("scheme", dropna=False)
            .agg(
                n=("rel_path", "count"),
                effective_rank_mean=("effective_rank", "mean"),
                cov_offdiag_ratio_mean=("cov_offdiag_ratio", "mean"),
                dim_var_cv_mean=("dim_var_cv", "mean"),
                global_rms_mean=("global_rms", "mean"),
                nmse_mean=("nmse", "mean"),
            )
            .reset_index()
            .sort_values("nmse_mean")
        )
        lines.append(markdown_table(code_scheme))
        lines.append("")

    if not pair_summary.empty:
        lines.append("## Pairwise 码字对齐摘要")
        lines.append("")
        lines.append(markdown_table(pair_summary))
        lines.append("")

    (out_dir / "deep_dive_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis_outputs/encoder_canonical_full")
    args = parser.parse_args()

    setup_matplotlib_fonts()
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    log_df = pd.read_csv(out_dir / "experiment_log_summary.csv")
    code_stats = read_csv_if_exists(out_dir / "codeword_stats.csv")
    pair_df = read_csv_if_exists(out_dir / "codeword_pairwise.csv")

    enc = encoder_rows(log_df)
    adapter = adapter_rows(log_df)

    enc_summary = build_encoder_scheme_summary(enc)
    adapter_summary = build_adapter_summary(adapter, enc)
    code_join = build_code_join(enc, code_stats)
    pair_summary = build_pair_summary(pair_df)

    enc_summary.to_csv(out_dir / "deep_encoder_scheme_summary.csv", index=False)
    adapter_summary.to_csv(out_dir / "deep_adapter_summary.csv", index=False)
    code_join.to_csv(out_dir / "deep_codeword_encoder_join.csv", index=False)
    pair_summary.to_csv(out_dir / "deep_pairwise_scheme_summary.csv", index=False)

    plot_encoder_summary(enc_summary, fig_dir)
    plot_adapter_summary(adapter_summary, fig_dir)
    plot_code_summary(code_join, pair_summary, fig_dir)
    write_markdown(out_dir, enc_summary, adapter_summary, code_join, pair_summary)

    print(f"Wrote deep dive outputs to {out_dir}")


if __name__ == "__main__":
    main()
