#!/usr/bin/env python
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import ensure_dir, setup_matplotlib_fonts


TARGET_SCHEME = "aux_pca_1e-2_code_mean1e-4_cov1e-4"


def savefig(path):
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def load_frames(out_dir):
    out_dir = Path(out_dir)
    log = pd.read_csv(out_dir / "experiment_log_summary.csv")
    code = pd.read_csv(out_dir / "codeword_stats.csv")
    pair = pd.read_csv(out_dir / "codeword_pairwise.csv")
    adapter = pd.read_csv(out_dir / "deep_adapter_summary.csv")
    enc = log[log["family"] == "encoder"].copy()
    enc["nmse"] = enc["best_nmse"].where(enc["best_nmse"].notna(), enc["final_test_nmse"])
    ad = log[log["family"].astype(str).str.startswith("adapter/")].copy()
    ad["nmse"] = ad["best_nmse"].where(ad["best_nmse"].notna(), ad["final_test_nmse"])
    return enc, ad, code, pair, adapter


def plot_encoder_scheme(enc, fig_dir):
    summary = (
        enc.groupby("scheme")
        .agg(n=("rel_path", "count"), mean_nmse=("nmse", "mean"), std_nmse=("nmse", "std"))
        .reset_index()
        .sort_values("mean_nmse")
    )
    plt.figure(figsize=(11, 8))
    sns.barplot(data=summary, x="mean_nmse", y="scheme", color="#4C78A8")
    plt.xlabel("平均 best NMSE (dB, 越低越好)")
    plt.ylabel("canonical 方案")
    plt.title("Encoder 训练：不同 canonical 方案重建性能")
    savefig(fig_dir / "final_encoder_scheme_mean_nmse.png")


def plot_encoder_alignment(pair, fig_dir):
    pe = pair[pair["family"] == "encoder"].copy()
    summary = (
        pe.groupby("scheme")
        .agg(n_pairs=("rel_a", "count"), cos_mean=("cos_mean", "mean"), cos_std=("cos_mean", "std"))
        .reset_index()
        .sort_values("cos_mean", ascending=False)
    )
    plt.figure(figsize=(11, 8))
    sns.barplot(data=summary, x="cos_mean", y="scheme", color="#59A14F")
    plt.xlabel("同一样本 code cosine 均值 (越高越对齐)")
    plt.ylabel("canonical 方案")
    plt.title("Encoder 码字坐标对齐程度")
    savefig(fig_dir / "final_encoder_pairwise_cosine.png")


def plot_code_rank_collapse(enc, code, fig_dir):
    ce = code[code["family"] == "encoder"].merge(
        enc[["rel_path", "nmse"]],
        on="rel_path",
        how="left",
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=ce,
        x="effective_rank",
        y="nmse",
        hue="scheme",
        legend=False,
        s=65,
    )
    plt.xlabel("effective rank")
    plt.ylabel("best NMSE (dB, 越低越好)")
    plt.title("码字有效秩与重建性能：code_reg 过强会压低有效秩")
    savefig(fig_dir / "final_code_effective_rank_vs_nmse.png")


def plot_target_arch_adapter(enc, adapter, fig_dir):
    target_enc = enc[enc["scheme"] == TARGET_SCHEME][["seed", "encoder", "nmse"]].copy()
    target_enc["type"] = "自编码器自重建"
    target_enc = target_enc.rename(columns={"nmse": "NMSE"})

    target_ad = adapter[
        (adapter["scheme"] == TARGET_SCHEME)
        & (adapter["epochs"] == 400)
        & (adapter["adapter"] == "gated_lowrank_affine_mlp")
    ][["seed", "encoder", "nmse"]].copy()
    target_ad["type"] = "接入 seed42 decoder 的 adapter"
    target_ad = target_ad.rename(columns={"nmse": "NMSE"})

    data = pd.concat([target_enc, target_ad], ignore_index=True)
    data["seed"] = data["seed"].astype(str)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="encoder", y="NMSE", hue="type", errorbar=None)
    plt.xlabel("encoder 架构")
    plt.ylabel("NMSE (dB, 越低越好)")
    plt.title(f"{TARGET_SCHEME}: 自重建与 adapter 测试对比")
    savefig(fig_dir / "final_target_scheme_self_vs_adapter_by_arch.png")

    gap = adapter[
        (adapter["scheme"] == TARGET_SCHEME)
        & (adapter["epochs"] == 400)
        & (adapter["adapter"] == "gated_lowrank_affine_mlp")
    ].copy()
    gap["seed"] = gap["seed"].astype(str)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=gap, x="encoder", y="adapter_vs_source_nmse_delta", hue="seed")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("encoder 架构")
    plt.ylabel("adapter NMSE - source 自重建 NMSE (dB, 越小越好)")
    plt.title("Adapter 相对原 encoder/decoder 自重建的性能损失")
    savefig(fig_dir / "final_target_scheme_adapter_gap.png")


def plot_adapter_loss_corr(adapter, fig_dir):
    data = adapter.copy()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="loss_code",
        y="nmse",
        hue="adapter",
        style="scheme",
        s=70,
    )
    plt.xscale("log")
    plt.xlabel("最后一轮 code loss (log)")
    plt.ylabel("adapter best NMSE (dB, 越低越好)")
    plt.title("Adapter 性能与 code loss 的关系")
    plt.legend(fontsize=7, loc="best")
    savefig(fig_dir / "final_adapter_code_loss_vs_nmse.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis_outputs/encoder_canonical_full")
    args = parser.parse_args()

    setup_matplotlib_fonts()
    out_dir = Path(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    enc, ad, code, pair, adapter = load_frames(out_dir)

    plot_encoder_scheme(enc, fig_dir)
    plot_encoder_alignment(pair, fig_dir)
    plot_code_rank_collapse(enc, code, fig_dir)
    plot_target_arch_adapter(enc, adapter, fig_dir)
    plot_adapter_loss_corr(adapter, fig_dir)
    print(f"Wrote final report figures to {fig_dir}")


if __name__ == "__main__":
    main()
