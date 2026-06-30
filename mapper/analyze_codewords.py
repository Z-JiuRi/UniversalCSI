#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


DEFAULT_TARGET = "exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt"
DEFAULT_SOURCES = [
    "exps/COST2100/in/seed2026/transnet_transnet/codewords/train_code.pt",
    "exps/COST2100/in/seed3407/transnet_transnet/codewords/train_code.pt",
    "exps/COST2100/in/seed2026/clnet_transnet/codewords/train_code.pt",
    "exps/COST2100/in/seed2026/crnet_transnet/codewords/train_code.pt",
    "exps/COST2100/in/seed2026/csinet_transnet/codewords/train_code.pt",
]
SONGTI_PATH = "/home/hujiacong/zxd/.envs/SongTi.ttf"


def setup_fonts():
    if Path(SONGTI_PATH).exists():
        fm.fontManager.addfont(SONGTI_PATH)
        font = fm.FontProperties(fname=SONGTI_PATH)
        plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_code(path, sample_size=0):
    x = torch.load(path, weights_only=True, map_location="cpu").float()
    if x.ndim != 2:
        raise ValueError(f"{path} should be 2D, got {tuple(x.shape)}")
    if sample_size and x.size(0) > sample_size:
        x = x[:sample_size].contiguous()
    return x


def short_name(path):
    parts = Path(path).parts
    if "seed2026" in parts or "seed3407" in parts or "seed42" in parts:
        for i, part in enumerate(parts):
            if part.startswith("seed") and i + 1 < len(parts):
                return f"{part}_{parts[i + 1]}"
    return Path(path).parent.parent.name


def cov_stats(x):
    xc = x - x.mean(0, keepdim=True)
    cov = xc.t().matmul(xc) / max(x.size(0) - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    eff_rank = (eig.sum().pow(2) / eig.pow(2).sum().clamp_min(1e-12)).item()
    off = cov - torch.diag_embed(cov.diag())
    off_ratio = (off.pow(2).sum().sqrt()
                 / cov.diag().pow(2).sum().sqrt().clamp_min(1e-12)).item()
    return {
        "effective_rank": eff_rank,
        "cov_offdiag_ratio": off_ratio,
        "eig_top1": eig[-1].item(),
        "eig_top10_sum_ratio": (
            eig[-10:].sum() / eig.sum().clamp_min(1e-12)).item(),
    }


def code_stats(name, x):
    dim_var = x.var(0, unbiased=False)
    out = {
        "name": name,
        "n": x.size(0),
        "dim": x.size(1),
        "global_mean": x.mean().item(),
        "global_std": x.std(unbiased=False).item(),
        "global_rms": x.pow(2).mean().sqrt().item(),
        "norm_mean": x.norm(dim=1).mean().item(),
        "norm_std": x.norm(dim=1).std(unbiased=False).item(),
        "dim_mean_abs": x.mean(0).abs().mean().item(),
        "dim_var_mean": dim_var.mean().item(),
        "dim_var_cv": (
            dim_var.std(unbiased=False) / dim_var.mean().clamp_min(1e-12)
        ).item(),
    }
    out.update(cov_stats(x))
    return out


def pair_stats(name, source, target):
    n = min(source.size(0), target.size(0))
    source = source[:n]
    target = target[:n]
    diff = source - target
    cos = F.cosine_similarity(source, target, dim=1)
    diff_cos = F.cosine_similarity(diff, target, dim=1)
    out = {
        "name": name,
        "n": n,
        "mse": diff.pow(2).mean().item(),
        "mae": diff.abs().mean().item(),
        "l2_mean": diff.norm(dim=1).mean().item(),
        "l2_std": diff.norm(dim=1).std(unbiased=False).item(),
        "cos_mean": cos.mean().item(),
        "cos_std": cos.std(unbiased=False).item(),
        "cos_p05": torch.quantile(cos, 0.05).item(),
        "cos_p50": torch.quantile(cos, 0.50).item(),
        "cos_p95": torch.quantile(cos, 0.95).item(),
        "diff_target_cos_mean": diff_cos.mean().item(),
        "diff_rms": diff.pow(2).mean().sqrt().item(),
        "scale_ratio_norm": (
            source.norm(dim=1).mean()
            / target.norm(dim=1).mean().clamp_min(1e-12)
        ).item(),
    }
    out.update({f"diff_{k}": v for k, v in cov_stats(diff).items()})
    return out


def plot_bar(df, x, y, title, path):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y, color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def write_report(out_dir, code_df, pair_df, target_name):
    def md_table(df):
        if df.empty:
            return ""
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Mapper Codeword 分析报告",
        "",
        f"目标 teacher code：`{target_name}`。",
        "",
        "## 结论摘要",
        "",
    ]
    if not pair_df.empty:
        best = pair_df.sort_values("mse").iloc[0]
        worst = pair_df.sort_values("mse").iloc[-1]
        lines += [
            f"- 当前最接近 teacher code 的 source 是 `{best['name']}`，"
            f"MSE={best['mse']:.4e}，cos={best['cos_mean']:.4f}。",
            f"- 当前最远的 source 是 `{worst['name']}`，"
            f"MSE={worst['mse']:.4e}，cos={worst['cos_mean']:.4f}。",
            "- 如果 raw source 与 teacher 的 cosine 接近 0，mapper 要学习的就是完整高维 transport，"
            "不是小幅校准。",
            "- 如果 diff effective rank 很高，低秩 affine 不够；如果 diff 有明显主方向，flow/MLP 可以先重点拟合主结构。",
            "",
        ]
    lines += [
        "## 单 codeword 分布",
        "",
        md_table(code_df),
        "",
        "## Source 到 teacher 的差异",
        "",
        md_table(pair_df),
        "",
        "## 方法建议",
        "",
        "- 首先用纯 MSE 训练 mapper，确认 `z_src -> z_teacher` 能否被学到 `1e-4` 级别。",
        "- 若 raw cosine 接近 0 且 diff rank 高，优先尝试 `flow` 或 `hybrid_flow_mlp`。",
        "- `lambda_cos` 可作为辅助，但不应替代 MSE；decoder 最终更关心绝对 code 误差。",
        "- `lambda_cov` 只建议弱开，用于让残差结构更规整，默认关闭。",
    ]
    (out_dir / "codeword_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    setup_fonts()
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_code", default=DEFAULT_TARGET)
    parser.add_argument("--source_code", action="append", default=[])
    parser.add_argument("--out_dir", default="mapper/reports/codeword_analysis")
    parser.add_argument("--sample_size", type=int, default=20000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    sources = args.source_code or [
        p for p in DEFAULT_SOURCES if Path(p).exists()
    ]
    target = load_code(args.target_code, args.sample_size)
    target_name = short_name(args.target_code)
    code_rows = [code_stats(target_name, target)]
    pair_rows = []
    for src_path in sources:
        if not Path(src_path).exists():
            print(f"skip missing source: {src_path}")
            continue
        name = short_name(src_path)
        source = load_code(src_path, args.sample_size)
        code_rows.append(code_stats(name, source))
        pair_rows.append(pair_stats(name, source, target))
    code_df = pd.DataFrame(code_rows)
    pair_df = pd.DataFrame(pair_rows)
    code_df.to_csv(out_dir / "code_stats.csv", index=False)
    pair_df.to_csv(out_dir / "pair_stats.csv", index=False)
    (out_dir / "inputs.json").write_text(
        json.dumps({
            "target_code": args.target_code,
            "source_code": sources,
            "sample_size": args.sample_size,
        }, indent=2),
        encoding="utf-8")
    if not pair_df.empty:
        plot_bar(pair_df, "name", "mse", "Source 到 teacher 的 code MSE",
                 fig_dir / "pair_mse.png")
        plot_bar(pair_df, "name", "cos_mean", "Source 到 teacher 的 cosine",
                 fig_dir / "pair_cosine.png")
        plot_bar(pair_df, "name", "diff_effective_rank",
                 "Source-teacher 残差 effective rank",
                 fig_dir / "diff_effective_rank.png")
    write_report(out_dir, code_df, pair_df, target_name)
    print(out_dir / "codeword_analysis.md")


if __name__ == "__main__":
    main()
