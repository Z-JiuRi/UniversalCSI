#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True).float()


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


def fc_decoder_stats(checkpoint_path, device):
    state_dict = load_state_dict(checkpoint_path)
    key = "decoder.fc_decoder.weight"
    if key not in state_dict:
        return {}
    weight = state_dict[key].float().to(device)
    singular_values = torch.linalg.svdvals(weight)
    col_norm = weight.norm(dim=0)
    row_norm = weight.norm(dim=1)
    return {
        "shape": list(weight.shape),
        "sv_max": float(singular_values.max().detach().cpu()),
        "sv_p95": float(torch.quantile(singular_values, 0.95).detach().cpu()),
        "sv_median": float(singular_values.median().detach().cpu()),
        "sv_mean": float(singular_values.mean().detach().cpu()),
        "sv_min": float(singular_values.min().detach().cpu()),
        "col_norm_mean": float(col_norm.mean().detach().cpu()),
        "col_norm_max": float(col_norm.max().detach().cpu()),
        "col_norm_min": float(col_norm.min().detach().cpu()),
        "row_norm_mean": float(row_norm.mean().detach().cpu()),
        "row_norm_max": float(row_norm.max().detach().cpu()),
        "row_norm_min": float(row_norm.min().detach().cpu()),
    }


def load_nmse_results(path):
    rows = []
    for item in path.glob("*.json"):
        try:
            data = json.loads(item.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("nmse_db") is None:
            continue
        rows.append({
            "name": item.stem,
            "nmse_db": float(data["nmse_db"]),
            "code_path": data.get("code_path", ""),
        })
    return rows


def save_fixed_decoder_bar(rows, out_dir):
    teacher = next(row for row in rows if row["name"] == "teacher_code")
    order = [
        "teacher_code",
        "hybrid_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400",
        "hybrid_smooth_tail_white_seed2026_transnet_best_mapper",
        "hybrid_smoothl1_0.5_seed2026_clnet_transnet",
        "mlp_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400",
        "flow_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400",
    ]
    labels = {
        "teacher_code": "teacher code",
        "hybrid_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": "hybrid old\ntransnet2026",
        "hybrid_smooth_tail_white_seed2026_transnet_best_mapper": "hybrid smooth+tail+white\ntransnet2026",
        "hybrid_smoothl1_0.5_seed2026_clnet_transnet": "hybrid smoothl1\nclnet2026",
        "mlp_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": "MLP old\ntransnet2026",
        "flow_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": "flow old\ntransnet2026",
    }
    selected = [next(row for row in rows if row["name"] == name) for name in order]
    xs = np.arange(len(selected))
    vals = [row["nmse_db"] for row in selected]
    colors = ["#4C78A8" if row["name"] == "teacher_code" else "#F58518" for row in selected]

    plt.figure(figsize=(9.2, 4.8))
    plt.bar(xs, vals, color=colors)
    plt.axhline(
        teacher["nmse_db"] + 1.0,
        color="#54A24B",
        linestyle="--",
        linewidth=1.5,
        label="1 dB gap target",
    )
    plt.axhline(
        teacher["nmse_db"],
        color="#4C78A8",
        linestyle=":",
        linewidth=1.2,
        label="teacher",
    )
    plt.xticks(xs, [labels[row["name"]] for row in selected], fontsize=9)
    plt.ylabel("Fixed decoder NMSE (dB)")
    plt.title("Fixed seed42 decoder performance")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "fixed_decoder_nmse_bar.png", dpi=180)
    plt.close()


def save_mse_nmse_scatter(rows, out_dir):
    teacher = next(row for row in rows if row["name"] == "teacher_code")
    mse_by_name = {
        "hybrid_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.003117,
        "mlp_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00392,
        "mlp_seed3407_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00445,
        "mlp_seed2026_clnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00473,
        "mlp_seed2026_crnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00475,
        "hybrid_seed2026_clnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00461,
        "hybrid_seed2026_crnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00478,
        "mlp_seed2026_csinet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00645,
        "hybrid_seed2026_csinet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00674,
        "flow_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400": 0.00686,
        "hybrid_smooth_tail_white_seed2026_transnet_best_mapper": 0.0021355,
        "hybrid_smoothl1_0.5_seed2026_clnet_transnet": 0.0030465,
    }

    plt.figure(figsize=(6.5, 4.8))
    for row in rows:
        mse = mse_by_name.get(row["name"])
        if mse is None:
            continue
        if "smooth_tail_white" in row["name"]:
            color, size, marker = "#E45756", 75, "*"
        elif row["name"].startswith("flow"):
            color, size, marker = "#72B7B2", 45, "s"
        elif row["name"].startswith("mlp"):
            color, size, marker = "#54A24B", 45, "o"
        else:
            color, size, marker = "#F58518", 45, "o"
        plt.scatter(mse, row["nmse_db"], c=color, s=size, marker=marker)
    plt.axhline(
        teacher["nmse_db"] + 1.0,
        color="#4C78A8",
        linestyle="--",
        linewidth=1.3,
        label="1 dB gap target",
    )
    plt.axvline(
        7.54e-4,
        color="#B279A2",
        linestyle="--",
        linewidth=1.3,
        label="noise 1 dB MSE",
    )
    plt.xscale("log")
    plt.xlabel("Code MSE to teacher (log)")
    plt.ylabel("Fixed decoder NMSE (dB)")
    plt.title("Code MSE is necessary but not sufficient")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "code_mse_vs_fixed_decoder_nmse.png", dpi=180)
    plt.close()


def residual_stats(diff):
    per_sample = diff.pow(2).mean(dim=1).detach().cpu().numpy()
    per_dim = diff.pow(2).mean(dim=0).sqrt().detach().cpu().numpy()
    return {
        "code_mse": float(per_sample.mean()),
        "code_rmse": float(np.sqrt(per_sample.mean())),
        "sample_p50": float(np.quantile(per_sample, 0.50)),
        "sample_p95": float(np.quantile(per_sample, 0.95)),
        "sample_p99": float(np.quantile(per_sample, 0.99)),
        "dim_rmse_mean": float(per_dim.mean()),
        "dim_rmse_max": float(per_dim.max()),
    }


def save_residual_distribution(old_diff, new_diff, out_dir, max_values=1_200_000):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)

    def sample_values(diff):
        flat = diff.detach().cpu().reshape(-1)
        if flat.numel() > max_values:
            indices = torch.randperm(flat.numel(), generator=generator)[:max_values]
            flat = flat[indices]
        return flat.numpy()

    old_values = sample_values(old_diff)
    new_values = sample_values(new_diff)
    lim = np.quantile(np.abs(np.concatenate([old_values, new_values])), 0.995)
    bins = np.linspace(-lim, lim, 160)

    plt.figure(figsize=(7.2, 4.8))
    plt.hist(old_values, bins=bins, density=True, alpha=0.45,
             label="old hybrid residual", color="#F58518")
    plt.hist(new_values, bins=bins, density=True, alpha=0.45,
             label="smooth+tail+white residual", color="#E45756")
    plt.yscale("log")
    plt.xlabel("mapped code - teacher code")
    plt.ylabel("density (log)")
    plt.title("Residual distribution remains heavy-tailed")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution_log.png", dpi=180)
    plt.close()


def save_sample_cdf(old_diff, new_diff, out_dir):
    old_sample = old_diff.pow(2).mean(dim=1).detach().cpu().numpy()
    new_sample = new_diff.pow(2).mean(dim=1).detach().cpu().numpy()
    plt.figure(figsize=(6.8, 4.8))
    for values, label, color in [
        (old_sample, "old hybrid", "#F58518"),
        (new_sample, "smooth+tail+white", "#E45756"),
    ]:
        xs = np.sort(values)
        ys = np.linspace(0, 1, len(xs), endpoint=False)
        plt.plot(
            xs,
            ys,
            label=f"{label} mean={values.mean():.4g} p95={np.quantile(values, 0.95):.4g}",
            color=color,
        )
    plt.axvline(7.54e-4, color="#4C78A8", linestyle="--",
                linewidth=1.2, label="noise 1 dB MSE")
    plt.xscale("log")
    plt.xlabel("Per-sample code MSE")
    plt.ylabel("CDF")
    plt.title("Per-sample residual tail is still above target")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "sample_mse_cdf.png", dpi=180)
    plt.close()


def save_residual_pca(old_diff, new_diff, out_dir, subset):
    def spectrum(diff):
        if diff.size(0) > subset:
            indices = torch.linspace(
                0, diff.size(0) - 1, steps=subset,
                device=diff.device).long()
            diff = diff.index_select(0, indices)
        x = diff.to(torch.float64)
        x = x - x.mean(dim=0, keepdim=True)
        cov = x.t().matmul(x) / max(x.size(0) - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
        cum = (torch.cumsum(eigvals, dim=0) / eigvals.sum()).detach().cpu().numpy()
        return cum

    old_cum = spectrum(old_diff)
    new_cum = spectrum(new_diff)
    xs = np.arange(1, len(old_cum) + 1)
    plt.figure(figsize=(6.8, 4.8))
    plt.plot(xs, old_cum, label=f"old hybrid: top50={old_cum[49]:.3f}",
             color="#F58518")
    plt.plot(xs, new_cum, label=f"smooth+tail+white: top50={new_cum[49]:.3f}",
             color="#E45756")
    plt.xlabel("PCA components")
    plt.ylabel("Cumulative residual energy")
    plt.title("Residual error is not concentrated in only a few directions")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_pca_cumulative.png", dpi=180)
    plt.close()
    return {
        "old_top10": float(old_cum[9]),
        "old_top50": float(old_cum[49]),
        "old_top100": float(old_cum[99]),
        "new_top10": float(new_cum[9]),
        "new_top50": float(new_cum[49]),
        "new_top100": float(new_cum[99]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--pca_subset", type=int, default=20000)
    parser.add_argument("--out_dir", type=Path,
                        default=ROOT / "mapper/reports/generative_code_mapping_feasibility")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    out_dir = args.out_dir
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = load_nmse_results(ROOT / "mapper/reports/decoder_nmse")
    save_fixed_decoder_bar(rows, figure_dir)
    save_mse_nmse_scatter(rows, figure_dir)

    teacher = load_tensor(
        ROOT / "exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt"
    ).to(device)
    old_code = load_tensor(
        ROOT / "mapper/exps/hybrid/seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400/mapped_code.pt"
    ).to(device)
    new_code = load_tensor(
        ROOT / "mapper/exps/hybrid/smooth_tail_white/seed2026_transnet_transnet_to_seed42_transnet_lr5e-4_ep400/codewords/mapped_code.pt"
    ).to(device)

    old_diff = old_code - teacher
    new_diff = new_code - teacher
    old_stats = residual_stats(old_diff)
    new_stats = residual_stats(new_diff)

    save_residual_distribution(old_diff, new_diff, figure_dir)
    save_sample_cdf(old_diff, new_diff, figure_dir)
    pca_stats = save_residual_pca(old_diff, new_diff, figure_dir, args.pca_subset)
    fc_stats = fc_decoder_stats(
        ROOT / "exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth",
        device,
    )

    teacher_nmse = next(row["nmse_db"] for row in rows if row["name"] == "teacher_code")
    old_nmse = next(
        row["nmse_db"] for row in rows
        if row["name"] == "hybrid_seed2026_transnet_transnet_to_seed42_transnet_code_mse0.0_cov0.0_lr5e-4_ep400"
    )
    new_nmse = next(
        row["nmse_db"] for row in rows
        if row["name"] == "hybrid_smooth_tail_white_seed2026_transnet_best_mapper"
    )
    summary = {
        "device": str(device),
        "teacher_nmse_db": teacher_nmse,
        "old_hybrid_nmse_db": old_nmse,
        "smooth_tail_white_nmse_db": new_nmse,
        "old_hybrid_gap_db": old_nmse - teacher_nmse,
        "smooth_tail_white_gap_db": new_nmse - teacher_nmse,
        "noise_1db_code_mse": 7.54e-4,
        "noise_1db_code_rmse": float(np.sqrt(7.54e-4)),
        "old_hybrid": old_stats,
        "smooth_tail_white": new_stats,
        "pca_subset": min(args.pca_subset, int(teacher.size(0))),
        "residual_pca": pca_stats,
        "fixed_decoder_fc_decoder": fc_stats,
        "figures": [
            str(path.relative_to(out_dir))
            for path in sorted(figure_dir.glob("*.png"))
        ],
    }
    (out_dir / "local_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
