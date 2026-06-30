#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import universal_csi  # noqa: E402


DEFAULT_TEACHER = "exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt"
DEFAULT_DECODER_CKPT = "exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth"
DEFAULT_DECODER_ARGS = "exps/COST2100/in/seed42/transnet_transnet/args.json"
DEFAULT_DATA = "/storage/hujiacong/zxd/datasets/cost2100/in_train.pt"
DEFAULT_OUT = "mapper/reports/mapper_advanced_analysis"
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


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tensor(path):
    return torch.load(path, weights_only=True, map_location="cpu").to(torch.float32)


def load_csi(path, channel, nt, nc):
    data = load_tensor(path)
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(f"{path} shape mismatch: {tuple(data.shape)}")
    return data


def clean_state_dict(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    for key in list(state_dict.keys()):
        if key.endswith("total_ops") or key.endswith("total_params"):
            del state_dict[key]
    return state_dict


def load_decoder(model, checkpoint_path):
    state_dict = clean_state_dict(checkpoint_path)
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        decoder_state = state_dict
    missing, unexpected = model.decoder.load_state_dict(
        decoder_state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"decoder mismatch: missing={missing}, unexpected={unexpected}")


def source_name(path):
    match = re.search(
        r"exps/COST2100/in/seed(\d+)/([^/]+)/codewords/train_code.pt",
        str(path))
    if match:
        return f"seed{match.group(1)}_{match.group(2)}"
    return Path(path).parent.parent.name


def read_exp_rows(skip_bad=True):
    rows = []
    for args_path in sorted(Path("mapper/exps").glob("*/*/args.json")):
        exp_dir = args_path.parent
        mapped_path = exp_dir / "mapped_code.pt"
        metrics_path = exp_dir / "metrics.json"
        if not mapped_path.exists() or not metrics_path.exists():
            continue
        if skip_bad and exp_dir.parent.name == "hybrid" and "seed3407" in exp_dir.name:
            continue
        args = json.loads(args_path.read_text())
        rows.append({
            "mapper": exp_dir.parent.name,
            "exp": exp_dir.name,
            "source": source_name(args["source_code"]),
            "source_path": args["source_code"],
            "mapped_path": str(mapped_path),
        })
    return rows


def quantiles(x):
    x = x.detach().cpu().float()
    return {
        "mean": float(x.mean()),
        "p50": float(x.quantile(0.50)),
        "p90": float(x.quantile(0.90)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
    }


def build_teacher_pca(teacher):
    x = teacher.to(torch.float64)
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    cov = centered.t().matmul(centered) / max(centered.size(0) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(1e-12).to(torch.float32)
    eigvecs = eigvecs[:, order].to(torch.float32)
    return mean.to(torch.float32), eigvals, eigvecs


def pca_residual_stats(diff, eigvals, eigvecs):
    proj = diff.matmul(eigvecs)
    energy = proj.pow(2).mean(dim=0)
    total = energy.sum().clamp_min(1e-12)
    mean_eig = eigvals.mean()
    inv_weight = 1.0 / (eigvals + 1e-4 * mean_eig)
    mahal = (proj.pow(2) * inv_weight).sum(dim=1)
    low_256 = energy[-256:].sum() / total
    return {
        "pca_top1_energy": float(energy[:1].sum() / total),
        "pca_top10_energy": float(energy[:10].sum() / total),
        "pca_top50_energy": float(energy[:50].sum() / total),
        "pca_top128_energy": float(energy[:128].sum() / total),
        "pca_low256_energy": float(low_256),
        "mahal_mean": float(mahal.mean()),
        "mahal_p95": float(mahal.quantile(0.95)),
        "mahal_p99": float(mahal.quantile(0.99)),
    }


def fc_sensitivity_stats(diff, decoder):
    if not hasattr(decoder, "fc_decoder"):
        return {}
    weight = decoder.fc_decoder.weight.detach().cpu().float()
    sensitivity = weight.pow(2).sum(dim=0)
    sensitivity = sensitivity / sensitivity.mean().clamp_min(1e-12)
    dim_mse = diff.pow(2).mean(dim=0)
    weighted = float((dim_mse * sensitivity).mean())
    topk = min(50, diff.size(1))
    top_idx = torch.topk(sensitivity, topk).indices
    low_idx = torch.topk(-sensitivity, topk).indices
    return {
        "fc_weighted_mse": weighted,
        "fc_top50_dim_mse": float(dim_mse[top_idx].mean()),
        "fc_low50_dim_mse": float(dim_mse[low_idx].mean()),
        "fc_sens_dim_mse_corr": float(torch.corrcoef(
            torch.stack([sensitivity, dim_mse]))[0, 1]),
    }


def nearest_neighbor_stats(mapped, teacher, rng, mapped_n, teacher_n,
                           batch_size, device):
    mapped_idx = torch.from_numpy(
        rng.choice(mapped.size(0), size=min(mapped_n, mapped.size(0)),
                   replace=False))
    teacher_idx = torch.from_numpy(
        rng.choice(teacher.size(0), size=min(teacher_n, teacher.size(0)),
                   replace=False))
    q = mapped[mapped_idx].to(device)
    ref = teacher[teacher_idx].to(device)
    min_dist = []
    with torch.no_grad():
        for start in range(0, q.size(0), batch_size):
            dist = torch.cdist(q[start:start + batch_size], ref)
            min_dist.append(dist.min(dim=1).values.cpu())
    min_dist = torch.cat(min_dist)
    return {
        "nn_l2_mean": float(min_dist.mean()),
        "nn_l2_p50": float(min_dist.quantile(0.50)),
        "nn_l2_p95": float(min_dist.quantile(0.95)),
        "nn_l2_p99": float(min_dist.quantile(0.99)),
    }


def decode_error_stats(decoder, code, teacher_code, csi, batch_size,
                       workers, device):
    loader = DataLoader(
        TensorDataset(code, teacher_code, csi),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    sample_nmse = []
    sample_teacher_nmse = []
    sample_recT_mse = []
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_recT = 0.0
    total_n = 0
    decoder.eval()
    with torch.no_grad():
        for z, zt, x in loader:
            z = z.to(device, non_blocking=True)
            zt = zt.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            y = decoder(z)
            yt = decoder(zt)
            err = (y - x).pow(2).flatten(1).sum(dim=1)
            err_t = (yt - x).pow(2).flatten(1).sum(dim=1)
            power = x.pow(2).flatten(1).sum(dim=1).clamp_min(1e-12)
            recT = (y - yt).pow(2).flatten(1).mean(dim=1)
            sample_nmse.append(10.0 * torch.log10(err / power).cpu())
            sample_teacher_nmse.append(10.0 * torch.log10(err_t / power).cpu())
            sample_recT_mse.append(recT.cpu())
            total_error += err.sum()
            total_power += power.sum()
            total_recT += float(recT.sum().detach().cpu())
            total_n += z.size(0)
    sample_nmse = torch.cat(sample_nmse)
    sample_teacher_nmse = torch.cat(sample_teacher_nmse)
    sample_recT_mse = torch.cat(sample_recT_mse)
    out = {
        "decoder_nmse_global": float(10.0 * torch.log10(
            total_error / total_power.clamp_min(1e-12)).detach().cpu()),
        "decoder_recT_mse": total_recT / max(total_n, 1),
    }
    for prefix, values in [
        ("sample_nmse", sample_nmse),
        ("sample_teacher_nmse", sample_teacher_nmse),
        ("sample_recT_mse", sample_recT_mse),
    ]:
        for key, value in quantiles(values).items():
            out[f"{prefix}_{key}"] = value
    return out, sample_nmse, sample_teacher_nmse, sample_recT_mse


def plot_decoder_error(out_path, sample_nmse, teacher_nmse, title):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    low = min(float(sample_nmse.quantile(0.005)),
              float(teacher_nmse.quantile(0.005)))
    high = max(float(sample_nmse.quantile(0.995)),
               float(teacher_nmse.quantile(0.995)))
    bins = np.linspace(low, high, 160)
    ax.hist(teacher_nmse.numpy(), bins=bins, density=True, alpha=0.35,
            label="teacher code")
    ax.hist(sample_nmse.numpy(), bins=bins, density=True, alpha=0.35,
            label="mapped code")
    ax.set_title(title)
    ax.set_xlabel("per-sample NMSE (dB)")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pca_energy(out_path, rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    xs = np.arange(len(rows))
    labels = [f"{r['mapper']}/{r['source'].replace('seed2026_', '').replace('_transnet', '')}"
              for r in rows]
    axes[0].bar(xs - 0.18, [r["raw_pca_top50_energy"] for r in rows],
                width=0.36, label="raw residual")
    axes[0].bar(xs + 0.18, [r["mapped_pca_top50_energy"] for r in rows],
                width=0.36, label="mapped residual")
    axes[0].set_ylabel("top50 PCA residual energy")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(xs - 0.18, [r["raw_pca_low256_energy"] for r in rows],
                width=0.36, label="raw residual")
    axes[1].bar(xs + 0.18, [r["mapped_pca_low256_energy"] for r in rows],
                width=0.36, label="mapped residual")
    axes[1].set_ylabel("last256 PCA residual energy")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tail_qq(out_path, residual, title, max_points, rng):
    flat = residual.reshape(-1)
    n = flat.numel()
    if n > max_points:
        idx = torch.from_numpy(rng.choice(n, size=max_points, replace=False))
        flat = flat[idx]
    values = flat.sort().values.numpy()
    probs = (np.arange(1, len(values) + 1) - 0.5) / len(values)
    loc = np.median(values)
    scale = np.mean(np.abs(values - loc))
    normal_q = values.mean() + values.std() * torch.distributions.Normal(
        0, 1).icdf(torch.from_numpy(probs).float()).numpy()
    laplace_q = loc - scale * np.sign(probs - 0.5) * np.log(
        1.0 - 2.0 * np.abs(probs - 0.5))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    axes[0].scatter(normal_q, values, s=2, alpha=0.25)
    axes[0].plot([values.min(), values.max()], [values.min(), values.max()],
                 color="black", lw=1)
    axes[0].set_title("Normal QQ")
    axes[1].scatter(laplace_q, values, s=2, alpha=0.25)
    axes[1].plot([values.min(), values.max()], [values.min(), values.max()],
                 color="black", lw=1)
    axes[1].set_title("Laplace QQ")
    for ax in axes:
        ax.set_xlabel("theoretical quantile")
        ax.set_ylabel("empirical quantile")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_code", default=DEFAULT_TEACHER)
    parser.add_argument("--decoder_checkpoint", default=DEFAULT_DECODER_CKPT)
    parser.add_argument("--decoder_args_json", default=DEFAULT_DECODER_ARGS)
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nn_mapped_n", type=int, default=5000)
    parser.add_argument("--nn_teacher_n", type=int, default=20000)
    parser.add_argument("--qq_points", type=int, default=300000)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    setup_matplotlib()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.gpu, args.cpu)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.decoder_args_json).read_text())
    channel = cfg.get("channel", 2)
    nt = cfg.get("nt", 32)
    nc = cfg.get("nc", 32)
    cr = cfg.get("cr", 4)
    decoder_name = cfg.get("decoder", "transnet")
    d_model = cfg.get("d_model", 64)
    dim_feedforward = cfg.get("dim_feedforward", 2048)
    hidden = cfg.get("hidden", 16)
    num_blocks = cfg.get("num_blocks", 2)

    teacher = load_tensor(args.teacher_code)
    csi = load_csi(args.data_path, channel, nt, nc)
    teacher_mean, eigvals, eigvecs = build_teacher_pca(teacher)

    model = universal_csi(
        encoder_name="transnet",
        decoder_name=decoder_name,
        reduction=cr,
        d_model=d_model,
        channel=channel,
        nt=nt,
        nc=nc,
        dim_feedforward=dim_feedforward,
        hidden=hidden,
        num_blocks=num_blocks)
    load_decoder(model, args.decoder_checkpoint)
    decoder = model.decoder.to(device).eval()

    rows = []
    for row in read_exp_rows():
        source = load_tensor(row["source_path"])
        mapped = load_tensor(row["mapped_path"])
        raw_diff = source - teacher
        mapped_diff = mapped - teacher
        raw_pca = pca_residual_stats(raw_diff, eigvals, eigvecs)
        mapped_pca = pca_residual_stats(mapped_diff, eigvals, eigvecs)
        fc_stats = fc_sensitivity_stats(mapped_diff, decoder)
        nn_stats = nearest_neighbor_stats(
            mapped, teacher, rng, args.nn_mapped_n, args.nn_teacher_n,
            args.batch_size, device)
        dec_stats, sample_nmse, teacher_nmse, recT = decode_error_stats(
            decoder, mapped, teacher, csi, args.batch_size, args.workers,
            device)

        prefix = f"{row['mapper']}_{row['exp']}"
        plot_decoder_error(
            fig_dir / f"{prefix}_decoder_sample_nmse.png",
            sample_nmse, teacher_nmse,
            f"{row['mapper']} / {row['source']} per-sample decoder NMSE")
        plot_tail_qq(
            fig_dir / f"{prefix}_mapped_residual_qq.png",
            mapped_diff,
            f"{row['mapper']} / {row['source']} mapped residual QQ",
            args.qq_points,
            rng)

        item = {
            "mapper": row["mapper"],
            "exp": row["exp"],
            "source": row["source"],
            "mapped_path": row["mapped_path"],
        }
        item.update({f"raw_{k}": v for k, v in raw_pca.items()})
        item.update({f"mapped_{k}": v for k, v in mapped_pca.items()})
        item.update(fc_stats)
        item.update(nn_stats)
        item.update(dec_stats)
        item["decoder_sample_gap_p95"] = (
            item["sample_nmse_p95"] - item["sample_teacher_nmse_p95"])
        item["decoder_sample_gap_p99"] = (
            item["sample_nmse_p99"] - item["sample_teacher_nmse_p99"])
        item["decoder_sample_tail_gt_teacher_p95"] = float(
            (sample_nmse > teacher_nmse.quantile(0.95)).float().mean())
        item["decoder_sample_nmse_png"] = str(
            fig_dir / f"{prefix}_decoder_sample_nmse.png")
        item["mapped_residual_qq_png"] = str(
            fig_dir / f"{prefix}_mapped_residual_qq.png")
        rows.append(item)

    rows_for_plot = sorted(rows, key=lambda x: x["decoder_nmse_global"])
    plot_pca_energy(fig_dir / "residual_pca_energy_summary.png", rows_for_plot)

    csv_path = out_dir / "advanced_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "advanced_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"device={device}")
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved figures: {fig_dir}")


if __name__ == "__main__":
    main()
