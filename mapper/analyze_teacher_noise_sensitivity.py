#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import universal_csi  # noqa: E402
from utils.statics import evaluator, nmse_from_sums  # noqa: E402


DEFAULT_TEACHER = "exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt"
DEFAULT_DECODER_CKPT = "exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth"
DEFAULT_DECODER_ARGS = "exps/COST2100/in/seed42/transnet_transnet/args.json"
DEFAULT_DATA = "/storage/hujiacong/zxd/datasets/cost2100/in_train.pt"
DEFAULT_OUT = "mapper/reports/teacher_noise_sensitivity"
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
        decoder_state,
        strict=False)
    if missing or unexpected:
        raise ValueError(
            f"decoder mismatch: missing={missing}, unexpected={unexpected}")


def decode_metrics(decoder, code, csi, batch_size, workers, device):
    loader = DataLoader(
        TensorDataset(code, csi),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    decoder.eval()
    with torch.no_grad():
        for z, x in loader:
            z = z.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            y = decoder(z)
            mse = F.mse_loss(y, x)
            error_sum, power_sum = evaluator(y, x)
            total_error += error_sum
            total_power += power_sum
            total_mse += float(mse.detach().cpu()) * z.size(0)
            total_n += z.size(0)
    nmse_db = nmse_from_sums(total_error, total_power)
    return {
        "recon_mse": total_mse / max(total_n, 1),
        "nmse_db": float(nmse_db.detach().cpu()),
        "nmse_linear": float((total_error / total_power.clamp_min(1e-12)).detach().cpu()),
    }


def make_noise(shape, dist, sigma, generator):
    if sigma == 0:
        return torch.zeros(shape, dtype=torch.float32)
    if dist == "gaussian":
        return torch.randn(shape, generator=generator, dtype=torch.float32) * sigma
    if dist == "laplace":
        # Match variance to Gaussian sigma^2: Laplace scale b has var=2b^2.
        scale = sigma / math.sqrt(2.0)
        u = torch.rand(shape, generator=generator, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
        return scale * torch.sign(u - 0.5) * (-torch.log1p(-2.0 * (u - 0.5).abs()))
    raise ValueError(dist)


def interpolate_threshold(rows, target_gap):
    rows = sorted(rows, key=lambda x: x["sigma"])
    prev = None
    for row in rows:
        if row["gap_db"] >= target_gap:
            if prev is None:
                return row
            denom = row["gap_db"] - prev["gap_db"]
            if abs(denom) < 1e-12:
                return row
            alpha = (target_gap - prev["gap_db"]) / denom
            out = {"dist": row["dist"], "target_gap_db": target_gap}
            for key in ["sigma", "code_mse", "code_rmse", "code_mae",
                        "nmse_db", "nmse_linear", "recon_mse"]:
                out[key] = prev[key] + alpha * (row[key] - prev[key])
            out["gap_db"] = target_gap
            out["lower_sigma"] = prev["sigma"]
            out["upper_sigma"] = row["sigma"]
            return out
        prev = row
    return rows[-1] if rows else None


def plot_results(rows, baseline_nmse, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for dist in ["gaussian", "laplace"]:
        part = [r for r in rows if r["dist"] == dist]
        part = sorted(part, key=lambda x: x["code_mse"])
        axes[0].plot([r["sigma"] for r in part], [r["gap_db"] for r in part],
                     marker="o", label=dist)
        axes[1].plot([r["code_mse"] for r in part], [r["gap_db"] for r in part],
                     marker="o", label=dist)
    for ax in axes:
        ax.axhline(1.0, color="black", ls="--", lw=1, label="1 dB gap")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_xlabel("noise sigma (std)")
    axes[0].set_ylabel("NMSE gap vs teacher (dB)")
    axes[0].set_title(f"teacher NMSE = {baseline_nmse:.3f} dB")
    axes[1].set_xlabel("code MSE = MSE(z_noisy, z_teacher)")
    axes[1].set_xscale("log")
    axes[1].set_ylabel("NMSE gap vs teacher (dB)")
    axes[1].set_title("code loss threshold")
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
    parser.add_argument("--sigmas", default="0,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.08,0.1")
    parser.add_argument("--target_gap", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    setup_matplotlib()
    device = resolve_device(args.gpu, args.cpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if args.max_samples and args.max_samples > 0:
        teacher = teacher[:args.max_samples]
        csi = csi[:args.max_samples]
    if teacher.size(0) != csi.size(0):
        raise ValueError(f"N mismatch: code={teacher.size(0)} data={csi.size(0)}")

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

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    baseline = decode_metrics(decoder, teacher, csi, args.batch_size,
                              args.workers, device)
    rows = []
    for dist in ["gaussian", "laplace"]:
        for sigma in sigmas:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(args.seed + int(round(sigma * 1_000_000)) + (0 if dist == "gaussian" else 10_000_000))
            noise = make_noise(teacher.shape, dist, sigma, gen)
            noisy = teacher + noise
            metrics = decode_metrics(decoder, noisy, csi, args.batch_size,
                                     args.workers, device)
            diff = noisy - teacher
            row = {
                "dist": dist,
                "sigma": sigma,
                "code_mse": float(diff.pow(2).mean()),
                "code_rmse": float(diff.pow(2).mean().sqrt()),
                "code_mae": float(diff.abs().mean()),
                "noise_std_empirical": float(diff.std()),
                "recon_mse": metrics["recon_mse"],
                "nmse_linear": metrics["nmse_linear"],
                "nmse_db": metrics["nmse_db"],
                "gap_db": metrics["nmse_db"] - baseline["nmse_db"],
            }
            rows.append(row)
            print(
                f"{dist:8s} sigma={sigma:.5f} "
                f"code_mse={row['code_mse']:.6e} "
                f"nmse={row['nmse_db']:.3f}dB "
                f"gap={row['gap_db']:.3f}dB")

    thresholds = {}
    for dist in ["gaussian", "laplace"]:
        thresholds[dist] = interpolate_threshold(
            [r for r in rows if r["dist"] == dist],
            args.target_gap)

    csv_path = out_dir / "noise_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "baseline": baseline,
        "target_gap_db": args.target_gap,
        "thresholds": thresholds,
        "rows": rows,
    }
    json_path = out_dir / "noise_sweep.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    plot_results(rows, baseline["nmse_db"], out_dir / "noise_sweep.png")
    print(json.dumps({
        "baseline": baseline,
        "thresholds": thresholds,
        "csv": str(csv_path),
        "json": str(json_path),
        "figure": str(out_dir / "noise_sweep.png"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
