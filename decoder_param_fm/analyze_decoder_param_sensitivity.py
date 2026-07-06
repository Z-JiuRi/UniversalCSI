import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoder_param_fm.param_utils import (  # noqa: E402
    build_decoder_from_args,
    extract_decoder_state,
    load_codes,
    load_csi,
    load_generated_decoder_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze generated decoder tensor MSE and noise sensitivity.")
    parser.add_argument("--exp_root", type=str, default="decoder_param_fm/exps")
    parser.add_argument(
        "--decoder_args_json", type=str,
        default="exps/COST2100/in/seed42/transnet_transnet/args.json")
    parser.add_argument(
        "--target_checkpoint", type=str,
        default="exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth")
    parser.add_argument(
        "--code_path", type=str,
        default="exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt")
    parser.add_argument("--csi_path", type=str, default="")
    parser.add_argument(
        "--output_dir", type=str,
        default="decoder_param_fm/reports/param_sensitivity")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise_sigmas", type=str,
        default="0,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2")
    return parser.parse_args()


def state_global_mse(a, b):
    se = 0.0
    numel = 0
    for name in a:
        diff = a[name].float() - b[name].float()
        se += diff.pow(2).sum().item()
        numel += diff.numel()
    return se / max(numel, 1)


def tensor_mse_rows(exp_name, generated_state, target_state):
    rows = []
    total_se = 0.0
    total_numel = 0
    for name, target in target_state.items():
        generated = generated_state[name]
        diff = generated.float() - target.float()
        mse = diff.pow(2).mean().item()
        rmse = mse ** 0.5
        target_rms = target.float().pow(2).mean().sqrt().item()
        rel_rmse = rmse / max(target_rms, 1e-12)
        rows.append({
            "exp_name": exp_name,
            "tensor": name,
            "shape": "x".join(str(v) for v in target.shape),
            "numel": target.numel(),
            "mse": mse,
            "rmse": rmse,
            "target_rms": target_rms,
            "rel_rmse": rel_rmse,
        })
        total_se += diff.pow(2).sum().item()
        total_numel += diff.numel()
    return rows, total_se / max(total_numel, 1)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_decoder_state(args, decoder_state, device, codes, csi):
    decoder, _ = build_decoder_from_args(args.decoder_args_json)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder = decoder.to(device).eval()
    error_sum = torch.zeros((), dtype=torch.float64, device=device)
    power_sum = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, codes.size(0), args.batch_size):
        end = min(start + args.batch_size, codes.size(0))
        code_batch = codes[start:end].to(device, non_blocking=True)
        csi_batch = csi[start:end].to(device, non_blocking=True)
        pred = decoder(code_batch)
        diff = pred - csi_batch
        error_sum += diff.double().pow(2).sum()
        power_sum += csi_batch.double().pow(2).sum()
    nmse = 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))
    return {
        "nmse_db": float(nmse.detach().cpu()),
        "mse_sum_per_sample": float(
            (error_sum / max(codes.size(0), 1)).detach().cpu()),
        "num_samples": int(codes.size(0)),
    }


def noisy_state(target_state, sigma, generator):
    out = {}
    for name, value in target_state.items():
        value = value.float()
        rms = value.pow(2).mean().sqrt().clamp_min(1e-12)
        noise = torch.randn(
            value.shape, dtype=value.dtype, generator=generator) * rms * sigma
        out[name] = value + noise
    return out


def analyze_generated(args, target_state, output_dir):
    tensor_rows = []
    summary_rows = []
    generated_paths = sorted(
        Path(args.exp_root).glob("*/generated/generated_decoder.pth"))
    for path in generated_paths:
        exp_name = path.parents[1].name
        generated_state = load_generated_decoder_state(path)
        rows, global_mse = tensor_mse_rows(
            exp_name, generated_state, target_state)
        tensor_rows.extend(rows)
        worst = max(rows, key=lambda row: row["mse"])
        summary_rows.append({
            "exp_name": exp_name,
            "global_param_mse": global_mse,
            "worst_tensor": worst["tensor"],
            "worst_tensor_mse": worst["mse"],
            "worst_tensor_rel_rmse": worst["rel_rmse"],
            "decoder_path": str(path),
        })
    summary_rows.sort(key=lambda row: row["global_param_mse"])
    write_csv(
        output_dir / "generated_tensor_mse.csv",
        tensor_rows,
        ["exp_name", "tensor", "shape", "numel", "mse", "rmse",
         "target_rms", "rel_rmse"])
    write_csv(
        output_dir / "generated_param_mse_summary.csv",
        summary_rows,
        ["exp_name", "global_param_mse", "worst_tensor",
         "worst_tensor_mse", "worst_tensor_rel_rmse", "decoder_path"])
    return summary_rows


def analyze_noise(args, target_state, output_dir, device):
    cfg = json.loads(Path(args.decoder_args_json).read_text(encoding="utf-8"))
    csi_path = args.csi_path or cfg["train_path"]
    codes = load_codes(args.code_path, max_samples=args.max_samples)
    csi = load_csi(
        csi_path,
        cfg.get("channel", 2),
        cfg.get("nt", 32),
        cfg.get("nc", 32),
        max_samples=args.max_samples)
    if codes.size(0) != csi.size(0):
        raise ValueError(f"code/csi size mismatch: {codes.size(0)} vs {csi.size(0)}")
    sigmas = [float(item) for item in args.noise_sigmas.split(",") if item]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    baseline = evaluate_decoder_state(args, target_state, device, codes, csi)
    rows = []
    for sigma in sigmas:
        if sigma == 0:
            state = target_state
        else:
            state = noisy_state(target_state, sigma, generator)
        metrics = evaluate_decoder_state(args, state, device, codes, csi)
        param_mse = state_global_mse(state, target_state)
        rows.append({
            "sigma_rel_tensor_rms": sigma,
            "global_param_mse": param_mse,
            "nmse_db": metrics["nmse_db"],
            "delta_nmse_db": metrics["nmse_db"] - baseline["nmse_db"],
            "mse_sum_per_sample": metrics["mse_sum_per_sample"],
            "num_samples": metrics["num_samples"],
        })
    write_csv(
        output_dir / "teacher_noise_sensitivity.csv",
        rows,
        ["sigma_rel_tensor_rms", "global_param_mse", "nmse_db",
         "delta_nmse_db", "mse_sum_per_sample", "num_samples"])
    return baseline, rows


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu
        else "cpu")
    target_state = extract_decoder_state(args.target_checkpoint)
    generated_summary = analyze_generated(args, target_state, output_dir)
    baseline, noise_rows = analyze_noise(args, target_state, output_dir, device)
    report = {
        "device": str(device),
        "target_checkpoint": args.target_checkpoint,
        "code_path": args.code_path,
        "csi_path": args.csi_path,
        "generated_count": len(generated_summary),
        "best_generated_by_param_mse": generated_summary[:5],
        "teacher_baseline": baseline,
        "noise_rows": noise_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
