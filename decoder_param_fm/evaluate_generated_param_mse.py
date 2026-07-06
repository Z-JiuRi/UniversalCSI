import argparse
import csv
import json
import subprocess
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
        description="Generate decoder parameters and compare them with target.")
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument(
        "--guide_code_path", type=str,
        default="exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt")
    parser.add_argument(
        "--target_checkpoint", type=str,
        default="exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth")
    parser.add_argument(
        "--decoder_args_json", type=str,
        default="exps/COST2100/in/seed42/transnet_transnet/args.json")
    parser.add_argument("--csi_path", type=str, default="")
    parser.add_argument(
        "--output_dir", type=str,
        default="decoder_param_fm/reports/generated_param_mse")
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_guide_codes", type=int, default=0)
    parser.add_argument("--force_sample", action="store_true")
    return parser.parse_args()


def sample_if_needed(exp_dir, args):
    exp_dir = Path(exp_dir)
    output = exp_dir / "generated" / "generated_decoder.pth"
    if output.exists() and not args.force_sample:
        return output

    cmd = [
        sys.executable,
        "-u",
        "decoder_param_fm/sample_param_fm.py",
        "--exp_dir",
        str(exp_dir),
        "--checkpoint",
        str(exp_dir / "checkpoints" / "best_loss.pth"),
        "--guide_code_path",
        args.guide_code_path,
        "--output",
        str(output),
        "--ode_steps",
        str(args.ode_steps),
        "--gpu",
        str(args.gpu),
        "--max_guide_codes",
        str(args.max_guide_codes),
    ]
    if args.cpu:
        cmd.append("--cpu")
    subprocess.run(cmd, check=True)
    return output


def compare_state(exp_name, generated_state, target_state):
    rows = []
    total_se = torch.zeros((), dtype=torch.float64)
    total_power = torch.zeros((), dtype=torch.float64)
    total_numel = 0

    for name, target in target_state.items():
        generated = generated_state[name].float()
        target = target.float()
        diff = generated - target
        se = diff.double().pow(2).sum()
        power = target.double().pow(2).sum()
        numel = target.numel()
        mse = se / max(numel, 1)
        nmse_db = 10.0 * torch.log10(se / power.clamp_min(1e-12))
        target_rms = torch.sqrt(power / max(numel, 1))
        rmse = torch.sqrt(mse)
        rows.append({
            "exp_name": exp_name,
            "tensor": name,
            "shape": "x".join(str(v) for v in target.shape),
            "numel": int(numel),
            "mse": float(mse),
            "rmse": float(rmse),
            "target_rms": float(target_rms),
            "param_nmse_db": float(nmse_db),
        })
        total_se += se
        total_power += power
        total_numel += numel

    global_mse = total_se / max(total_numel, 1)
    global_nmse_db = 10.0 * torch.log10(total_se / total_power.clamp_min(1e-12))
    worst_mse = max(rows, key=lambda row: row["mse"])
    worst_nmse = max(rows, key=lambda row: row["param_nmse_db"])
    summary = {
        "exp_name": exp_name,
        "global_param_mse": float(global_mse),
        "global_param_nmse_db": float(global_nmse_db),
        "total_numel": int(total_numel),
        "worst_tensor_by_mse": worst_mse["tensor"],
        "worst_tensor_mse": worst_mse["mse"],
        "worst_tensor_by_nmse": worst_nmse["tensor"],
        "worst_tensor_nmse_db": worst_nmse["param_nmse_db"],
    }
    return rows, summary


@torch.no_grad()
def evaluate_csi_nmse(args, decoder_state):
    decoder_cfg = json.loads(Path(args.decoder_args_json).read_text(encoding="utf-8"))
    csi_path = args.csi_path or decoder_cfg["train_path"]
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu
        else "cpu")
    codes = load_codes(args.guide_code_path, max_samples=args.max_guide_codes)
    csi = load_csi(
        csi_path,
        decoder_cfg.get("channel", 2),
        decoder_cfg.get("nt", 32),
        decoder_cfg.get("nc", 32),
        max_samples=args.max_guide_codes,
    )
    if codes.size(0) != csi.size(0):
        raise ValueError(f"code/csi sample mismatch: {codes.size(0)} vs {csi.size(0)}")

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
        "csi_path": str(csi_path),
        "num_samples": int(codes.size(0)),
        "csi_mse_sum_per_sample": float(
            (error_sum / max(codes.size(0), 1)).detach().cpu()),
        "csi_nmse_db": float(nmse.detach().cpu()),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def write_markdown(path, args, summaries, rows_by_exp):
    lines = [
        "# Generated Decoder Parameter MSE",
        "",
        "## Setup",
        "",
        f"- Guide code: `{args.guide_code_path}`",
        f"- Target checkpoint: `{args.target_checkpoint}`",
        f"- Decoder args: `{args.decoder_args_json}`",
        f"- ODE steps: `{args.ode_steps}`",
        f"- Max guide codes: `{args.max_guide_codes}`",
        "",
        "## Summary",
        "",
        "| Experiment | Param Global MSE | CSI NMSE (dB) | CSI MSE Sum/Sample | Samples | Worst Tensor by MSE | Worst MSE |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['exp_name']}` | {fmt(item['global_param_mse'])} | "
            f"{fmt(item['csi_nmse_db'])} | {fmt(item['csi_mse_sum_per_sample'])} | "
            f"{item['num_samples']} | `{item['worst_tensor_by_mse']}` | "
            f"{fmt(item['worst_tensor_mse'])} |")

    for summary in summaries:
        exp_name = summary["exp_name"]
        lines.extend([
            "",
            f"## {exp_name}",
            "",
            "| Tensor | Shape | Numel | MSE | RMSE | Target RMS | Param NMSE (dB) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows_by_exp[exp_name]:
            lines.append(
                f"| `{row['tensor']}` | `{row['shape']}` | {row['numel']} | "
                f"{fmt(row['mse'])} | {fmt(row['rmse'])} | "
                f"{fmt(row['target_rms'])} | {fmt(row['param_nmse_db'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_state = extract_decoder_state(args.target_checkpoint)

    all_rows = []
    summaries = []
    rows_by_exp = {}
    generated_paths = {}
    for exp_dir in args.exp_dirs:
        exp_path = Path(exp_dir)
        generated_path = sample_if_needed(exp_path, args)
        generated_state = load_generated_decoder_state(generated_path)
        rows, summary = compare_state(exp_path.name, generated_state, target_state)
        csi_metrics = evaluate_csi_nmse(args, generated_state)
        summary.update(csi_metrics)
        summary["generated_path"] = str(generated_path)
        all_rows.extend(rows)
        summaries.append(summary)
        rows_by_exp[exp_path.name] = rows
        generated_paths[exp_path.name] = str(generated_path)

    summaries.sort(key=lambda row: row["global_param_mse"])
    write_csv(
        output_dir / "tensor_mse.csv",
        all_rows,
        ["exp_name", "tensor", "shape", "numel", "mse", "rmse",
         "target_rms", "param_nmse_db"])
    write_csv(
        output_dir / "summary.csv",
        summaries,
        ["exp_name", "global_param_mse", "global_param_nmse_db",
         "csi_nmse_db", "csi_mse_sum_per_sample", "num_samples", "csi_path",
         "total_numel", "worst_tensor_by_mse", "worst_tensor_mse",
         "worst_tensor_by_nmse", "worst_tensor_nmse_db", "generated_path"])
    report = {
        "guide_code_path": args.guide_code_path,
        "target_checkpoint": args.target_checkpoint,
        "decoder_args_json": args.decoder_args_json,
        "ode_steps": args.ode_steps,
        "max_guide_codes": args.max_guide_codes,
        "generated_paths": generated_paths,
        "summary": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(output_dir / "generated_param_mse.md", args, summaries,
                   rows_by_exp)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
