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

from decoder_param_fm.analyze_decoder_param_sensitivity import (  # noqa: E402
    evaluate_decoder_state,
)
from decoder_param_fm.param_utils import (  # noqa: E402
    extract_decoder_state,
    load_codes,
    load_csi,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Noise sensitivity for selected decoder parameter scopes.")
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
        default="decoder_param_fm/reports/scoped_noise_sensitivity")
    parser.add_argument("--scopes", type=str, default="fc,ffn")
    parser.add_argument(
        "--noise_sigmas", type=str,
        default="0,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def in_scope(name, scope):
    if scope == "fc":
        return name.startswith("fc_decoder.")
    if scope == "ffn":
        return (
            ".linear1." in name
            or ".linear2." in name
        )
    if scope == "fc_ffn":
        return in_scope(name, "fc") or in_scope(name, "ffn")
    raise ValueError(f"Unknown scope: {scope}")


def mse_for_names(a, b, names):
    se = 0.0
    numel = 0
    for name in names:
        diff = a[name].float() - b[name].float()
        se += diff.pow(2).sum().item()
        numel += diff.numel()
    return se / max(numel, 1)


def noisy_scoped_state(target_state, selected_names, sigma, generator):
    out = {name: value.detach().clone().float()
           for name, value in target_state.items()}
    for name in selected_names:
        value = out[name]
        rms = value.pow(2).mean().sqrt().clamp_min(1e-12)
        noise = torch.randn(
            value.shape, dtype=value.dtype, generator=generator) * rms * sigma
        out[name] = value + noise
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "selected_tensors",
        "selected_params",
        "sigma_rel_tensor_rms",
        "scoped_param_mse",
        "global_param_mse",
        "nmse_db",
        "delta_nmse_db",
        "mse_sum_per_sample",
        "num_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu
        else "cpu")

    cfg = json.loads(Path(args.decoder_args_json).read_text(encoding="utf-8"))
    csi_path = args.csi_path or cfg["train_path"]
    codes = load_codes(args.code_path, max_samples=args.max_samples)
    csi = load_csi(
        csi_path,
        cfg.get("channel", 2),
        cfg.get("nt", 32),
        cfg.get("nc", 32),
        max_samples=args.max_samples)
    target_state = extract_decoder_state(args.target_checkpoint)
    baseline = evaluate_decoder_state(args, target_state, device, codes, csi)

    scopes = [item.strip() for item in args.scopes.split(",") if item.strip()]
    sigmas = [float(item) for item in args.noise_sigmas.split(",") if item]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    all_names = list(target_state.keys())
    rows = []
    summary = {
        "device": str(device),
        "baseline": baseline,
        "scopes": {},
    }

    for scope in scopes:
        selected = [name for name in all_names if in_scope(name, scope)]
        selected_params = sum(target_state[name].numel() for name in selected)
        scope_rows = []
        for sigma in sigmas:
            if sigma == 0:
                state = target_state
            else:
                state = noisy_scoped_state(target_state, selected, sigma, generator)
            metrics = evaluate_decoder_state(args, state, device, codes, csi)
            row = {
                "scope": scope,
                "selected_tensors": len(selected),
                "selected_params": selected_params,
                "sigma_rel_tensor_rms": sigma,
                "scoped_param_mse": mse_for_names(state, target_state, selected),
                "global_param_mse": mse_for_names(state, target_state, all_names),
                "nmse_db": metrics["nmse_db"],
                "delta_nmse_db": metrics["nmse_db"] - baseline["nmse_db"],
                "mse_sum_per_sample": metrics["mse_sum_per_sample"],
                "num_samples": metrics["num_samples"],
            }
            rows.append(row)
            scope_rows.append(row)
        summary["scopes"][scope] = {
            "selected_tensors": len(selected),
            "selected_params": selected_params,
            "selected_names": selected,
            "rows": scope_rows,
        }

    write_csv(output_dir / "scoped_noise_sensitivity.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
