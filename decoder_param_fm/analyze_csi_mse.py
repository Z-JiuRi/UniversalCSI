"""
Analyze per-sample MSE statistics of generated CSI vs ground truth.
For each completed experiment, produces:
  - Per-sample MSE distribution (mean, std, percentiles)
  - Aggregated MSE / NMSE
  - Saves per-sample MSE tensor for further analysis
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decoder_param_fm.param_utils import (
    build_decoder_from_args,
    load_codes,
    load_csi,
    load_generated_decoder_state,
)


def per_sample_mse_stats(decoder, codes, csi, batch_size, device):
    """
    Run decoder on codes, compare with ground-truth CSI,
    return per-sample MSE and power tensors.
    """
    decoder.eval()
    sample_mses = []
    sample_powers = []
    with torch.no_grad():
        for start in range(0, codes.size(0), batch_size):
            end = min(start + batch_size, codes.size(0))
            code_batch = codes[start:end].to(device, non_blocking=True)
            csi_batch = csi[start:end].to(device, non_blocking=True)
            pred = decoder(code_batch)                           # (B, C, nt, nc)
            diff = pred - csi_batch
            # Per-sample MSE = mean over all elements (C*nt*nc)
            mse = diff.pow(2).mean(dim=(1, 2, 3))                # (B,)
            power = csi_batch.pow(2).mean(dim=(1, 2, 3))         # (B,)
            sample_mses.append(mse.cpu())
            sample_powers.append(power.cpu())
    sample_mses = torch.cat(sample_mses)
    sample_powers = torch.cat(sample_powers)
    # Per-sample NMSE in dB
    sample_nmses = 10.0 * torch.log10(
        sample_mses / sample_powers.clamp_min(1e-12))
    return sample_mses, sample_powers, sample_nmses


def compute_stats(tensor, name=""):
    """Compute summary statistics for a 1-D tensor."""
    stats = {
        "name": name,
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "median": float(tensor.median()),
        "p5": float(tensor.kthvalue(max(1, int(0.05 * len(tensor)))).values),
        "p25": float(tensor.kthvalue(max(1, int(0.25 * len(tensor)))).values),
        "p75": float(tensor.kthvalue(max(1, int(0.75 * len(tensor)))).values),
        "p95": float(tensor.kthvalue(max(1, int(0.95 * len(tensor)))).values),
        "num_samples": len(tensor),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-sample MSE of generated decoders.")
    parser.add_argument("--exp_names", type=str, nargs="*", default=[],
                        help="Experiment names. Empty = auto-detect all completed.")
    parser.add_argument("--exps_base", type=str,
                        default="decoder_param_fm/exps")
    parser.add_argument("--target_exp", type=str,
                        default="exps/COST2100/in/seed42/transnet_transnet")
    parser.add_argument("--code_path", type=str, default="")
    parser.add_argument("--csi_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="",
                        help="If set, saves combined results to this JSON.")
    args = parser.parse_args()

    target_exp = Path(args.target_exp)
    decoder_args_json = target_exp / "args.json"
    exps_base = Path(args.exps_base)

    # Default code/csi paths: test set
    code_path = args.code_path or str(target_exp / "codewords" / "test_code.pt")
    csi_path = args.csi_path or "/storage/hujiacong/zxd/datasets/cost2100/in_test.pt"

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Codes: {code_path}")
    print(f"CSI:   {csi_path}")

    # Load reference data once
    print("Loading reference data...")
    codes = load_codes(code_path)
    csi = load_csi(
        csi_path,
        2, 32, 32,  # channel, nt, nc from default (matches target exp)
    )
    print(f"  codes: {codes.shape}, csi: {csi.shape}")

    # Auto-detect completed experiments if none specified
    if not args.exp_names:
        print("\nAuto-detecting completed experiments...")
        for d in sorted(exps_base.iterdir()):
            if d.is_dir():
                log_file = d / "run.log"
                gen_file = d / "generated" / "generated_decoder.pth"
                if log_file.exists() and gen_file.exists():
                    # Check for Finished training.
                    content = log_file.read_text(encoding="utf-8", errors="replace")
                    if "Finished training." in content:
                        args.exp_names.append(d.name)
        print(f"  Found {len(args.exp_names)} completed experiments")

    all_results = {}
    for exp_name in args.exp_names:
        exp_dir = exps_base / exp_name
        gen_decoder_path = exp_dir / "generated" / "generated_decoder.pth"
        output_dir = exp_dir / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not gen_decoder_path.exists():
            print(f"\n[SKIP] {exp_name}: no generated decoder at {gen_decoder_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[{exp_name}]")
        print(f"{'='*60}")

        # Load decoder
        print("  Loading generated decoder...")
        decoder_state = load_generated_decoder_state(str(gen_decoder_path))
        decoder, _ = build_decoder_from_args(str(decoder_args_json))
        missing, unexpected = decoder.load_state_dict(decoder_state, strict=True)
        if missing or unexpected:
            print(f"  WARNING: load mismatch: missing={missing}, unexpected={unexpected}")
        decoder = decoder.to(device)

        # Compute per-sample MSE
        print("  Computing per-sample MSE...")
        sample_mses, sample_powers, sample_nmses = per_sample_mse_stats(
            decoder, codes, csi, args.batch_size, device)

        mse_stats = compute_stats(sample_mses, "per_sample_mse")
        power_stats = compute_stats(sample_powers, "per_sample_power")
        nmse_stats = compute_stats(sample_nmses, "per_sample_nmse_db")

        # Aggregate (same formula as test_generated_nmse.py)
        total_error = sample_mses.sum() * (2 * 32 * 32)  # undo mean -> sum
        total_power = sample_powers.sum() * (2 * 32 * 32)
        # Actually: sample_mse = mean over elements. So total_error = sum(sample_mse * num_elements)
        num_elements = 2 * 32 * 32
        agg_mse = (sample_mses * num_elements).sum() / (len(sample_mses) * num_elements)
        agg_nmse = 10.0 * torch.log10(
            (sample_mses * num_elements).sum() / (sample_powers * num_elements).sum().clamp_min(1e-12)
        )

        result = {
            "exp_name": exp_name,
            "num_samples": len(sample_mses),
            "aggregate": {
                "mse": float(agg_mse),
                "nmse_db": float(agg_nmse),
            },
            "per_sample_mse": mse_stats,
            "per_sample_power": power_stats,
            "per_sample_nmse_db": nmse_stats,
        }
        all_results[exp_name] = result

        print(f"  Aggregate MSE: {agg_mse:.6f}")
        print(f"  Aggregate NMSE: {agg_nmse:.2f} dB")
        print(f"  Per-sample MSE: mean={mse_stats['mean']:.6f}, "
              f"std={mse_stats['std']:.6f}, "
              f"median={mse_stats['median']:.6f}")
        print(f"  Per-sample NMSE: mean={nmse_stats['mean']:.2f} dB, "
              f"std={nmse_stats['std']:.2f} dB, "
              f"[p5, p95] = [{nmse_stats['p5']:.2f}, {nmse_stats['p95']:.2f}] dB")

        # Save per-sample tensors for later analysis
        mse_pt_path = output_dir / "per_sample_mse.pt"
        torch.save({
            "sample_mse": sample_mses,
            "sample_power": sample_powers,
            "sample_nmse": sample_nmses,
            "exp_name": exp_name,
        }, str(mse_pt_path))
        print(f"  Saved per-sample MSE tensor to {mse_pt_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<55s} {'Agg NMSE':>10s} {'MSE mean':>10s} {'MSE std':>10s} "
          f"{'NMSE p5':>10s} {'NMSE p95':>10s}")
    print("-" * 105)
    for exp_name in args.exp_names:
        if exp_name in all_results:
            r = all_results[exp_name]
            ms = r["per_sample_mse"]
            ns = r["per_sample_nmse_db"]
            print(f"{exp_name:<55s} {r['aggregate']['nmse_db']:>10.2f} "
                  f"{ms['mean']:>10.6f} {ms['std']:>10.6f} "
                  f"{ns['p5']:>10.2f} {ns['p95']:>10.2f}")

    # Save combined JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(all_results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"\nSaved combined results to {output_path}")


if __name__ == "__main__":
    main()
