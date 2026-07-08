import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoder_generalization_fm.dataset import parse_data_txt  # noqa: E402
from decoder_generalization_fm.param_utils import (  # noqa: E402
    clone_decoder_with_state,
    load_codes,
    load_csi,
)
from decoder_generalization_fm.sample import build_system, sample_tokens  # noqa: E402
from decoder_generalization_fm.param_utils import (  # noqa: E402
    denormalize_state,
    meta_tensors_from_meta,
    tokens_to_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated decoders for all data.txt entries.")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_txt", default="")
    parser.add_argument("--split", choices=["all", "train", "test"], default="all")
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_entries", type=int, default=0)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate_decoder(decoder, codes, csi, batch_size, device):
    error_sum = torch.zeros((), dtype=torch.float64, device=device)
    power_sum = torch.zeros((), dtype=torch.float64, device=device)
    n = min(codes.size(0), csi.size(0))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        code_batch = codes[start:end].to(device, non_blocking=True)
        csi_batch = csi[start:end].to(device, non_blocking=True)
        pred = decoder(code_batch)
        error_sum += (pred - csi_batch).double().pow(2).sum()
        power_sum += csi_batch.double().pow(2).sum()
    nmse = 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))
    return float(nmse.detach().cpu())


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    train_args = json.loads((exp_dir / "args.json").read_text(encoding="utf-8"))
    data_txt = args.data_txt or train_args["data_txt"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        exp_dir / "checkpoints" / "best_loss.pth")
    output_json = Path(args.output_json) if args.output_json else (
        exp_dir / "generated" / f"eval_{args.split}.json")
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.sample_seed)
    torch.cuda.manual_seed_all(args.sample_seed)

    meta = json.loads(
        (exp_dir / "artifacts" / "param_meta.json").read_text(encoding="utf-8"))
    stats = torch.load(
        exp_dir / "artifacts" / "train_tensor_zscore_stats.pt",
        weights_only=True,
        map_location="cpu")
    entries = parse_data_txt(data_txt)
    if args.split != "all":
        entries = [item for item in entries if item.split == args.split]
    if args.max_entries:
        entries = entries[:args.max_entries]

    first_codes = load_codes(
        entries[0].code_path,
        max_samples=train_args.get("max_condition_codes", 0))
    model = build_system(train_args, first_codes.size(1), meta).to(device)
    ckpt = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    meta_tensors = meta_tensors_from_meta(meta, device=device)
    rows = []
    for entry in entries:
        condition_codes = load_codes(
            entry.code_path,
            max_samples=train_args.get("max_condition_codes", 0)).to(device)
        theta_tokens = sample_tokens(
            model, meta_tensors, condition_codes, meta, args.ode_steps, device)
        norm_state = tokens_to_state(theta_tokens, meta)
        generated_state = denormalize_state(norm_state, stats)
        decoder = clone_decoder_with_state(
            entry.args_json, generated_state, device)
        cfg = json.loads(Path(entry.args_json).read_text(encoding="utf-8"))
        csi_path = cfg.get("train_path")
        codes = load_codes(entry.code_path, max_samples=args.max_samples)
        csi = load_csi(
            csi_path,
            cfg.get("channel", 2),
            cfg.get("nt", 32),
            cfg.get("nc", 32),
            max_samples=args.max_samples)
        nmse = evaluate_decoder(decoder, codes, csi, args.batch_size, device)
        row = {
            "split": entry.split,
            "exp_dir": str(entry.exp_dir),
            "encoder": entry.encoder,
            "seed": entry.seed,
            "nmse_db": nmse,
        }
        rows.append(row)
        print(f"{entry.split},{entry.exp_dir},{nmse:.6e}dB")
    by_split = {}
    for split in sorted({row["split"] for row in rows}):
        vals = [row["nmse_db"] for row in rows if row["split"] == split]
        by_split[split] = {
            "n": len(vals),
            "mean_nmse_db": sum(vals) / max(len(vals), 1),
        }
    result = {
        "exp_dir": str(exp_dir),
        "checkpoint": str(checkpoint),
        "data_txt": str(data_txt),
        "split": args.split,
        "ode_steps": args.ode_steps,
        "sample_seed": args.sample_seed,
        "summary": by_split,
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"saved={output_json}")


if __name__ == "__main__":
    main()
