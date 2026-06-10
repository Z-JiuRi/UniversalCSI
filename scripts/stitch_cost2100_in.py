#!/usr/bin/env python3
"""Evaluate encoder/decoder checkpoint stitching on COST2100 indoor.

The default mode keeps the architecture fixed and stitches weights trained
with different seeds. For each run name under exps/COST2100/in, it evaluates
all ordered seed pairs with encoder_seed != decoder_seed:

  encoder from seed A + decoder from seed B

The resulting metrics are written to CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import universal_csi
from utils.statics import evaluator, nmse_from_sums


DEFAULT_ROOT = Path("exps/COST2100/in")
DEFAULT_OUTPUT = Path("exps/COST2100/in/stitch_tests")
DEFAULT_RUNS = ("transnet_transnet", "transnet_hybrid")
DEFAULT_SEEDS = (42, 2026, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch encoder weights from one checkpoint with decoder "
                    "weights from another checkpoint and test on COST2100/in.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Root directory containing seed subdirectories.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds to evaluate.")
    parser.add_argument("--runs", type=str, nargs=2, default=list(DEFAULT_RUNS),
                        metavar=("RUN_A", "RUN_B"),
                        help="Two run names to evaluate.")
    parser.add_argument("--mode", choices=["cross-seed", "cross-run"],
                        default="cross-seed",
                        help="cross-seed keeps architecture fixed and stitches "
                             "different seeds; cross-run stitches the two runs "
                             "within each seed.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for results CSV/JSON.")
    parser.add_argument("--output-name", type=str, default="results",
                        help="Base filename for results without extension.")
    parser.add_argument("--checkpoint-name", type=str, default="best_nmse.pth",
                        help="Checkpoint filename under checkpoints/.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override test batch size. Defaults to args.json.")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker count.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even when CUDA is available.")
    parser.add_argument("--gpu", type=int, default=None,
                        help="CUDA_VISIBLE_DEVICES id to use.")
    parser.add_argument("--print-freq", type=int, default=20,
                        help="Progress print frequency in test batches.")
    return parser.parse_args()


def find_run_dir(seed_dir: Path, run_name: str, checkpoint_name: str) -> Path:
    candidates = [
        seed_dir / run_name,
        seed_dir / run_name / "base",
    ]
    candidates.extend(sorted(seed_dir.glob(f"{run_name}/**/args.json")))

    seen = set()
    run_dirs = []
    for candidate in candidates:
        run_dir = candidate.parent if candidate.name == "args.json" else candidate
        if run_dir in seen:
            continue
        seen.add(run_dir)
        if ((run_dir / "args.json").is_file()
                and (run_dir / "checkpoints" / checkpoint_name).is_file()):
            run_dirs.append(run_dir)

    if not run_dirs:
        raise FileNotFoundError(
            f"Could not find {run_name} with args.json and checkpoints/"
            f"{checkpoint_name} under {seed_dir}")
    if len(run_dirs) > 1:
        run_dirs.sort(key=lambda p: len(p.parts))
    return run_dirs[0]


def load_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def run_label(run_dir: Path) -> str:
    if run_dir.name == "base":
        return run_dir.parent.name
    return run_dir.name


def checkpoint_state(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict]:
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    state = checkpoint["state_dict"]
    cleaned = {
        key: value
        for key, value in state.items()
        if not (key.endswith("total_ops") or key.endswith("total_params"))
    }
    return cleaned, checkpoint


def copy_prefixed(dst: Dict[str, torch.Tensor],
                  src: Dict[str, torch.Tensor],
                  prefix: str) -> int:
    copied = 0
    for key, value in src.items():
        if not key.startswith(prefix):
            continue
        if key not in dst:
            raise KeyError(f"Unexpected key for target model: {key}")
        if tuple(dst[key].shape) != tuple(value.shape):
            raise ValueError(
                f"Shape mismatch for {key}: target {tuple(dst[key].shape)} "
                f"vs source {tuple(value.shape)}")
        dst[key] = value
        copied += 1
    if copied == 0:
        raise ValueError(f"No parameters copied for prefix {prefix}")
    return copied


def compatible_model_args(encoder_args: Dict, decoder_args: Dict) -> Dict:
    shared_keys = [
        "cr", "d_model", "channel", "nt", "nc", "dim_feedforward",
        "code_adapter",
    ]
    for key in shared_keys:
        enc_value = encoder_args.get(key)
        dec_value = decoder_args.get(key)
        if enc_value != dec_value:
            raise ValueError(
                f"Cannot stitch checkpoints with different {key}: "
                f"encoder={enc_value}, decoder={dec_value}")

    return {
        "encoder_name": encoder_args["encoder"],
        "decoder_name": decoder_args["decoder"],
        "reduction": encoder_args["cr"],
        "d_model": encoder_args["d_model"],
        "channel": encoder_args["channel"],
        "nt": encoder_args["nt"],
        "nc": encoder_args["nc"],
        "dim_feedforward": encoder_args.get("dim_feedforward"),
        "code_adapter": encoder_args.get("code_adapter", False),
        "hidden": decoder_args.get("hidden", 16),
        "num_blocks": decoder_args.get("num_blocks", 2),
    }


def build_stitched_model(encoder_run: Path, decoder_run: Path,
                         checkpoint_name: str) -> Tuple[nn.Module, Dict]:
    encoder_args = load_json(encoder_run / "args.json")
    decoder_args = load_json(decoder_run / "args.json")
    model_args = compatible_model_args(encoder_args, decoder_args)
    model = universal_csi(**model_args)

    encoder_state, encoder_ckpt = checkpoint_state(
        encoder_run / "checkpoints" / checkpoint_name)
    decoder_state, decoder_ckpt = checkpoint_state(
        decoder_run / "checkpoints" / checkpoint_name)

    stitched_state = model.state_dict()
    encoder_keys = copy_prefixed(stitched_state, encoder_state, "encoder.")
    decoder_keys = copy_prefixed(stitched_state, decoder_state, "decoder.")
    adapter_keys = 0
    if model_args["code_adapter"]:
        adapter_keys = copy_prefixed(stitched_state, encoder_state,
                                     "code_adapter.")
    model.load_state_dict(stitched_state, strict=True)

    metadata = {
        "encoder_run": str(encoder_run),
        "decoder_run": str(decoder_run),
        "encoder_checkpoint": str(encoder_run / "checkpoints" / checkpoint_name),
        "decoder_checkpoint": str(decoder_run / "checkpoints" / checkpoint_name),
        "encoder_checkpoint_best_nmse": encoder_ckpt.get("best_nmse"),
        "decoder_checkpoint_best_nmse": decoder_ckpt.get("best_nmse"),
        "model_args": model_args,
        "copied_encoder_keys": encoder_keys,
        "copied_decoder_keys": decoder_keys,
        "copied_adapter_keys": adapter_keys,
    }
    return model, metadata


def load_test_loader(args_json: Dict, batch_size: Optional[int],
                     workers: int) -> DataLoader:
    test_path = Path(args_json["test_path"])
    data = torch.load(test_path, weights_only=True,
                      map_location=torch.device("cpu")).to(torch.float32)
    expected = (args_json["channel"], args_json["nt"], args_json["nc"])
    if data.ndim == 2:
        data = data.view(-1, *expected)
    if data.ndim != 4 or tuple(data.shape[1:]) != expected:
        raise ValueError(
            f"{test_path} should have shape (N, {expected[0]}, "
            f"{expected[1]}, {expected[2]}), got {tuple(data.shape)}")
    return DataLoader(
        TensorDataset(data),
        batch_size=batch_size or args_json["batch_size"],
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             print_freq: int) -> Tuple[float, float]:
    criterion = nn.MSELoss().to(device)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_error = torch.tensor(0., device=device)
    total_power = torch.tensor(0., device=device)
    count = 0
    with torch.no_grad():
        for batch_idx, (sparse_gt,) in enumerate(loader):
            sparse_gt = sparse_gt.to(device)
            sparse_pred = model(sparse_gt)
            loss = criterion(sparse_pred, sparse_gt)
            error_sum, power_sum = evaluator(sparse_pred, sparse_gt)
            total_error += error_sum
            total_power += power_sum
            nmse = nmse_from_sums(total_error, total_power)
            total_loss += float(loss.detach().cpu())
            count += 1
            if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
                print(f"  [{batch_idx + 1}/{len(loader)}] "
                      f"loss={total_loss / count:.4e} "
                      f"NMSE={float(nmse.detach().cpu()):.4e}")
    nmse = nmse_from_sums(total_error, total_power)
    return total_loss / count, float(nmse.detach().cpu())


def evaluate_pair(seed: int, encoder_run: Path, decoder_run: Path,
                  args: argparse.Namespace, device: torch.device,
                  encoder_seed: Optional[int] = None,
                  decoder_seed: Optional[int] = None) -> Dict:
    encoder_args = load_json(encoder_run / "args.json")
    decoder_args = load_json(decoder_run / "args.json")
    label = (
        f"encoder=seed{encoder_seed if encoder_seed is not None else seed}_"
        f"{run_label(encoder_run)} "
        f"decoder=seed{decoder_seed if decoder_seed is not None else seed}_"
        f"{run_label(decoder_run)}")
    print(f"\n==> {label}")
    model, metadata = build_stitched_model(
        encoder_run, decoder_run, args.checkpoint_name)
    loader = load_test_loader(encoder_args, args.batch_size, args.workers)
    loss, nmse = evaluate(model, loader, device, args.print_freq)
    row = {
        "seed": seed,
        "encoder_seed": encoder_seed if encoder_seed is not None else seed,
        "decoder_seed": decoder_seed if decoder_seed is not None else seed,
        "encoder_run_name": run_label(encoder_run),
        "decoder_run_name": run_label(decoder_run),
        "encoder_arch": encoder_args["encoder"],
        "decoder_arch": decoder_args["decoder"],
        "test_loss": loss,
        "test_nmse": nmse,
        "encoder_best_nmse": (
            metadata["encoder_checkpoint_best_nmse"] or {}).get("nmse"),
        "encoder_best_epoch": (
            metadata["encoder_checkpoint_best_nmse"] or {}).get("epoch"),
        "decoder_best_nmse": (
            metadata["decoder_checkpoint_best_nmse"] or {}).get("nmse"),
        "decoder_best_epoch": (
            metadata["decoder_checkpoint_best_nmse"] or {}).get("epoch"),
        "encoder_run": metadata["encoder_run"],
        "decoder_run": metadata["decoder_run"],
        "encoder_checkpoint": metadata["encoder_checkpoint"],
        "decoder_checkpoint": metadata["decoder_checkpoint"],
        "copied_encoder_keys": metadata["copied_encoder_keys"],
        "copied_decoder_keys": metadata["copied_decoder_keys"],
    }
    print(f"  => loss={loss:.4e} NMSE={nmse:.4e}")
    return row


def make_cross_run_rows(args: argparse.Namespace,
                        device: torch.device) -> List[Dict]:
    rows = []
    for seed in args.seeds:
        seed_dir = args.root / f"seed{seed}"
        run_a = find_run_dir(seed_dir, args.runs[0], args.checkpoint_name)
        run_b = find_run_dir(seed_dir, args.runs[1], args.checkpoint_name)
        pairs = [(run_a, run_b), (run_b, run_a)]

        for encoder_run, decoder_run in pairs:
            rows.append(evaluate_pair(seed, encoder_run, decoder_run, args,
                                      device))
    return rows


def make_cross_seed_rows(args: argparse.Namespace,
                         device: torch.device) -> List[Dict]:
    rows = []
    for run_name in args.runs:
        run_dirs = {}
        for seed in args.seeds:
            seed_dir = args.root / f"seed{seed}"
            run_dirs[seed] = find_run_dir(seed_dir, run_name,
                                          args.checkpoint_name)

        for decoder_seed in args.seeds:
            for encoder_seed in args.seeds:
                if encoder_seed == decoder_seed:
                    continue
                seed = encoder_seed
                rows.append(evaluate_pair(
                    seed,
                    run_dirs[encoder_seed],
                    run_dirs[decoder_seed],
                    args,
                    device,
                    encoder_seed=encoder_seed,
                    decoder_seed=decoder_seed,
                ))
    return rows


def write_results(rows: Iterable[Dict], output_dir: Path,
                  output_name: str) -> Tuple[Path, Path]:
    rows = list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_name}.csv"
    json_path = output_dir / f"{output_name}.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    return csv_path, json_path


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu")
    print(f"=> PyTorch {torch.__version__}; device={device}")
    if args.mode == "cross-seed":
        rows = make_cross_seed_rows(args, device)
    else:
        rows = make_cross_run_rows(args, device)
    csv_path, json_path = write_results(rows, args.output_dir,
                                        args.output_name)
    print(f"\n=> Wrote {csv_path}")
    print(f"=> Wrote {json_path}")


if __name__ == "__main__":
    main()
