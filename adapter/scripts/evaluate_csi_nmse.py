#!/usr/bin/env python
import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter.models import build_mapper  # noqa: E402
from adapter.train_adapter import load_csi, split_paths  # noqa: E402
from utils.statics import evaluator, nmse_from_sums  # noqa: E402


PROCESS_PATTERN = r"adapter/(train_multi_adapter|train_adapter.py)"


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def process_running():
    result = subprocess.run(
        ["pgrep", "-af", PROCESS_PATTERN],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def load_main_models_package():
    package_name = "adapter_eval_main_models"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "models" / "__init__.py",
        submodule_search_locations=[str(ROOT / "models")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def clean_state_dict(checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    for key in list(state_dict.keys()):
        if key.endswith("total_ops") or key.endswith("total_params"):
            del state_dict[key]
    return state_dict


def count_params(model):
    return sum(param.numel() for param in model.parameters())


def build_original_model(exp_dir, device):
    args = load_json(Path(exp_dir) / "args.json")
    if not args:
        raise FileNotFoundError(f"missing args.json: {exp_dir}")
    main_models = load_main_models_package()
    model = main_models.universal_csi(
        encoder_name=args.get("encoder", "transnet"),
        decoder_name=args.get("decoder", "transnet"),
        reduction=args.get("cr", 4),
        d_model=args.get("d_model", 64),
        channel=args.get("channel", 2),
        nt=args.get("nt", 32),
        nc=args.get("nc", 32),
        dim_feedforward=args.get("dim_feedforward", 2048),
        hidden=args.get("hidden", 16),
        num_blocks=args.get("num_blocks", 2))
    checkpoint = Path(exp_dir) / "checkpoints" / "best_nmse.pth"
    model.load_state_dict(clean_state_dict(checkpoint))
    model.to(device).eval()
    return model, args, checkpoint


def build_adapter_model(exp_dir, device):
    args = load_json(Path(exp_dir) / "args.json")
    if not args:
        raise FileNotFoundError(f"missing args.json: {exp_dir}")
    decoder_args = load_json(args.get("decoder_args_json"))
    decoder_cfg = decoder_args or args
    main_models = load_main_models_package()
    wrapper = main_models.universal_csi(
        encoder_name="transnet",
        decoder_name=decoder_cfg.get("decoder", args.get("decoder", "transnet")),
        reduction=decoder_cfg.get("cr", args.get("cr", 4)),
        d_model=decoder_cfg.get("d_model", args.get("d_model", 64)),
        channel=decoder_cfg.get("channel", args.get("channel", 2)),
        nt=decoder_cfg.get("nt", args.get("nt", 32)),
        nc=decoder_cfg.get("nc", args.get("nc", 32)),
        dim_feedforward=decoder_cfg.get("dim_feedforward", args.get("dim_feedforward", 2048)),
        hidden=decoder_cfg.get("hidden", args.get("hidden", 16)),
        num_blocks=decoder_cfg.get("num_blocks", args.get("decoder_num_blocks", 2)))
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in clean_state_dict(args["decoder_checkpoint"]).items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        decoder_state = clean_state_dict(args["decoder_checkpoint"])
    wrapper.decoder.load_state_dict(decoder_state, strict=False)
    decoder = wrapper.decoder.to(device).eval()

    affine = torch.load(Path(exp_dir) / "affine_alignment.pt",
                        weights_only=True,
                        map_location="cpu")
    mapper = build_mapper(
        args.get("mapper_type", "affine_residual_mlp"),
        affine["weight"],
        affine["bias"],
        hidden_dim=args.get("hidden_dim", 1024),
        lowrank_rank=args.get("lowrank_rank", 64),
        num_blocks=args.get("num_blocks", 4),
        dropout=args.get("dropout", 0.0),
        residual_scale=args.get("residual_scale", 0.1),
        use_block_norm=not args.get("no_block_norm", False),
        use_final_norm=args.get("use_final_norm", False),
        train_affine=args.get("train_affine", False),
        learnable_residual_gate=args.get("learnable_residual_gate", False),
        gate_max=args.get("gate_max", 0.5))
    ckpt = Path(exp_dir) / "checkpoints" / "best_decoder_nmse.pth"
    if not ckpt.exists():
        ckpt = Path(exp_dir) / "checkpoints" / "best_code_mse.pth"
    if not ckpt.exists():
        ckpt = Path(exp_dir) / "checkpoints" / "last.pth"
    state = torch.load(ckpt, weights_only=True, map_location="cpu")
    mapper.load_state_dict(state["state_dict"])
    mapper.to(device).eval()
    return mapper, decoder, args, ckpt


@torch.no_grad()
def eval_original(model, csi, device, batch_size):
    loader = DataLoader(TensorDataset(csi), batch_size=batch_size, shuffle=False)
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    mse_sum = 0.0
    n_total = 0
    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x)
        err, power = evaluator(pred, x)
        total_error += err
        total_power += power
        mse_sum += float((pred - x).pow(2).mean().cpu()) * x.size(0)
        n_total += x.size(0)
    return {
        "mse": mse_sum / max(n_total, 1),
        "nmse": float(nmse_from_sums(total_error, total_power).cpu()),
        "n": n_total,
    }


@torch.no_grad()
def eval_adapter(mapper, decoder, source_code, csi, device, batch_size):
    loader = DataLoader(
        TensorDataset(source_code, csi),
        batch_size=batch_size,
        shuffle=False)
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    mse_sum = 0.0
    n_total = 0
    for source, x in loader:
        source = source.to(device, non_blocking=True)
        x = x.to(device, non_blocking=True)
        pred = decoder(mapper(source))
        err, power = evaluator(pred, x)
        total_error += err
        total_power += power
        mse_sum += float((pred - x).pow(2).mean().cpu()) * x.size(0)
        n_total += x.size(0)
    return {
        "mse": mse_sum / max(n_total, 1),
        "nmse": float(nmse_from_sums(total_error, total_power).cpu()),
        "n": n_total,
    }


def evaluate_original_exp(exp_dir, csi_cache, device, batch_size):
    model, args, checkpoint = build_original_model(exp_dir, device)
    row = {
        "kind": "original",
        "name": Path(exp_dir).name,
        "path": str(exp_dir),
        "checkpoint": str(checkpoint),
        "encoder": args.get("encoder"),
        "decoder": args.get("decoder"),
        "hidden_dim": None,
        "trainable_params": count_params(model),
    }
    for split in ("train", "val", "test"):
        csi = csi_cache[(args[f"{split}_path"], args.get("channel", 2),
                         args.get("nt", 32), args.get("nc", 32))]
        metrics = eval_original(model, csi, device, batch_size)
        row[f"{split}_nmse"] = metrics["nmse"]
        row[f"{split}_mse"] = metrics["mse"]
        row[f"{split}_n"] = metrics["n"]
    return row


def evaluate_adapter_exp(exp_dir, csi_cache, code_cache, device, batch_size):
    mapper, decoder, args, checkpoint = build_adapter_model(exp_dir, device)
    row = {
        "kind": "adapter",
        "name": Path(exp_dir).name,
        "path": str(exp_dir),
        "checkpoint": str(checkpoint),
        "encoder": "source_code",
        "decoder": args.get("decoder"),
        "mapper_type": args.get("mapper_type"),
        "hidden_dim": args.get("hidden_dim"),
        "lowrank_rank": args.get("lowrank_rank"),
        "lambda_code": args.get("lambda_code"),
        "lambda_recon": args.get("lambda_recon"),
        "code_loss_type": args.get("code_loss_type"),
        "residual_scale": args.get("residual_scale"),
        "learnable_residual_gate": args.get("learnable_residual_gate"),
        "gate_max": args.get("gate_max"),
        "align_ridge": args.get("align_ridge"),
        "block_norm": not args.get("no_block_norm", False),
        "train_affine": args.get("train_affine"),
        "trainable_params": count_params(mapper),
    }
    ns = argparse.Namespace(**args)
    for split in ("train", "val", "test"):
        csi_key = (args[f"{split}_csi"], args.get("channel", 2),
                   args.get("nt", 32), args.get("nc", 32))
        code_path = split_paths(ns, "source", split)
        metrics = eval_adapter(
            mapper,
            decoder,
            code_cache[code_path],
            csi_cache[csi_key],
            device,
            batch_size)
        row[f"{split}_nmse"] = metrics["nmse"]
        row[f"{split}_mse"] = metrics["mse"]
        row[f"{split}_n"] = metrics["n"]
    return row


def collect_experiments(adapter_root):
    adapter_dirs = [
        path.parent
        for path in sorted(Path(adapter_root).rglob("args.json"))
        if (path.parent / "metrics.json").exists()
    ]
    original_dirs = {}
    for exp_dir in adapter_dirs:
        args = load_json(exp_dir / "args.json") or {}
        for key in ("source_exp", "target_exp"):
            value = args.get(key)
            if value and (Path(value) / "checkpoints" / "best_nmse.pth").exists():
                original_dirs[value] = Path(value)
    return sorted(original_dirs.values()), adapter_dirs


def write_outputs(rows, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: (
        row.get("test_nmse", math.inf),
        row.get("trainable_params", math.inf)))
    lines = ["# CSI NMSE Comparison", ""]
    lines.append("| rank | kind | test_nmse | val_nmse | train_nmse | params_M | hidden | name |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for idx, row in enumerate(ranked, 1):
        params_m = row.get("trainable_params", 0) / 1e6
        hidden = row.get("hidden_dim")
        lines.append(
            f"| {idx} | {row.get('kind')} | {row.get('test_nmse', math.nan):.3f} | "
            f"{row.get('val_nmse', math.nan):.3f} | "
            f"{row.get('train_nmse', math.nan):.3f} | "
            f"{params_m:.3f} | {hidden if hidden is not None else '-'} | "
            f"`{row.get('name')}` |")
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter_root",
        default="adapter/exps/affine_residual_mlp/seed1014/transnet")
    parser.add_argument(
        "--output_prefix",
        default="adapter/exps/csi_nmse_comparison")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--interval_seconds", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.wait:
        while process_running():
            print(f"adapter training is still running; sleep {args.interval_seconds}s", flush=True)
            time.sleep(args.interval_seconds)
    if args.gpu is not None and not args.cpu:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    original_dirs, adapter_dirs = collect_experiments(args.adapter_root)
    csi_cache = {}
    code_cache = {}
    for exp_dir in original_dirs:
        original_args = load_json(exp_dir / "args.json") or {}
        for split in ("train", "val", "test"):
            key = (
                original_args[f"{split}_path"],
                original_args.get("channel", 2),
                original_args.get("nt", 32),
                original_args.get("nc", 32),
            )
            if key not in csi_cache:
                csi_cache[key] = load_csi(*key, max_samples=args.max_samples)
    for exp_dir in adapter_dirs:
        adapter_args = load_json(exp_dir / "args.json") or {}
        ns = argparse.Namespace(**adapter_args)
        for split in ("train", "val", "test"):
            csi_key = (
                adapter_args[f"{split}_csi"],
                adapter_args.get("channel", 2),
                adapter_args.get("nt", 32),
                adapter_args.get("nc", 32),
            )
            if csi_key not in csi_cache:
                csi_cache[csi_key] = load_csi(*csi_key, max_samples=args.max_samples)
            code_path = split_paths(ns, "source", split)
            if code_path not in code_cache:
                code = torch.load(
                    code_path, weights_only=True, map_location="cpu").float()
                if args.max_samples and code.size(0) > args.max_samples:
                    code = code[:args.max_samples].contiguous()
                code_cache[code_path] = code

    rows = []
    for exp_dir in original_dirs:
        print(f"evaluating original: {exp_dir}", flush=True)
        rows.append(evaluate_original_exp(exp_dir, csi_cache, device, args.batch_size))
    for exp_dir in adapter_dirs:
        print(f"evaluating adapter: {exp_dir}", flush=True)
        rows.append(evaluate_adapter_exp(exp_dir, csi_cache, code_cache, device, args.batch_size))
    paths = write_outputs(rows, args.output_prefix)
    print("saved:", " ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
