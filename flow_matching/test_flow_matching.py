#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

FLOW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FLOW_DIR))

from models import FlowMatchingTranslator


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def nmse_db(pred, target):
    err = (pred - target).pow(2).sum()
    power = target.pow(2).sum().clamp_min(1e-12)
    return 10.0 * torch.log10(err / power)


def build_model(cfg, dim):
    return FlowMatchingTranslator(
        dim,
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
        time_dim=cfg["time_dim"],
        condition=cfg["condition"],
        dropout=cfg["dropout"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--source_code", default=None)
    parser.add_argument("--target_code", default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ode_steps", type=int, default=None)
    parser.add_argument("--ode_method", default=None,
                        choices=["euler", "heun"])
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg = json.loads((exp_dir / "args.json").read_text())
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        exp_dir / "checkpoints" / "best_mse.pth")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source_code = args.source_code or cfg["source_code"]
    target_code = args.target_code or cfg["target_code"]
    source = torch.load(
        source_code,
        weights_only=True,
        map_location=torch.device("cpu")).float()
    target = torch.load(
        target_code,
        weights_only=True,
        map_location=torch.device("cpu")).float()
    max_samples = cfg.get("max_samples", 0)
    if max_samples and source.size(0) > max_samples:
        source = source[:max_samples].contiguous()
        target = target[:max_samples].contiguous()

    model = build_model(cfg, source.size(1))
    ckpt = torch.load(
        checkpoint,
        weights_only=True,
        map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    device = resolve_device(args.gpu, args.cpu)
    model.to(device).eval()
    loader = DataLoader(
        TensorDataset(source, target),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")
    ode_steps = args.ode_steps or cfg["ode_steps"]
    ode_method = args.ode_method or cfg["ode_method"]
    preds = []
    starts = []
    targets = []
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device, non_blocking=True)
            starts.append(model.start(src).cpu())
            preds.append(model.sample(src, steps=ode_steps, method=ode_method).cpu())
            targets.append(tgt)
    pred = torch.cat(preds, dim=0)
    start = torch.cat(starts, dim=0)
    target = torch.cat(targets, dim=0)
    result = {
        "checkpoint": str(checkpoint),
        "epoch": ckpt.get("epoch"),
        "num_samples": int(source.size(0)),
        "code_dim": int(source.size(1)),
        "ode_steps": int(ode_steps),
        "ode_method": ode_method,
        "start_mse": float(F.mse_loss(start, target)),
        "mapped_mse": float(F.mse_loss(pred, target)),
        "mapped_rmse": float(F.mse_loss(pred, target).sqrt()),
        "mapped_cos": float(F.cosine_similarity(pred, target, dim=1).mean()),
        "mapped_nmse": float(nmse_db(pred, target)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

