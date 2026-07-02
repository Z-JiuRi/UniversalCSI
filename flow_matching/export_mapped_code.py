#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

FLOW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FLOW_DIR))

from models import FlowMatchingTranslator


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    parser.add_argument("--output", action="append", default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ode_steps", type=int, default=None)
    parser.add_argument("--ode_method", default=None,
                        choices=["euler", "heun"])
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
    source = torch.load(
        source_code,
        weights_only=True,
        map_location=torch.device("cpu")).float()
    model = build_model(cfg, source.size(1))
    ckpt = torch.load(
        checkpoint,
        weights_only=True,
        map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"])
    device = resolve_device(args.gpu, args.cpu)
    model.to(device).eval()

    loader = DataLoader(
        source,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")
    ode_steps = args.ode_steps or cfg["ode_steps"]
    ode_method = args.ode_method or cfg["ode_method"]
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            outputs.append(model.sample(
                batch,
                steps=ode_steps,
                method=ode_method).cpu())
    outputs = torch.cat(outputs, dim=0)

    output_paths = args.output or [
        str(exp_dir / "codewords" / "mapped_code.pt"),
        str(exp_dir / "mapped_code.pt"),
    ]
    for path in output_paths:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, output_path)
        print(f"saved={output_path} shape={tuple(outputs.shape)}")
    print(f"checkpoint={checkpoint}")
    print(f"epoch={ckpt.get('epoch')} best={ckpt.get('best')}")
    print(f"ode_steps={ode_steps} ode_method={ode_method} device={device}")


if __name__ == "__main__":
    main()

