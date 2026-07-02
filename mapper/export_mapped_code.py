#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

MAPPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAPPER_DIR))

from models import build_mapper


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--source_code", default=None)
    parser.add_argument("--output", action="append", default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    cfg = json.loads((exp_dir / "args.json").read_text())
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        exp_dir / "checkpoints" / "best_mse.pth")
    if not checkpoint.exists():
        fallback = exp_dir / "checkpoints" / "best_mapper.pth"
        if fallback.exists():
            checkpoint = fallback
        else:
            raise FileNotFoundError(checkpoint)

    source_code = args.source_code or cfg["source_code"]
    source = torch.load(
        source_code,
        weights_only=True,
        map_location=torch.device("cpu")).float()
    code_dim = source.size(1)
    model = build_mapper(
        cfg["mapper"],
        code_dim,
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
        flow_hidden_dim=cfg["flow_hidden_dim"],
        flow_blocks=cfg["flow_blocks"],
        clamp=cfg["flow_clamp"],
        dropout=cfg["dropout"])
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
    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            outputs.append(model(batch).cpu())
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
    print(f"device={device}")


if __name__ == "__main__":
    main()
