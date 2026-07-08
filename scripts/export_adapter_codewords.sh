#!/bin/bash

# 重新导出现有 adapter 实验的 train codeword，并覆盖原 codewords/train_code.pt。
#
# 用途：
# - 旧版导出逻辑保存的是 model.encoder(x)，也就是 adapter 前 code。
# - 现在需要保存 model.encode(x)，也就是 encoder -> adapter 后、decoder 实际接收的 code。
# - 本脚本不重新训练，不调用 main.py，不覆盖原 args.json/run.log。
# - 脚本会读取每个 adapter 实验的 args.json，加载 checkpoints/best_nmse.pth，
#   然后把 adapter 后 code 覆盖写入该实验目录下的 codewords/train_code.pt。
#
# 默认导出全部 adapter 实验：
#   gpu=0 bash scripts/export_adapter_codewords.sh
#
# 指定根目录：
#   root=exps/COST2100/in/teacher_code_adapter gpu=0 \
#     bash scripts/export_adapter_codewords.sh
#
# 只导出某一个实验目录：
#   exp_dir=exps/COST2100/in/teacher_code_adapter/gated_lowrank_affine_mlp/example \
#     gpu=0 bash scripts/export_adapter_codewords.sh
#
# 使用 CPU：
#   cpu=1 bash scripts/export_adapter_codewords.sh
#
# 调小 batch size 或 workers：
#   batch_size=128 workers=0 gpu=0 bash scripts/export_adapter_codewords.sh
#
# 先 dry run 看会处理哪些实验：
#   dry_run=1 bash scripts/export_adapter_codewords.sh

set -euo pipefail

root=${root:-exps/COST2100/in/teacher_code_adapter}
exp_dir=${exp_dir:-}
gpu=${gpu:-0}
cpu=${cpu:-0}
batch_size=${batch_size:-}
workers=${workers:-0}
dry_run=${dry_run:-0}

python - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models import universal_csi


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def clean_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True,
                            map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    for key in list(state_dict.keys()):
        if key.endswith("total_ops") or key.endswith("total_params"):
            del state_dict[key]
    return state_dict


def load_train_tensor(path, channel, nt, nc):
    data = torch.load(path, weights_only=True,
                      map_location=torch.device("cpu")).to(torch.float32)
    expected_shape = (channel, nt, nc)
    if data.ndim == 2:
        data = data.view(-1, *expected_shape)
    if data.ndim != 4 or tuple(data.shape[1:]) != expected_shape:
        raise ValueError(
            f"{path} should have shape (N, {channel}, {nt}, {nc}), "
            f"got {tuple(data.shape)}")
    return data


def build_model(args):
    return universal_csi(
        encoder_name=args.get("encoder", "transnet"),
        decoder_name=args.get("decoder", "transnet"),
        reduction=args.get("cr", 4),
        d_model=args.get("d_model", 64),
        channel=args.get("channel", 2),
        nt=args.get("nt", 32),
        nc=args.get("nc", 32),
        dim_feedforward=args.get("dim_feedforward"),
        hidden=args.get("hidden", 16),
        num_blocks=args.get("num_blocks", 2),
        adapter=args.get("adapter"),
        adapter_hidden_dim=args.get("adapter_hidden_dim"),
        adapter_rank=args.get("adapter_rank", 32),
        adapter_gate_init=args.get("adapter_gate_init", 0.1),
    )


def discover_experiments():
    exp_dir = os.environ.get("exp_dir", "")
    if exp_dir:
        paths = [Path(exp_dir) / "args.json"]
    else:
        root = Path(os.environ.get(
            "root", "exps/COST2100/in/teacher_code_adapter"))
        paths = sorted(root.rglob("args.json"))

    experiments = []
    for args_path in paths:
        if not args_path.exists():
            raise FileNotFoundError(f"missing args.json: {args_path}")
        with open(args_path) as f:
            args = json.load(f)
        if not args.get("adapter"):
            continue
        exp = args_path.parent
        ckpt = exp / "checkpoints" / "best_nmse.pth"
        if not ckpt.exists():
            print(f"[skip] missing checkpoint: {ckpt}", file=sys.stderr)
            continue
        experiments.append((exp, args, ckpt))
    return experiments


def export_one(exp, args, ckpt, device, batch_size, workers, dry_run):
    output_path = exp / "codewords" / "train_code.pt"
    print(f"[export] {exp}")
    print(f"         checkpoint: {ckpt}")
    print(f"         output:     {output_path}")
    if dry_run:
        return

    channel = args.get("channel", 2)
    nt = args.get("nt", 32)
    nc = args.get("nc", 32)
    train_path = args["train_path"]
    data = load_train_tensor(train_path, channel, nt, nc)
    indices = torch.arange(data.size(0), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(data, indices),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    model = build_model(args)
    model.load_state_dict(clean_state_dict(ckpt))
    model.to(device)
    model.eval()

    codewords = []
    sample_indices = []
    with torch.no_grad():
        for sparse_gt, batch_indices in loader:
            sparse_gt = sparse_gt.to(device, non_blocking=True)
            code = model.encode(sparse_gt)
            codewords.append(code.cpu())
            sample_indices.append(batch_indices.cpu())

    codewords = torch.cat(codewords, dim=0)
    sample_indices = torch.cat(sample_indices, dim=0).to(torch.long)
    if not torch.equal(sample_indices, torch.arange(sample_indices.numel())):
        aligned = torch.empty_like(codewords)
        aligned[sample_indices] = codewords
        codewords = aligned

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codewords, output_path)
    print(f"         saved shape: {tuple(codewords.shape)}")


def main():
    dry_run = env_bool("dry_run", False)
    use_cpu = env_bool("cpu", False)
    gpu = int(os.environ.get("gpu", "0"))
    if not use_cpu and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    experiments = discover_experiments()
    print(f"Found {len(experiments)} adapter experiment(s).")
    if not experiments:
        return

    batch_size_env = os.environ.get("batch_size", "")
    workers = int(os.environ.get("workers", "0"))

    for exp, args, ckpt in experiments:
        batch_size = int(batch_size_env or args.get("batch_size", 256))
        export_one(exp, args, ckpt, device, batch_size, workers, dry_run)


if __name__ == "__main__":
    main()
PY
