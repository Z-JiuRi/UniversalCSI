#!/usr/bin/env python
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CodewordPairDataset
from models import build_mapper, count_parameters


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def cosine_mean(pred, target):
    return F.cosine_similarity(pred, target, dim=1).mean()


def offdiag_cov_loss(z):
    z = z - z.mean(dim=0, keepdim=True)
    denom = max(z.size(0) - 1, 1)
    cov = z.t().matmul(z) / denom
    diag = cov.diag()
    offdiag = cov - torch.diag_embed(diag)
    return offdiag.pow(2).mean()


def build_optimizer(model, lr, weight_decay):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


def run_epoch(model, loader, device, optimizer=None, lambda_cos=0.0,
              lambda_cov=0.0):
    train = optimizer is not None
    model.train(train)
    total = {
        "loss": 0.0,
        "mse": 0.0,
        "cos": 0.0,
        "nmse": 0.0,
        "n": 0,
    }
    for source, target, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(source)
        mse = F.mse_loss(pred, target)
        loss = mse
        if lambda_cos:
            loss = loss + lambda_cos * (
                1.0 - F.cosine_similarity(pred, target, dim=1).mean())
        if lambda_cov:
            loss = loss + lambda_cov * offdiag_cov_loss(pred - target)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_n = source.size(0)
        total["loss"] += float(loss.detach().cpu()) * batch_n
        total["mse"] += float(mse.detach().cpu()) * batch_n
        total["cos"] += float(cosine_mean(pred, target).detach().cpu()) * batch_n
        total["nmse"] += float(nmse_db(pred, target).detach().cpu()) * batch_n
        total["n"] += batch_n
    return {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}


def save_outputs(model, loader, device, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    outs = []
    with torch.no_grad():
        for source, _, _ in loader:
            source = source.to(device, non_blocking=True)
            outs.append(model(source).cpu())
    torch.save(torch.cat(outs, dim=0), output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_code", required=True)
    parser.add_argument("--target_code", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--mapper", default="flow",
                        choices=["identity", "mlp", "deep_mlp",
                                 "residual_mlp", "flow", "coupling_flow",
                                 "hybrid", "hybrid_flow_mlp"])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=1024)
    parser.add_argument("--flow_blocks", type=int, default=8)
    parser.add_argument("--flow_clamp", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--lambda_cos", type=float, default=0.0)
    parser.add_argument("--lambda_cov", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    set_seed(args.seed)
    device = resolve_device(args.gpu, args.cpu)
    train_set = CodewordPairDataset(
        args.source_code,
        args.target_code,
        split="train",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
    val_set = CodewordPairDataset(
        args.source_code,
        args.target_code,
        split="val",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
    all_set = CodewordPairDataset(
        args.source_code,
        args.target_code,
        split="all",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
    code_dim = train_set.source.size(1)
    model = build_mapper(
        args.mapper,
        code_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_blocks=args.flow_blocks,
        clamp=args.flow_clamp,
        dropout=args.dropout).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")
    all_loader = DataLoader(
        all_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")

    print(f"device={device}")
    print(f"mapper={args.mapper}, params={count_parameters(model)}")
    print(f"train={len(train_set)}, val={len(val_set)}, code_dim={code_dim}")
    best = {"mse": math.inf, "epoch": 0}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            lambda_cos=args.lambda_cos,
            lambda_cov=args.lambda_cov)
        val_metrics = run_epoch(model, val_loader, device)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(
            f"epoch={epoch:04d} "
            f"train_mse={train_metrics['mse']:.6e} "
            f"val_mse={val_metrics['mse']:.6e} "
            f"val_cos={val_metrics['cos']:.6f} "
            f"val_nmse={val_metrics['nmse']:.3f}dB")
        if val_metrics["mse"] < best["mse"]:
            best = {"mse": val_metrics["mse"], "epoch": epoch}
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best": best,
            }, exp_dir / "best_mapper.pth")

    (exp_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    if (exp_dir / "best_mapper.pth").exists():
        ckpt = torch.load(exp_dir / "best_mapper.pth",
                          weights_only=True,
                          map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    final_metrics = run_epoch(model, all_loader, device)
    (exp_dir / "metrics.json").write_text(
        json.dumps({"best": best, "all": final_metrics}, indent=2),
        encoding="utf-8")
    save_outputs(model, all_loader, device, exp_dir / "mapped_code.pt")
    print(f"best_epoch={best['epoch']} best_val_mse={best['mse']:.6e}")
    print(f"all_mse={final_metrics['mse']:.6e} "
          f"all_cos={final_metrics['cos']:.6f} "
          f"all_nmse={final_metrics['nmse']:.3f}dB")


if __name__ == "__main__":
    main()
