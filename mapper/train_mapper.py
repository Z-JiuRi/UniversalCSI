#!/usr/bin/env python
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import importlib.util

_logger_spec = importlib.util.spec_from_file_location(
    "mapper_project_logger",
    ROOT / "utils" / "logger.py")
_logger_module = importlib.util.module_from_spec(_logger_spec)
_logger_spec.loader.exec_module(_logger_module)
logger = _logger_module.logger
setup_logging = _logger_module.setup_logging

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


def sample_tail_mse_loss(pred, target, ratio):
    if ratio <= 0:
        return pred.new_tensor(0.0)
    per_sample = (pred - target).pow(2).mean(dim=1)
    k = max(1, int(math.ceil(per_sample.numel() * ratio)))
    k = min(k, per_sample.numel())
    return torch.topk(per_sample, k=k, largest=True).values.mean()


def dim_tail_mse_loss(pred, target, ratio):
    if ratio <= 0:
        return pred.new_tensor(0.0)
    per_dim = (pred - target).pow(2).mean(dim=0)
    k = max(1, int(math.ceil(per_dim.numel() * ratio)))
    k = min(k, per_dim.numel())
    return torch.topk(per_dim, k=k, largest=True).values.mean()


def fit_teacher_whiten_stats(target_codes, eps_ratio):
    target = target_codes.to(torch.float64)
    mean = target.mean(dim=0, keepdim=True)
    centered = target - mean
    cov = centered.t().matmul(centered) / max(centered.size(0) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0).to(torch.float32)
    eigvecs = eigvecs[:, order].to(torch.float32)
    eps = eigvals.mean().clamp_min(1e-12) * eps_ratio
    inv = 1.0 / (eigvals + eps)
    # Normalize the average weight to 1 so lambda_whiten is interpretable.
    inv = inv / inv.mean().clamp_min(1e-12)
    return eigvecs, inv


def whiten_pair_loss(pred, target, eigvecs, inv_eig):
    diff = pred - target
    proj = diff.matmul(eigvecs)
    return (proj.pow(2) * inv_eig).mean()


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
              lambda_cov=0.0, lambda_smoothl1=0.0, smoothl1_beta=0.05,
              lambda_sample_tail=0.0, sample_tail_ratio=0.2,
              lambda_dim_tail=0.0, dim_tail_ratio=0.05,
              lambda_whiten=0.0, whiten_stats=None):
    train = optimizer is not None
    model.train(train)
    total = {
        "loss": 0.0,
        "mse": 0.0,
        "smoothl1": 0.0,
        "sample_tail": 0.0,
        "dim_tail": 0.0,
        "whiten": 0.0,
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
        smoothl1 = pred.new_tensor(0.0)
        sample_tail = pred.new_tensor(0.0)
        dim_tail = pred.new_tensor(0.0)
        whiten = pred.new_tensor(0.0)
        if lambda_smoothl1:
            smoothl1 = F.smooth_l1_loss(pred, target, beta=smoothl1_beta)
            loss = loss + lambda_smoothl1 * smoothl1
        if lambda_sample_tail:
            sample_tail = sample_tail_mse_loss(
                pred, target, sample_tail_ratio)
            loss = loss + lambda_sample_tail * sample_tail
        if lambda_dim_tail:
            dim_tail = dim_tail_mse_loss(pred, target, dim_tail_ratio)
            loss = loss + lambda_dim_tail * dim_tail
        if lambda_whiten:
            if whiten_stats is None:
                raise ValueError("lambda_whiten requires whiten_stats")
            eigvecs, inv_eig = whiten_stats
            whiten = whiten_pair_loss(pred, target, eigvecs, inv_eig)
            loss = loss + lambda_whiten * whiten
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
        total["smoothl1"] += float(smoothl1.detach().cpu()) * batch_n
        total["sample_tail"] += float(sample_tail.detach().cpu()) * batch_n
        total["dim_tail"] += float(dim_tail.detach().cpu()) * batch_n
        total["whiten"] += float(whiten.detach().cpu()) * batch_n
        total["cos"] += float(cosine_mean(pred, target).detach().cpu()) * batch_n
        total["nmse"] += float(nmse_db(pred, target).detach().cpu()) * batch_n
        total["n"] += batch_n
    return {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}


def save_outputs(model, loader, device, output_paths):
    if isinstance(output_paths, (str, Path)):
        output_paths = [output_paths]
    output_paths = [Path(path) for path in output_paths]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    outs = []
    with torch.no_grad():
        for source, _, _ in loader:
            source = source.to(device, non_blocking=True)
            outs.append(model(source).cpu())
    outputs = torch.cat(outs, dim=0)
    for output_path in output_paths:
        torch.save(outputs, output_path)


def log_metrics_to_tensorboard(writer, prefix, metrics, epoch):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, global_step=epoch)


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
    parser.add_argument("--lambda_smoothl1", type=float, default=0.0)
    parser.add_argument("--smoothl1_beta", type=float, default=0.05)
    parser.add_argument("--lambda_sample_tail", type=float, default=0.0)
    parser.add_argument("--sample_tail_ratio", type=float, default=0.2)
    parser.add_argument("--lambda_dim_tail", type=float, default=0.0)
    parser.add_argument("--dim_tail_ratio", type=float, default=0.05)
    parser.add_argument("--lambda_whiten", type=float, default=0.0)
    parser.add_argument("--whiten_eps_ratio", type=float, default=1e-3)
    parser.add_argument("--save_last", action="store_true",
                        help="save the last epoch instead of selecting by val MSE")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    codeword_dir = exp_dir / "codewords"
    tensorboard_dir = exp_dir / "tensorboard"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    codeword_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    with (exp_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    logger.info(f"=> Experiment directory: {exp_dir}")
    logger.info(f"=> Checkpoint directory: {checkpoint_dir}")
    logger.info(f"=> Codeword directory: {codeword_dir}")
    logger.info(f"=> TensorBoard directory: {tensorboard_dir}")

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
    whiten_stats = None
    if args.lambda_whiten:
        eigvecs, inv_eig = fit_teacher_whiten_stats(
            all_set.target,
            eps_ratio=args.whiten_eps_ratio)
        whiten_stats = (eigvecs.to(device), inv_eig.to(device))

    logger.info(f"device={device}")
    logger.info(f"mapper={args.mapper}, params={count_parameters(model)}")
    logger.info(f"train={len(train_set)}, val={len(val_set)}, code_dim={code_dim}")
    logger.info(
        "loss="
        f"mse + smoothl1*{args.lambda_smoothl1} "
        f"+ sample_tail*{args.lambda_sample_tail} "
        f"+ dim_tail*{args.lambda_dim_tail} "
        f"+ whiten*{args.lambda_whiten} "
        f"+ cos*{args.lambda_cos} + cov*{args.lambda_cov}")
    best = {"mse": math.inf, "epoch": 0}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            lambda_cos=args.lambda_cos,
            lambda_cov=args.lambda_cov,
            lambda_smoothl1=args.lambda_smoothl1,
            smoothl1_beta=args.smoothl1_beta,
            lambda_sample_tail=args.lambda_sample_tail,
            sample_tail_ratio=args.sample_tail_ratio,
            lambda_dim_tail=args.lambda_dim_tail,
            dim_tail_ratio=args.dim_tail_ratio,
            lambda_whiten=args.lambda_whiten,
            whiten_stats=whiten_stats)
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            lambda_smoothl1=args.lambda_smoothl1,
            smoothl1_beta=args.smoothl1_beta,
            lambda_sample_tail=args.lambda_sample_tail,
            sample_tail_ratio=args.sample_tail_ratio,
            lambda_dim_tail=args.lambda_dim_tail,
            dim_tail_ratio=args.dim_tail_ratio,
            lambda_whiten=args.lambda_whiten,
            whiten_stats=whiten_stats)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        log_metrics_to_tensorboard(writer, "train", train_metrics, epoch)
        log_metrics_to_tensorboard(writer, "val", val_metrics, epoch)
        logger.info(
            f"epoch={epoch:04d} "
            f"train_loss={train_metrics['loss']:.6e} "
            f"train_mse={train_metrics['mse']:.6e} "
            f"val_mse={val_metrics['mse']:.6e} "
            f"val_cos={val_metrics['cos']:.6f} "
            f"val_nmse={val_metrics['nmse']:.3f}dB")
        save_last = args.save_last or args.val_ratio <= 0
        save_metric = train_metrics["mse"] if save_last else val_metrics["mse"]
        if save_last or save_metric < best["mse"]:
            best = {"mse": save_metric, "epoch": epoch}
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best": best,
            }, checkpoint_dir / "best_mapper.pth")

    (exp_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    if (checkpoint_dir / "best_mapper.pth").exists():
        ckpt = torch.load(checkpoint_dir / "best_mapper.pth",
                          weights_only=True,
                          map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    final_metrics = run_epoch(
        model,
        all_loader,
        device,
        lambda_smoothl1=args.lambda_smoothl1,
        smoothl1_beta=args.smoothl1_beta,
        lambda_sample_tail=args.lambda_sample_tail,
        sample_tail_ratio=args.sample_tail_ratio,
        lambda_dim_tail=args.lambda_dim_tail,
        dim_tail_ratio=args.dim_tail_ratio,
        lambda_whiten=args.lambda_whiten,
        whiten_stats=whiten_stats)
    log_metrics_to_tensorboard(writer, "all", final_metrics, args.epochs)
    (exp_dir / "metrics.json").write_text(
        json.dumps({"best": best, "all": final_metrics}, indent=2),
        encoding="utf-8")
    # Keep the old location for compatibility with existing analysis scripts.
    save_outputs(model, all_loader, device, [
        codeword_dir / "mapped_code.pt",
        exp_dir / "mapped_code.pt",
    ])
    writer.flush()
    writer.close()
    logger.info(f"best_epoch={best['epoch']} best_mse={best['mse']:.6e}")
    logger.info(f"all_mse={final_metrics['mse']:.6e} "
                f"all_cos={final_metrics['cos']:.6f} "
                f"all_nmse={final_metrics['nmse']:.3f}dB")


if __name__ == "__main__":
    main()
