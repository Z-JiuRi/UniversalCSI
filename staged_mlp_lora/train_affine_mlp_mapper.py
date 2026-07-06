#!/usr/bin/env python
import argparse
import importlib.util
import json
import math
import os
import sys
import uuid
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard.writer import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

_logger_spec = importlib.util.spec_from_file_location(
    "staged_mlp_lora_logger",
    ROOT / "utils" / "logger.py")
_logger_module = importlib.util.module_from_spec(_logger_spec)
_logger_spec.loader.exec_module(_logger_module)
logger = _logger_module.logger
setup_logging = _logger_module.setup_logging
log_experiment_header = _logger_module.log_experiment_header
log_parameter_table = _logger_module.log_parameter_table

_scheduler_spec = importlib.util.spec_from_file_location(
    "staged_mlp_lora_scheduler",
    ROOT / "utils" / "scheduler.py")
_scheduler_module = importlib.util.module_from_spec(_scheduler_spec)
_scheduler_spec.loader.exec_module(_scheduler_module)
FakeLR = _scheduler_module.FakeLR
WarmUpCosineAnnealingLR = _scheduler_module.WarmUpCosineAnnealingLR


class CodeDataset(Dataset):
    def __init__(self, source_path, target_path, split="train",
                 val_ratio=0.0, max_samples=0):
        source = torch.load(source_path, weights_only=True,
                            map_location="cpu").float()
        target = torch.load(target_path, weights_only=True,
                            map_location="cpu").float()
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source and target codewords must be 2D")
        if source.shape != target.shape:
            raise ValueError(
                f"source/target shape mismatch: {source.shape} vs "
                f"{target.shape}")
        if max_samples and source.size(0) > max_samples:
            source = source[:max_samples].contiguous()
            target = target[:max_samples].contiguous()
        n = source.size(0)
        if val_ratio <= 0:
            sl = slice(0, n)
        else:
            n_val = int(round(n * val_ratio))
            n_val = max(1, min(n_val, n - 1)) if n > 1 else 0
            if split == "train":
                sl = slice(0, n - n_val)
            elif split in ("val", "test"):
                sl = slice(n - n_val, n)
            elif split == "all":
                sl = slice(0, n)
            else:
                raise ValueError(f"Unknown split: {split}")
        self.source = source[sl].contiguous()
        self.target = target[sl].contiguous()
        self.indices = torch.arange(n, dtype=torch.long)[sl].contiguous()

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx], self.indices[idx]


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0,
                 residual_scale=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.residual_scale = residual_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[3].weight)
        nn.init.zeros_(self.net[3].bias)

    def forward(self, x):
        return x + self.residual_scale * self.net(self.norm(x))


class AffineResidualMLPMapper(nn.Module):
    def __init__(self, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=1.0, use_final_norm=True):
        super().__init__()
        dim = weight.size(0)
        self.register_buffer("alignment_weight", weight)
        self.register_buffer("alignment_bias", bias)
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, hidden_dim, dropout, residual_scale)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(dim) if use_final_norm else nn.Identity()

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def forward(self, source):
        x = self.start(source)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_affine(source, target, ridge=1e-4):
    dim = source.size(1)
    src = source.to(torch.float64)
    tgt = target.to(torch.float64)
    ones = torch.ones(src.size(0), 1, dtype=src.dtype)
    aug = torch.cat([src, ones], dim=1)
    reg = ridge * torch.eye(dim + 1, dtype=src.dtype)
    reg[-1, -1] = 0.0
    lhs = aug.t().matmul(aug) + reg
    rhs = aug.t().matmul(tgt)
    solution = torch.linalg.solve(lhs, rhs)
    return solution[:-1].float(), solution[-1].float()


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def build_scheduler(optimizer, name, epochs, steps_per_epoch, eta_min):
    if name == "const":
        return FakeLR(optimizer=optimizer)
    if name == "cosine":
        total_steps = epochs * steps_per_epoch
        return WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            T_warmup=0.1 * total_steps,
            eta_min=eta_min)
    raise ValueError(f"Unknown scheduler: {name}")


def code_nmse_db(pred, target):
    err = (pred - target).pow(2).sum()
    power = target.pow(2).sum().clamp_min(1e-12)
    return 10.0 * torch.log10(err / power)


def cosine_mean(pred, target):
    return F.cosine_similarity(pred, target, dim=1).mean()


def load_main_models_package():
    package_name = f"main_project_models_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "models" / "__init__.py",
        submodule_search_locations=[str(ROOT / "models")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def clean_state_dict(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)
    for key in list(state_dict.keys()):
        if key.endswith("total_ops") or key.endswith("total_params"):
            del state_dict[key]
    return state_dict


def load_decoder(args, device):
    cfg = json.loads(Path(args.decoder_args_json).read_text())
    main_models = load_main_models_package()
    decoder_name = cfg.get("decoder", "transnet")
    cr = cfg.get("cr", 4)
    d_model = cfg.get("d_model", 64)
    channel = cfg.get("channel", 2)
    nt = cfg.get("nt", 32)
    nc = cfg.get("nc", 32)
    dim_feedforward = cfg.get("dim_feedforward", 2048)
    hidden = cfg.get("hidden", 16)
    num_blocks = cfg.get("num_blocks", 2)
    model = main_models.universal_csi(
        encoder_name="transnet",
        decoder_name=decoder_name,
        reduction=cr,
        d_model=d_model,
        channel=channel,
        nt=nt,
        nc=nc,
        dim_feedforward=dim_feedforward,
        hidden=hidden,
        num_blocks=num_blocks)
    state_dict = clean_state_dict(args.decoder_checkpoint)
    decoder_state = {
        key[len("decoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        decoder_state = state_dict
    missing, unexpected = model.decoder.load_state_dict(
        decoder_state,
        strict=False)
    if missing or unexpected:
        raise ValueError(
            f"decoder checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}")
    decoder = model.decoder.to(device).eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder, {"channel": channel, "nt": nt, "nc": nc}


def load_csi(path, channel, nt, nc, max_samples=0):
    data = torch.load(path, weights_only=True,
                      map_location=torch.device("cpu")).float()
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(
            f"{path} should have shape (N,{channel},{nt},{nc}), "
            f"got {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data


@torch.no_grad()
def evaluate_decoder_nmse(model, loader, csi_tensor, decoder, device):
    model.eval()
    decoder.eval()
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    for source, _, indices in loader:
        source = source.to(device, non_blocking=True)
        gt = csi_tensor[indices].to(device, non_blocking=True)
        recon = decoder(model(source))
        mse = F.mse_loss(recon, gt)
        total_error += (recon - gt).pow(2).sum()
        total_power += gt.pow(2).sum()
        total_mse += float(mse.detach().cpu()) * source.size(0)
        total_n += source.size(0)
    nmse = 10.0 * torch.log10(total_error / total_power.clamp_min(1e-12))
    return {
        "decoder_mse": total_mse / max(total_n, 1),
        "decoder_nmse": float(nmse.detach().cpu()),
        "n": total_n,
    }


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "nmse": 0.0, "n": 0}
    for source, target, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(source)
        loss = F.mse_loss(pred, target)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        n = source.size(0)
        total["loss"] += float(loss.detach().cpu()) * n
        total["mse"] += float(loss.detach().cpu()) * n
        total["cos"] += float(cosine_mean(pred, target).detach().cpu()) * n
        total["nmse"] += float(code_nmse_db(pred, target).detach().cpu()) * n
        total["n"] += n
    return {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}


@torch.no_grad()
def save_mapped_code(model, loader, device, paths):
    model.eval()
    outs = []
    for source, _, _ in loader:
        source = source.to(device, non_blocking=True)
        outs.append(model(source).cpu())
    mapped = torch.cat(outs, dim=0)
    for path in paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mapped, path)


def log_metrics(writer, prefix, metrics, epoch):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, global_step=epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_code", required=True)
    parser.add_argument("--target_code", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--source_name", default="source")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--no_final_norm", action="store_true",
                        help="disable the output LayerNorm after residual blocks")
    parser.add_argument("--align_ridge", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine",
                        choices=["const", "cosine"])
    parser.add_argument("--eta_min", type=float, default=5e-5)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--eval_decoder_every", type=int, default=20)
    parser.add_argument("--eval_decoder_max_samples", type=int, default=0)
    parser.add_argument("--decoder_checkpoint",
                        default="exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth")
    parser.add_argument("--decoder_args_json",
                        default="exps/COST2100/in/seed42/transnet_transnet/args.json")
    parser.add_argument("--csi_path",
                        default="/storage/hujiacong/zxd/datasets/cost2100/in_train.pt")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    codeword_dir = exp_dir / "codewords"
    tensorboard_dir = exp_dir / "tensorboard"
    for path in (checkpoint_dir, codeword_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    (exp_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8")

    device = resolve_device(args.gpu, args.cpu)
    set_seed(args.seed)
    log_experiment_header(args, exp_dir=exp_dir, target_logger=logger)

    train_set = CodeDataset(
        args.source_code,
        args.target_code,
        split="train",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
    use_val = args.val_ratio > 0
    val_set = None
    if use_val:
        val_set = CodeDataset(
            args.source_code,
            args.target_code,
            split="val",
            val_ratio=args.val_ratio,
            max_samples=args.max_samples)
    all_set = CodeDataset(
        args.source_code,
        args.target_code,
        split="all",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
    weight, bias = fit_affine(
        train_set.source,
        train_set.target,
        ridge=args.align_ridge)
    model = AffineResidualMLPMapper(
        weight,
        bias,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        use_final_norm=not args.no_final_norm).to(device)
    log_parameter_table(model, logger)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")
    val_loader = None
    if use_val:
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

    decoder = None
    csi_tensor = None
    decoder_eval_loader = None
    if args.eval_decoder_every:
        decoder, decoder_cfg = load_decoder(args, device)
        csi_tensor = load_csi(
            args.csi_path,
            decoder_cfg["channel"],
            decoder_cfg["nt"],
            decoder_cfg["nc"],
            max_samples=args.max_samples)
        decoder_eval_set = all_set
        if args.eval_decoder_max_samples:
            n_eval = min(args.eval_decoder_max_samples, len(all_set))
            decoder_eval_set = Subset(all_set, range(n_eval))
        decoder_eval_loader = DataLoader(
            decoder_eval_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda")
        logger.info(
            "=> Periodic true NMSE eval: every=%d max_samples=%s",
            args.eval_decoder_every,
            args.eval_decoder_max_samples or "full")

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        len(train_loader),
        args.eta_min)

    logger.info("=> Device: %s", device)
    logger.info("=> Source: %s", args.source_code)
    logger.info("=> Target: %s", args.target_code)
    logger.info(
        "=> Dataset: train=%d val=%d all=%d code_dim=%d",
        len(train_set),
        len(val_set) if use_val else 0,
        len(all_set),
        train_set.source.size(1))
    logger.info(
        "=> Affine buffers: weight=%s bias=%s numel=%d",
        tuple(weight.shape),
        tuple(bias.shape),
        weight.numel() + bias.numel())
    logger.info(
        "=> Mapper: hidden_dim=%d num_blocks=%d residual_scale=%s "
        "trainable_params=%s",
        args.hidden_dim,
        args.num_blocks,
        args.residual_scale,
        f"{count_trainable(model):,}")
    logger.info(
        "=> Optimizer: AdamW lr=%s weight_decay=%s scheduler=%s eta_min=%s",
        args.lr,
        args.weight_decay,
        args.scheduler,
        args.eta_min)

    best_mse = {"metric": math.inf, "epoch": 0, "selection": ""}
    best_nmse = {"metric": math.inf, "epoch": 0, "selection": ""}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scheduler=scheduler)
        eval_prefix = "train"
        eval_metrics = train_metrics
        if use_val:
            eval_prefix = "val"
            eval_metrics = run_epoch(model, val_loader, device)

        record = {
            "epoch": epoch,
            "lr": scheduler.get_lr()[0],
            "train": train_metrics,
            eval_prefix: eval_metrics,
        }
        log_metrics(writer, "train", train_metrics, epoch)
        if use_val:
            log_metrics(writer, "val", eval_metrics, epoch)
        writer.add_scalar("train/lr", scheduler.get_lr()[0], epoch)

        decoder_metrics = None
        if args.eval_decoder_every and epoch % args.eval_decoder_every == 0:
            decoder_metrics = evaluate_decoder_nmse(
                model,
                decoder_eval_loader,
                csi_tensor,
                decoder,
                device)
            record["true_decoder_eval"] = decoder_metrics
            log_metrics(writer, "true_decoder_eval", decoder_metrics, epoch)
            logger.info(
                f"Epoch [{epoch}/{args.epochs}] true_decoder_eval "
                f"n={decoder_metrics['n']} "
                f"decoder_mse={decoder_metrics['decoder_mse']:.6e} "
                f"decoder_nmse={decoder_metrics['decoder_nmse']:.3f}dB")
            if decoder_metrics["decoder_nmse"] < best_nmse["metric"]:
                best_nmse = {
                    "metric": decoder_metrics["decoder_nmse"],
                    "decoder_mse": decoder_metrics["decoder_mse"],
                    "epoch": epoch,
                    "selection": "true_decoder_nmse",
                }
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best": best_nmse,
                    "args": vars(args),
                }, checkpoint_dir / "best_nmse.pth")

        if eval_metrics["mse"] < best_mse["metric"]:
            best_mse = {
                "metric": eval_metrics["mse"],
                "mse": eval_metrics["mse"],
                "loss": eval_metrics["loss"],
                "epoch": epoch,
                "selection": f"{eval_prefix}_mse",
            }
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best_mse,
                "args": vars(args),
            }, checkpoint_dir / "best_mse.pth")

        history.append(record)
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"lr={scheduler.get_lr()[0]:.6e} "
            f"train_loss={train_metrics['loss']:.6e} "
            f"{eval_prefix}_mse={eval_metrics['mse']:.6e} "
            f"{eval_prefix}_cos={eval_metrics['cos']:.6f} "
            f"{eval_prefix}_nmse={eval_metrics['nmse']:.3f}dB")

    (exp_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8")

    def load_and_export(tag, ckpt_path, make_default=False):
        ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = run_epoch(model, all_loader, device)
        paths = [codeword_dir / f"mapped_code_{tag}.pt",
                 exp_dir / f"mapped_code_{tag}.pt"]
        if make_default:
            paths.extend([codeword_dir / "mapped_code.pt",
                          exp_dir / "mapped_code.pt"])
        save_mapped_code(model, all_loader, device, paths)
        return metrics

    all_best_mse = None
    all_best_nmse_code = None
    if (checkpoint_dir / "best_mse.pth").exists():
        all_best_mse = load_and_export(
            "best_mse",
            checkpoint_dir / "best_mse.pth",
            make_default=True)
        log_metrics(writer, "all_best_mse", all_best_mse, args.epochs)
    if (checkpoint_dir / "best_nmse.pth").exists():
        all_best_nmse_code = load_and_export(
            "best_nmse",
            checkpoint_dir / "best_nmse.pth",
            make_default=False)
        log_metrics(writer, "all_best_nmse_code",
                    all_best_nmse_code, args.epochs)

    metrics = {
        "best_mse": best_mse,
        "best_nmse": best_nmse,
        "all_best_mse": all_best_mse,
        "all_best_nmse_code": all_best_nmse_code,
    }
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8")
    writer.flush()
    writer.close()
    logger.info(
        f"best_mse_epoch={best_mse['epoch']} "
        f"best_mse={best_mse['metric']:.6e}")
    if best_nmse["epoch"]:
        logger.info(
            f"best_nmse_epoch={best_nmse['epoch']} "
            f"best_nmse={best_nmse['metric']:.3f}dB")
    if all_best_mse is not None:
        logger.info(
            f"all_best_mse_mse={all_best_mse['mse']:.6e} "
            f"all_best_mse_cos={all_best_mse['cos']:.6f} "
            f"all_best_mse_nmse={all_best_mse['nmse']:.3f}dB")


if __name__ == "__main__":
    main()
