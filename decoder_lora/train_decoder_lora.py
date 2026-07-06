#!/usr/bin/env python
import argparse
import importlib.util
import json
import math
import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard.writer import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

_logger_spec = importlib.util.spec_from_file_location(
    "decoder_lora_project_logger",
    ROOT / "utils" / "logger.py")
_logger_module = importlib.util.module_from_spec(_logger_spec)
_logger_spec.loader.exec_module(_logger_module)
logger = _logger_module.logger
setup_logging = _logger_module.setup_logging
log_experiment_header = _logger_module.log_experiment_header
log_parameter_table = _logger_module.log_parameter_table

_scheduler_spec = importlib.util.spec_from_file_location(
    "decoder_lora_project_scheduler",
    ROOT / "utils" / "scheduler.py")
_scheduler_module = importlib.util.module_from_spec(_scheduler_spec)
_scheduler_spec.loader.exec_module(_scheduler_module)
FakeLR = _scheduler_module.FakeLR
WarmUpCosineAnnealingLR = _scheduler_module.WarmUpCosineAnnealingLR

from models import (GatedCodeResidualAdapter, count_lora_parameters,  # noqa: E402
                    count_trainable_parameters, inject_decoder_lora,
                    mark_only_lora_trainable)


class DecoderLoraSystem(nn.Module):
    def __init__(self, decoder, code_adapter=None):
        super().__init__()
        self.decoder = decoder
        self.code_adapter = code_adapter if code_adapter is not None else nn.Identity()

    def adapt_code(self, code):
        return self.code_adapter(code)

    def forward(self, code):
        return self.decoder(self.adapt_code(code))


class AlignedCodeCsiDataset(Dataset):
    def __init__(self, source_code_path, target_code_path, csi_path, weight,
                 bias, channel=2, nt=32, nc=32, split="train",
                 val_ratio=0.0, max_samples=0):
        source = torch.load(source_code_path, weights_only=True,
                            map_location="cpu").float()
        target = torch.load(target_code_path, weights_only=True,
                            map_location="cpu").float()
        csi = torch.load(csi_path, weights_only=True,
                         map_location="cpu").float()
        if csi.ndim == 2:
            csi = csi.view(-1, channel, nt, nc)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source/target code must be 2D tensors")
        if source.shape != target.shape:
            raise ValueError(
                f"source and target shape mismatch: {source.shape} vs "
                f"{target.shape}")
        if csi.ndim != 4 or tuple(csi.shape[1:]) != (channel, nt, nc):
            raise ValueError(
                f"CSI should have shape (N,{channel},{nt},{nc}), got "
                f"{tuple(csi.shape)}")
        n = min(source.size(0), target.size(0), csi.size(0))
        source = source[:n].contiguous()
        target = target[:n].contiguous()
        csi = csi[:n].contiguous()
        if max_samples and n > max_samples:
            source = source[:max_samples].contiguous()
            target = target[:max_samples].contiguous()
            csi = csi[:max_samples].contiguous()
            n = max_samples

        z0 = source.matmul(weight) + bias
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
        self.code = z0[sl].contiguous()
        self.target_code = target[sl].contiguous()
        self.csi = csi[sl].contiguous()
        self.indices = torch.arange(n, dtype=torch.long)[sl].contiguous()

    def __len__(self):
        return self.code.size(0)

    def __getitem__(self, idx):
        return (
            self.code[idx],
            self.target_code[idx],
            self.csi[idx],
            self.indices[idx],
        )


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


def load_decoder_from_checkpoint(args, device):
    cfg = {}
    if args.decoder_args_json:
        cfg = json.loads(Path(args.decoder_args_json).read_text())
    main_models = load_main_models_package()
    decoder_name = args.decoder_name or cfg.get("decoder", "transnet")
    cr = args.decoder_cr or cfg.get("cr", 4)
    d_model = args.decoder_d_model or cfg.get("d_model", 64)
    channel = args.decoder_channel or cfg.get("channel", 2)
    nt = args.decoder_nt or cfg.get("nt", 32)
    nc = args.decoder_nc or cfg.get("nc", 32)
    dim_feedforward = (
        args.decoder_dim_feedforward
        or cfg.get("dim_feedforward", 2048))
    hidden = args.decoder_hidden or cfg.get("hidden", 16)
    num_blocks = args.decoder_num_blocks or cfg.get("num_blocks", 2)
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
    decoder = model.decoder.to(device)
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder, {
        "decoder": decoder_name,
        "cr": cr,
        "d_model": d_model,
        "channel": channel,
        "nt": nt,
        "nc": nc,
        "dim_feedforward": dim_feedforward,
    }


def fit_alignment(mode, source, target, ridge=1e-4):
    mode = mode.lower()
    dim = source.size(1)
    if mode in ("identity", "source", "none"):
        return torch.eye(dim), torch.zeros(dim)

    src = source.to(torch.float64)
    tgt = target.to(torch.float64)
    src_mean = src.mean(dim=0, keepdim=True)
    tgt_mean = tgt.mean(dim=0, keepdim=True)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    if mode == "procrustes":
        cross = src_c.t().matmul(tgt_c)
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        weight = u.matmul(vh)
        bias = (tgt_mean - src_mean.matmul(weight)).squeeze(0)
        return weight.float(), bias.float()

    if mode in ("affine", "full_affine"):
        ones = torch.ones(src.size(0), 1, dtype=src.dtype)
        aug = torch.cat([src, ones], dim=1)
        reg = ridge * torch.eye(dim + 1, dtype=src.dtype)
        reg[-1, -1] = 0.0
        lhs = aug.t().matmul(aug) + reg
        rhs = aug.t().matmul(tgt)
        solution = torch.linalg.solve(lhs, rhs)
        return solution[:-1].float(), solution[-1].float()

    raise ValueError(f"Unknown align_mode: {mode}")


def load_code_pair(source_path, target_path, max_samples=0):
    source = torch.load(source_path, weights_only=True,
                        map_location="cpu").float()
    target = torch.load(target_path, weights_only=True,
                        map_location="cpu").float()
    if max_samples and source.size(0) > max_samples:
        source = source[:max_samples].contiguous()
        target = target[:max_samples].contiguous()
    return source, target


def nmse_db_from_sums(error_sum, power_sum):
    return 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))


def code_nmse_db(pred, target):
    err = (pred - target).pow(2).sum()
    power = target.pow(2).sum().clamp_min(1e-12)
    return 10.0 * torch.log10(err / power)


def cosine_mean(pred, target):
    return F.cosine_similarity(pred, target, dim=1).mean()


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


def run_epoch(model, loader, device, optimizer=None, scheduler=None,
              lambda_code=0.0, lambda_delta=0.0, lambda_recT=0.0, lambda_fc=0.0,
              teacher_decoder=None):
    train = optimizer is not None
    model.train(train)
    if teacher_decoder is not None:
        teacher_decoder.eval()
    total = {
        "loss": 0.0,
        "rec": 0.0,
        "code": 0.0,
        "recT": 0.0,
        "fc": 0.0,
        "delta": 0.0,
        "code_mse": 0.0,
        "code_cos": 0.0,
        "code_nmse": 0.0,
        "n": 0,
    }
    for code, target_code, gt, _ in loader:
        code = code.to(device, non_blocking=True)
        target_code = target_code.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        adapted_code = model.adapt_code(code)
        recon = model.decoder(adapted_code)
        rec = F.mse_loss(recon, gt)
        loss = rec
        code_loss = F.mse_loss(adapted_code, target_code)
        delta_loss = F.mse_loss(adapted_code, code)
        recT = code.new_tensor(0.0)
        fc = code.new_tensor(0.0)
        if lambda_code:
            loss = loss + lambda_code * code_loss
        if lambda_delta:
            loss = loss + lambda_delta * delta_loss
        if lambda_recT:
            with torch.no_grad():
                teacher_recon = teacher_decoder(target_code)
            recT = F.mse_loss(recon, teacher_recon)
            loss = loss + lambda_recT * recT
        if lambda_fc:
            fc_pred = model.decoder.fc_decoder(adapted_code)
            with torch.no_grad():
                fc_teacher = teacher_decoder.fc_decoder(target_code)
            fc = F.mse_loss(fc_pred, fc_teacher)
            loss = loss + lambda_fc * fc
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        n = code.size(0)
        total["loss"] += float(loss.detach().cpu()) * n
        total["rec"] += float(rec.detach().cpu()) * n
        total["code"] += float(code_loss.detach().cpu()) * n
        total["recT"] += float(recT.detach().cpu()) * n
        total["fc"] += float(fc.detach().cpu()) * n
        total["delta"] += float(delta_loss.detach().cpu()) * n
        total["code_mse"] += float(code_loss.detach().cpu()) * n
        total["code_cos"] += float(cosine_mean(adapted_code, target_code).detach().cpu()) * n
        total["code_nmse"] += float(code_nmse_db(adapted_code, target_code).detach().cpu()) * n
        total["n"] += n
    return {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}


@torch.no_grad()
def evaluate_decoder(model, loader, device):
    model.eval()
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    for code, _, gt, _ in loader:
        code = code.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        recon = model(code)
        mse = F.mse_loss(recon, gt)
        total_error += (recon - gt).pow(2).sum()
        total_power += gt.pow(2).sum()
        total_mse += float(mse.detach().cpu()) * code.size(0)
        total_n += code.size(0)
    return {
        "decoder_mse": total_mse / max(total_n, 1),
        "decoder_nmse": float(nmse_db_from_sums(
            total_error,
            total_power).detach().cpu()),
        "n": total_n,
    }


def log_metrics(writer, prefix, metrics, epoch):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, global_step=epoch)


def save_lora_state(model, path, epoch, best, args, optimizer, scheduler):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_" in key or key.startswith("code_adapter.")
    }
    torch.save({
        "epoch": epoch,
        "state_dict": state_dict,
        "best": best,
        "args": vars(args),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)


def load_lora_state(model, path, device):
    ckpt = torch.load(path, weights_only=True, map_location=device)
    missing, unexpected = model.load_state_dict(
        ckpt["state_dict"],
        strict=False)
    unexpected = [
        key for key in unexpected
        if "lora_" in key or key.startswith("code_adapter.")
    ]
    if unexpected:
        raise ValueError(f"Unexpected LoRA keys: {unexpected}")
    return ckpt, missing


def build_code_adapter(args, dim, device):
    if args.code_adapter == "none":
        return None
    if args.code_adapter != "gated_lr_mlp":
        raise ValueError(f"Unknown code_adapter: {args.code_adapter}")
    if args.code_lowrank_rank <= 0 and args.code_mlp_hidden <= 0:
        raise ValueError(
            "gated_lr_mlp needs code_lowrank_rank>0 or code_mlp_hidden>0")
    return GatedCodeResidualAdapter(
        dim=dim,
        lowrank_rank=args.code_lowrank_rank,
        mlp_hidden=args.code_mlp_hidden,
        gate_lr_init=args.code_gate_lr,
        gate_mlp_init=args.code_gate_mlp,
        dropout=args.code_adapter_dropout).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_code", required=True)
    parser.add_argument("--target_code", required=True)
    parser.add_argument("--csi_path", required=True)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--decoder_args_json", default=None)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--source_name", default="source")
    parser.add_argument("--align_mode", default="affine",
                        choices=["identity", "procrustes", "affine"])
    parser.add_argument("--align_ridge", type=float, default=1e-4)
    parser.add_argument("--lora_target", default="fc_ffn",
                        choices=["fc", "ffn", "fc_ffn"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--fc_lora_rank", type=int, default=None)
    parser.add_argument("--ffn_lora_rank", type=int, default=None)
    parser.add_argument("--fc_lora_alpha", type=float, default=None)
    parser.add_argument("--ffn_lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--code_adapter", default="none",
                        choices=["none", "gated_lr_mlp"])
    parser.add_argument("--code_lowrank_rank", type=int, default=0)
    parser.add_argument("--code_mlp_hidden", type=int, default=0)
    parser.add_argument("--code_gate_lr", type=float, default=0.1)
    parser.add_argument("--code_gate_mlp", type=float, default=0.1)
    parser.add_argument("--code_adapter_dropout", type=float, default=0.0)
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
    parser.add_argument("--lambda_code", type=float, default=0.0)
    parser.add_argument("--lambda_delta", type=float, default=0.0)
    parser.add_argument("--lambda_recT", type=float, default=0.0)
    parser.add_argument("--lambda_fc", type=float, default=0.0)
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--decoder_name", default=None)
    parser.add_argument("--decoder_cr", type=int, default=None)
    parser.add_argument("--decoder_d_model", type=int, default=None)
    parser.add_argument("--decoder_dim_feedforward", type=int, default=None)
    parser.add_argument("--decoder_channel", type=int, default=None)
    parser.add_argument("--decoder_nt", type=int, default=None)
    parser.add_argument("--decoder_nc", type=int, default=None)
    parser.add_argument("--decoder_hidden", type=int, default=None)
    parser.add_argument("--decoder_num_blocks", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    tensorboard_dir = exp_dir / "tensorboard"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    with (exp_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    device = resolve_device(args.gpu, args.cpu)
    set_seed(args.seed)
    log_experiment_header(args, exp_dir=exp_dir, target_logger=logger)

    base_decoder, decoder_cfg = load_decoder_from_checkpoint(args, device)
    teacher_decoder, _ = load_decoder_from_checkpoint(args, device)
    injected = inject_decoder_lora(
        base_decoder,
        target=args.lora_target,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        fc_rank=args.fc_lora_rank,
        ffn_rank=args.ffn_lora_rank,
        fc_alpha=args.fc_lora_alpha,
        ffn_alpha=args.ffn_lora_alpha)
    mark_only_lora_trainable(base_decoder)
    source_fit, target_fit = load_code_pair(
        args.source_code,
        args.target_code,
        max_samples=args.max_samples)
    weight, bias = fit_alignment(
        args.align_mode,
        source_fit,
        target_fit,
        ridge=args.align_ridge)
    train_set = AlignedCodeCsiDataset(
        args.source_code,
        args.target_code,
        args.csi_path,
        weight,
        bias,
        channel=decoder_cfg["channel"],
        nt=decoder_cfg["nt"],
        nc=decoder_cfg["nc"],
        split="train",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
    use_val = args.val_ratio > 0
    val_set = None
    if use_val:
        val_set = AlignedCodeCsiDataset(
            args.source_code,
            args.target_code,
            args.csi_path,
            weight,
            bias,
            channel=decoder_cfg["channel"],
            nt=decoder_cfg["nt"],
            nc=decoder_cfg["nc"],
            split="val",
            val_ratio=args.val_ratio,
            max_samples=args.max_samples)
    all_set = AlignedCodeCsiDataset(
        args.source_code,
        args.target_code,
        args.csi_path,
        weight,
        bias,
        channel=decoder_cfg["channel"],
        nt=decoder_cfg["nt"],
        nc=decoder_cfg["nc"],
        split="all",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
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
    code_adapter = build_code_adapter(
        args,
        dim=train_set.code.size(1),
        device=device)
    model = DecoderLoraSystem(base_decoder, code_adapter).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        len(train_loader),
        args.eta_min)

    logger.info("=> Device: %s", device)
    logger.info("=> Source code: %s", args.source_code)
    logger.info("=> Target code: %s", args.target_code)
    logger.info("=> CSI path: %s", args.csi_path)
    logger.info("=> Decoder checkpoint: %s", args.decoder_checkpoint)
    logger.info(
        "=> Alignment: mode=%s ridge=%s weight=%s bias=%s buffers=%d",
        args.align_mode,
        args.align_ridge,
        tuple(weight.shape),
        tuple(bias.shape),
        weight.numel() + bias.numel())
    logger.info(
        "=> LoRA: target=%s rank=%d fc_rank=%s ffn_rank=%s "
        "alpha=%s fc_alpha=%s ffn_alpha=%s dropout=%s injected=%s",
        args.lora_target,
        args.lora_rank,
        args.fc_lora_rank if args.fc_lora_rank is not None else args.lora_rank,
        args.ffn_lora_rank if args.ffn_lora_rank is not None else args.lora_rank,
        args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
        args.fc_lora_alpha,
        args.ffn_lora_alpha,
        args.lora_dropout,
        ",".join(injected))
    logger.info(
        "=> Code adapter: type=%s lowrank_rank=%d mlp_hidden=%d "
        "gate_lr=%s gate_mlp=%s dropout=%s",
        args.code_adapter,
        args.code_lowrank_rank,
        args.code_mlp_hidden,
        args.code_gate_lr,
        args.code_gate_mlp,
        args.code_adapter_dropout)
    logger.info(
        "=> Parameters: trainable=%s lora=%s",
        f"{count_trainable_parameters(model):,}",
        f"{count_lora_parameters(base_decoder):,}")
    log_parameter_table(model, logger)
    logger.info(
        "=> Dataset sizes: train=%d val=%d all=%d",
        len(train_set),
        len(val_set) if use_val else 0,
        len(all_set))
    logger.info(
        "=> DataLoader: batch_size=%d workers=%d pin_memory=%s",
        args.batch_size,
        args.workers,
        device.type == "cuda")
    logger.info(
        "=> Optimizer: AdamW lr=%s weight_decay=%s",
        args.lr,
        args.weight_decay)
    logger.info(
        "=> Scheduler: %s eta_min=%s steps_per_epoch=%d total_steps=%d",
        args.scheduler,
        args.eta_min,
        len(train_loader),
        args.epochs * len(train_loader))
    logger.info(
        "=> Objective: rec + code*%s + delta*%s + recT*%s + fc*%s",
        args.lambda_code,
        args.lambda_delta,
        args.lambda_recT,
        args.lambda_fc)
    if not use_val:
        logger.info("val_ratio<=0: select best checkpoint by train loss")
    if args.eval_decoder_every:
        logger.info(
            "=> Periodic true NMSE eval: every=%d max_samples=%s",
            args.eval_decoder_every,
            args.eval_decoder_max_samples or "full")

    history = []
    best_loss = {"metric": math.inf, "epoch": 0, "selection": ""}
    best_nmse = {"metric": math.inf, "epoch": 0, "selection": ""}
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            lambda_code=args.lambda_code,
            lambda_delta=args.lambda_delta,
            lambda_recT=args.lambda_recT,
            lambda_fc=args.lambda_fc,
            teacher_decoder=teacher_decoder)
        if use_val:
            eval_prefix = "val"
            eval_metrics = run_epoch(
                model,
                val_loader,
                device,
                lambda_code=args.lambda_code,
                lambda_delta=args.lambda_delta,
                lambda_recT=args.lambda_recT,
                lambda_fc=args.lambda_fc,
                teacher_decoder=teacher_decoder)
        else:
            eval_prefix = "train"
            eval_metrics = train_metrics
        record = {
            "epoch": epoch,
            "lr": scheduler.get_lr()[0],
            "train": train_metrics,
            eval_prefix: eval_metrics,
        }
        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, eval_prefix, eval_metrics, epoch)
        writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step=epoch)

        decoder_metrics = None
        if args.eval_decoder_every and epoch % args.eval_decoder_every == 0:
            decoder_metrics = evaluate_decoder(
                model,
                decoder_eval_loader,
                device)
            record["true_decoder_eval"] = decoder_metrics
            log_metrics(writer, "true_decoder_eval", decoder_metrics, epoch)
            logger.info(
                f"Epoch [{epoch}/{args.epochs}] true_decoder_eval "
                f"n={decoder_metrics['n']} "
                f"decoder_mse={decoder_metrics['decoder_mse']:.6e} "
                f"decoder_nmse={decoder_metrics['decoder_nmse']:.3f}dB")
        history.append(record)

        selected_loss = eval_metrics["loss"]
        loss_metric = -float(epoch) if args.save_last else selected_loss
        if loss_metric < best_loss["metric"]:
            best_loss = {
                "metric": loss_metric,
                "loss": selected_loss,
                "rec": eval_metrics["rec"],
                "epoch": epoch,
                "selection": "last" if args.save_last
                else f"{eval_prefix}_loss",
            }
            save_lora_state(
                model,
                checkpoint_dir / "best_loss.pth",
                epoch,
                best_loss,
                args,
                optimizer,
                scheduler)
        if decoder_metrics is not None:
            nmse_metric = decoder_metrics["decoder_nmse"]
            if nmse_metric < best_nmse["metric"]:
                best_nmse = {
                    "metric": nmse_metric,
                    "decoder_mse": decoder_metrics["decoder_mse"],
                    "epoch": epoch,
                    "selection": "true_decoder_nmse",
                }
                save_lora_state(
                    model,
                    checkpoint_dir / "best_nmse.pth",
                    epoch,
                    best_nmse,
                    args,
                    optimizer,
                    scheduler)
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"lr={scheduler.get_lr()[0]:.6e} "
            f"train_opt_loss={train_metrics['loss']:.6e} "
            f"{eval_prefix}_select_loss={selected_loss:.6e} "
            f"{eval_prefix}_rec={eval_metrics['rec']:.6e} "
            f"{eval_prefix}_code={eval_metrics['code']:.6e} "
            f"{eval_prefix}_delta={eval_metrics['delta']:.6e} "
            f"{eval_prefix}_recT={eval_metrics['recT']:.6e} "
            f"{eval_prefix}_fc={eval_metrics['fc']:.6e}")

    (exp_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8")

    metrics = {"best_loss": best_loss, "best_nmse": best_nmse}
    for tag, path in [
            ("best_loss", checkpoint_dir / "best_loss.pth"),
            ("best_nmse", checkpoint_dir / "best_nmse.pth")]:
        if not path.exists():
            continue
        base_decoder, _ = load_decoder_from_checkpoint(args, device)
        inject_decoder_lora(
            base_decoder,
            target=args.lora_target,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            fc_rank=args.fc_lora_rank,
            ffn_rank=args.ffn_lora_rank,
            fc_alpha=args.fc_lora_alpha,
            ffn_alpha=args.ffn_lora_alpha)
        final_code_adapter = build_code_adapter(
            args,
            dim=train_set.code.size(1),
            device=device)
        final_model = DecoderLoraSystem(base_decoder, final_code_adapter).to(device)
        load_lora_state(final_model, path, device)
        final_metrics = evaluate_decoder(final_model, all_loader, device)
        metrics[tag]["all"] = final_metrics
        logger.info(
            f"all_{tag}_decoder_mse={final_metrics['decoder_mse']:.6e} "
            f"all_{tag}_decoder_nmse={final_metrics['decoder_nmse']:.3f}dB "
            f"epoch={metrics[tag]['epoch']}")
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8")
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
