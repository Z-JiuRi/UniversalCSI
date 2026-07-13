#!/usr/bin/env python
import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard.writer import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter.models import build_mapper  # noqa: E402
from utils.logger import (logger, log_experiment_header, log_parameter_table,  # noqa: E402
                          setup_logging)
from utils.scheduler import FakeLR, WarmUpCosineAnnealingLR  # noqa: E402


def parse_int_list(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item) for item in text.split(",") if item.strip()]


class CodeCsiDataset(Dataset):
    def __init__(self, source_code, target_code, csi):
        if source_code.ndim != 2 or target_code.ndim != 2:
            raise ValueError("source_code and target_code must be 2D tensors")
        if source_code.shape != target_code.shape:
            raise ValueError(
                f"source/target code shape mismatch: "
                f"{tuple(source_code.shape)} vs {tuple(target_code.shape)}")
        if csi.ndim != 4:
            raise ValueError(f"csi must be 4D, got {tuple(csi.shape)}")
        n = min(source_code.size(0), target_code.size(0), csi.size(0))
        self.source = source_code[:n].contiguous()
        self.target = target_code[:n].contiguous()
        self.csi = csi[:n].contiguous()
        self.indices = torch.arange(n, dtype=torch.long)

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx], self.csi[idx], self.indices[idx]


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


def load_code(path, max_samples=0):
    code = torch.load(path, weights_only=True, map_location="cpu").float()
    if code.ndim != 2:
        raise ValueError(f"{path}: expected 2D code tensor, got {tuple(code.shape)}")
    if max_samples and code.size(0) > max_samples:
        code = code[:max_samples].contiguous()
    return code


def load_optional_code(path, max_samples=0):
    if not path:
        return None
    return load_code(path, max_samples)


def load_csi(path, channel=2, nt=32, nc=32, max_samples=0):
    data = torch.load(path, weights_only=True, map_location="cpu").float()
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(
            f"{path}: expected (N,{channel},{nt},{nc}), got {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data


def fit_affine(source, target, ridge=1.0):
    dim = source.size(1)
    src = source.to(torch.float64)
    tgt = target.to(torch.float64)
    ones = torch.ones(src.size(0), 1, dtype=src.dtype)
    aug = torch.cat([src, ones], dim=1)
    reg = ridge * torch.eye(dim + 1, dtype=src.dtype)
    reg[-1, -1] = 0.0
    solution = torch.linalg.solve(aug.t().matmul(aug) + reg, aug.t().matmul(tgt))
    return solution[:-1].float().contiguous(), solution[-1].float().contiguous()


def load_main_models_package():
    package_name = "adapter_main_models"
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


def load_decoder(args, device):
    cfg = {}
    if args.decoder_args_json:
        cfg = json.loads(Path(args.decoder_args_json).read_text())
    main_models = load_main_models_package()
    decoder_name = cfg.get("decoder", args.decoder)
    cr = cfg.get("cr", args.cr)
    d_model = cfg.get("d_model", args.d_model)
    channel = cfg.get("channel", args.channel)
    nt = cfg.get("nt", args.nt)
    nc = cfg.get("nc", args.nc)
    dim_feedforward = cfg.get("dim_feedforward", args.dim_feedforward)
    hidden = cfg.get("hidden", args.hidden)
    num_blocks = cfg.get("num_blocks", args.decoder_num_blocks)
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
    missing, unexpected = model.decoder.load_state_dict(decoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"decoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    decoder = model.decoder.to(device).eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder, {"channel": channel, "nt": nt, "nc": nc}


def load_target_encoder(args, device):
    cfg = {}
    args_json = args.encoder_args_json or args.decoder_args_json
    if args_json:
        cfg = json.loads(Path(args_json).read_text())
    main_models = load_main_models_package()
    encoder_name = cfg.get("encoder", args.encoder)
    decoder_name = cfg.get("decoder", args.decoder)
    cr = cfg.get("cr", args.cr)
    d_model = cfg.get("d_model", args.d_model)
    channel = cfg.get("channel", args.channel)
    nt = cfg.get("nt", args.nt)
    nc = cfg.get("nc", args.nc)
    dim_feedforward = cfg.get("dim_feedforward", args.dim_feedforward)
    hidden = cfg.get("hidden", args.hidden)
    num_blocks = cfg.get("num_blocks", args.decoder_num_blocks)
    model = main_models.universal_csi(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        reduction=cr,
        d_model=d_model,
        channel=channel,
        nt=nt,
        nc=nc,
        dim_feedforward=dim_feedforward,
        hidden=hidden,
        num_blocks=num_blocks)
    checkpoint = args.encoder_checkpoint or args.decoder_checkpoint
    state_dict = clean_state_dict(checkpoint)
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise RuntimeError(f"encoder checkpoint has no encoder.* weights: {checkpoint}")
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    encoder = model.encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def build_optimizer(model, lr, weight_decay, affine_lr_multiplier=1.0):
    groups = {
        "main_decay": [],
        "main_no_decay": [],
        "affine_decay": [],
        "affine_no_decay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_affine = name in ("alignment_weight", "alignment_bias")
        no_decay = param.ndim == 1 or name.endswith(".bias")
        prefix = "affine" if is_affine else "main"
        suffix = "no_decay" if no_decay else "decay"
        groups[f"{prefix}_{suffix}"].append(param)
    param_groups = []
    for name, params in groups.items():
        if not params:
            continue
        group_lr = lr * affine_lr_multiplier if name.startswith("affine") else lr
        group_decay = 0.0 if name.endswith("no_decay") else weight_decay
        param_groups.append({
            "params": params,
            "lr": group_lr,
            "initial_lr": group_lr,
            "weight_decay": group_decay,
            "group_name": name,
        })
    return torch.optim.AdamW(param_groups, lr=lr)


def linear_warmup(epoch, start_epoch, warmup_epochs, target):
    if target <= 0 or epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return target
    progress = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
    return target * progress


def stage_hyperparameters(args, epoch):
    if args.stage1_epochs <= 0 or epoch <= args.stage1_epochs:
        return {
            "stage": 1 if args.stage1_epochs > 0 else 0,
            "lambda_recon": (
                args.stage1_lambda_recon if args.stage1_epochs > 0
                else args.lambda_recon),
            "lambda_encoder_consistency": (
                args.stage1_lambda_encoder_consistency
                if args.stage1_epochs > 0
                else args.lambda_encoder_consistency),
            "code_noise_std": (
                args.stage1_code_noise_std if args.stage1_epochs > 0
                else args.code_noise_std),
        }

    stage2_epoch = epoch - args.stage1_epochs
    recon = linear_warmup(
        stage2_epoch, 1, args.stage2_recon_warmup_epochs,
        args.lambda_recon)
    encoder = linear_warmup(
        stage2_epoch, args.stage2_encoder_delay_epochs + 1,
        args.stage2_encoder_warmup_epochs,
        args.lambda_encoder_consistency)
    noise = args.code_noise_std
    if args.stage2_noise_decay_epochs > 0:
        stage2_epochs = args.epochs - args.stage1_epochs
        decay_start = max(1, stage2_epochs - args.stage2_noise_decay_epochs + 1)
        if stage2_epoch >= decay_start:
            remaining = stage2_epochs - stage2_epoch
            noise *= max(0.0, remaining / args.stage2_noise_decay_epochs)
    return {
        "stage": 2,
        "lambda_recon": recon,
        "lambda_encoder_consistency": encoder,
        "code_noise_std": noise,
    }


def build_scheduler(optimizer, name, epochs, steps_per_epoch, eta_min):
    if name == "const":
        return FakeLR(optimizer)
    if name == "cosine":
        total_steps = max(1, epochs * steps_per_epoch)
        return WarmUpCosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            T_warmup=0.1 * total_steps,
            eta_min=eta_min)
    raise ValueError(f"Unknown scheduler: {name}")


class ModelEMA:
    def __init__(self, model, decay=0.999, update_every=1):
        if not 0.0 < decay < 1.0:
            raise ValueError("ema_decay must satisfy 0 < decay < 1")
        if update_every <= 0:
            raise ValueError("ema_update_every must be positive")
        self.decay = decay
        self.update_every = update_every
        self.num_updates = 0
        self.shadow = {}
        self.backup = None
        self.reset(model)

    @torch.no_grad()
    def reset(self, model):
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        if self.num_updates % self.update_every:
            return
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if value.is_floating_point():
                shadow.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    @torch.no_grad()
    def apply(self, model):
        if self.backup is not None:
            raise RuntimeError("EMA weights are already applied")
        self.backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)

    @torch.no_grad()
    def restore(self, model):
        if self.backup is None:
            raise RuntimeError("EMA weights are not applied")
        model.load_state_dict(self.backup)
        self.backup = None

    def state_dict(self):
        return {
            "decay": self.decay,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }


@torch.no_grad()
def code_metrics(pred, target):
    err = pred - target
    mse = err.pow(2).mean()
    nmse = 10.0 * torch.log10(
        err.pow(2).sum() / target.pow(2).sum().clamp_min(1e-12))
    cos = F.cosine_similarity(pred, target, dim=1).mean()
    return {
        "code_mse": float(mse.cpu()),
        "code_nmse": float(nmse.cpu()),
        "code_cos": float(cos.cpu()),
    }


def estimate_decoder_jacobian_sensitivity(
        decoder, probe_codes, device, n_hutchinson=8, max_samples=2048,
        batch_size=256):
    """Hutchinson estimate of diag(J^T J) for y=D_t(z), normalized to mean 1."""
    decoder.eval()
    codes = probe_codes[:max_samples].to(device)
    if codes.numel() == 0:
        raise ValueError("probe_codes is empty")
    dim = codes.size(1)
    sens = torch.zeros(dim, device=device, dtype=torch.float64)
    total = 0
    for start in range(0, codes.size(0), batch_size):
        z = codes[start:start + batch_size].detach().requires_grad_(True)
        y = decoder(z)
        for _ in range(n_hutchinson):
            v = torch.randn_like(y)
            grad_z = torch.autograd.grad(
                (y * v).sum(), z, retain_graph=True)[0]
            sens += grad_z.detach().double().pow(2).sum(dim=0)
        total += z.size(0) * n_hutchinson
    sens = sens / max(total, 1)
    sens = sens / sens.mean().clamp_min(1e-12)
    return sens.float().cpu()


def estimate_decoder_fc_sensitivity(decoder):
    if not hasattr(decoder, "fc_decoder"):
        raise AttributeError(
            "fc_decoder sensitivity requires decoder.fc_decoder")
    with torch.no_grad():
        sensitivity = decoder.fc_decoder.weight.detach().float().pow(2).sum(dim=0)
        sensitivity = sensitivity / sensitivity.mean().clamp_min(1e-12)
    return sensitivity.cpu()


def build_code_loss_weight(target_code, args, residual_code=None):
    if args.code_loss_type not in (
            "clipped_std_mse",
            "clipped_var_mse",
            "clipped_power_mse",
            "clipped_residual_std_mse",
            "decoder_sensitivity_mse",
            "decoder_jac_residual_mse"):
        return None, None
    if args.code_loss_type in (
            "decoder_sensitivity_mse", "decoder_jac_residual_mse"):
        if not hasattr(args, "_decoder_sensitivity_weight"):
            raise ValueError(
                f"{args.code_loss_type} requires _decoder_sensitivity_weight")
        weight = args._decoder_sensitivity_weight.float()
        scale = weight
        if args.code_loss_type == "decoder_jac_residual_mse":
            if residual_code is None:
                raise ValueError(
                    "residual_code is required for decoder_jac_residual_mse")
            # Emphasize dims that are both decoder-sensitive and hard to map
            # (large residual std after affine).
            resid_std = residual_code.float().std(dim=0).clamp_min(
                args.std_weight_eps)
            resid_factor = resid_std / resid_std.mean().clamp_min(1e-12)
            weight = weight * resid_factor
            scale = weight
    elif args.code_loss_type == "clipped_residual_std_mse":
        if residual_code is None:
            raise ValueError("residual_code is required for clipped_residual_std_mse")
        scale = residual_code.float().std(dim=0).clamp_min(args.std_weight_eps)
        weight = scale.mean() / scale
    elif args.code_loss_type == "clipped_var_mse":
        scale = target_code.float().var(dim=0, unbiased=False).clamp_min(
            args.std_weight_eps)
        weight = 1.0 / scale
    elif args.code_loss_type == "clipped_power_mse":
        scale = target_code.float().pow(2).mean(dim=0).clamp_min(
            args.std_weight_eps)
        weight = 1.0 / scale
    else:
        scale = target_code.float().std(dim=0).clamp_min(args.std_weight_eps)
        weight = 1.0 / scale
    power = float(getattr(args, "sensitivity_power", 1.0) or 1.0)
    if power != 1.0 and args.code_loss_type in (
            "decoder_sensitivity_mse", "decoder_jac_residual_mse"):
        weight = weight.clamp_min(args.std_weight_eps).pow(power)
        weight = weight / weight.mean().clamp_min(1e-12)
    weight = weight.clamp(args.std_weight_min, args.std_weight_max)
    return weight.contiguous(), scale.contiguous()


def compute_code_loss(pred, target, code_loss_type="mse", std_weight=None):
    sqerr = (pred - target).pow(2)
    raw_mse = sqerr.mean()
    if code_loss_type == "mse":
        return raw_mse, raw_mse
    if code_loss_type in (
            "clipped_std_mse",
            "clipped_var_mse",
            "clipped_power_mse",
            "clipped_residual_std_mse",
            "decoder_sensitivity_mse",
            "decoder_jac_residual_mse"):
        if std_weight is None:
            raise ValueError(f"std_weight is required for {code_loss_type}")
        weighted = sqerr * std_weight.view(1, -1)
        return weighted.mean(), raw_mse
    raise ValueError(f"Unknown code_loss_type: {code_loss_type}")


def get_start_code(model, source):
    if not hasattr(model, "start"):
        return None
    return model.start(source)


def init_delta_totals(device):
    return {
        "delta_target_cos": 0.0,
        "delta_z0_cos": 0.0,
        "n": 0,
        "delta_sq": torch.tensor(0.0, device=device),
        "target_residual_sq": torch.tensor(0.0, device=device),
        "z0_sq": torch.tensor(0.0, device=device),
        "delta_sum": torch.tensor(0.0, device=device),
        "delta_sumsq": torch.tensor(0.0, device=device),
        "delta_numel": 0,
        "delta_small": torch.tensor(0.0, device=device),
    }


@torch.no_grad()
def update_delta_totals(totals, mapped, target, z0, small_eps=1e-4):
    if z0 is None:
        return
    mapped = mapped.detach()
    target = target.detach()
    z0 = z0.detach()
    delta = mapped - z0
    target_residual = target - z0
    n = mapped.size(0)
    totals["delta_target_cos"] += float(
        F.cosine_similarity(delta, target_residual, dim=1).mean().cpu()) * n
    totals["delta_z0_cos"] += float(
        F.cosine_similarity(delta, z0, dim=1).mean().cpu()) * n
    totals["n"] += n
    totals["delta_sq"] += delta.pow(2).sum()
    totals["target_residual_sq"] += target_residual.pow(2).sum()
    totals["z0_sq"] += z0.pow(2).sum()
    totals["delta_sum"] += delta.sum()
    totals["delta_sumsq"] += delta.pow(2).sum()
    totals["delta_numel"] += delta.numel()
    totals["delta_small"] += (delta.abs() < small_eps).float().sum()


def finalize_delta_metrics(totals):
    if totals["n"] == 0:
        return {}
    delta_numel = max(totals["delta_numel"], 1)
    delta_mean = totals["delta_sum"] / delta_numel
    delta_var = totals["delta_sumsq"] / delta_numel - delta_mean.pow(2)
    return {
        "delta_target_cos": totals["delta_target_cos"] / totals["n"],
        "delta_z0_cos": totals["delta_z0_cos"] / totals["n"],
        "residual_coverage": float(torch.sqrt(
            totals["delta_sq"] /
            totals["target_residual_sq"].clamp_min(1e-12)).cpu()),
        "delta_ratio_from_z0": float(torch.sqrt(
            totals["delta_sq"] / totals["z0_sq"].clamp_min(1e-12)).cpu()),
        "delta_mean": float(delta_mean.cpu()),
        "delta_std": float(torch.sqrt(delta_var.clamp_min(0.0)).cpu()),
        "delta_small_frac": float((totals["delta_small"] / delta_numel).cpu()),
    }


@torch.no_grad()
def collect_parameter_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        key = name.replace(".", "/")
        data = param.detach().float()
        stats[f"{key}/mean"] = float(data.mean().cpu())
        stats[f"{key}/std"] = float(data.std(unbiased=False).cpu())
        stats[f"{key}/min"] = float(data.min().cpu())
        stats[f"{key}/max"] = float(data.max().cpu())
        stats[f"{key}/norm"] = float(data.norm().cpu())
        if param.grad is not None:
            grad = param.grad.detach().float()
            grad_norm = grad.norm()
            stats[f"{key}/grad_norm"] = float(grad_norm.cpu())
            stats[f"{key}/update_ratio"] = float(
                (grad_norm / data.norm().clamp_min(1e-12)).cpu())
    return stats


def summarize_parameter_stats(stats):
    groups = {}
    for key, value in stats.items():
        if not key.endswith("/norm"):
            continue
        group = key.split("/", 1)[0]
        groups.setdefault(group, 0.0)
        groups[group] += value
    return {f"{group}/norm_sum": value for group, value in sorted(groups.items())}


def loss_gradient_diagnostics(losses, parameters):
    """Return weighted-loss gradient norms/cosines without changing .grad."""
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = {}
    metrics = {}
    for name, loss in losses.items():
        if loss is None or not loss.requires_grad:
            continue
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = grads
        norm_sq = sum(
            grad.detach().double().pow(2).sum()
            for grad in grads if grad is not None)
        metrics[f"grad_norm_{name}"] = float(torch.sqrt(norm_sq).cpu())
    names = sorted(gradients)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            dot = sum(
                grad_left.detach().double().mul(grad_right.detach().double()).sum()
                for grad_left, grad_right in zip(
                    gradients[left], gradients[right])
                if grad_left is not None and grad_right is not None)
            denominator = (
                metrics[f"grad_norm_{left}"] * metrics[f"grad_norm_{right}"])
            cosine = float(dot.cpu()) / max(denominator, 1e-30)
            metrics[f"grad_cos_{left}_{right}"] = cosine
    return metrics


def train_epoch(model, loader, decoder, device, optimizer, scheduler,
                lambda_code, lambda_recon, lambda_feature=0.0,
                code_loss_type="mse", std_weight=None, gate_l1=0.0,
                target_encoder=None, lambda_encoder_consistency=0.0,
                encoder_consistency_target="mapped",
                code_noise_std=0.0, lambda_delta_norm=0.0, ema=None,
                teacher_code=None, lambda_teacher_code=0.0,
                fisher_basis=None, fisher_weight=None, lambda_fisher=0.0,
                gradient_diagnostics=False):
    model.train()
    decoder.eval()
    total = {
        "loss": 0.0,
        "code_loss": 0.0,
        "code_mse": 0.0,
        "feature_mse": 0.0,
        "teacher_code_mse": 0.0,
        "fisher_mse": 0.0,
        "encoder_consistency_mse": 0.0,
        "delta_norm_mse": 0.0,
        "gate_l1": 0.0,
        "recon_mse": 0.0,
        "cos": 0.0,
        "n": 0,
    }
    code_err = torch.tensor(0.0, device=device)
    code_power = torch.tensor(0.0, device=device)
    recon_err = torch.tensor(0.0, device=device)
    recon_power = torch.tensor(0.0, device=device)
    delta_totals = init_delta_totals(device)
    gradient_metrics = {}
    for batch_index, (source, target, csi, indices) in enumerate(loader):
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        csi = csi.to(device, non_blocking=True)
        source_input = source
        if code_noise_std > 0:
            noise_scale = source.detach().std(dim=0, unbiased=False).clamp_min(1e-6)
            source_input = source + code_noise_std * noise_scale * torch.randn_like(source)
        mapped = model(source_input)
        with torch.no_grad():
            z0_diag = get_start_code(model, source_input)
            update_delta_totals(delta_totals, mapped, target, z0_diag)
        code_loss, code_mse = compute_code_loss(
            mapped, target, code_loss_type, std_weight)
        teacher_code_loss = mapped.new_tensor(0.0)
        if lambda_teacher_code > 0:
            if teacher_code is None:
                raise ValueError("teacher_code is required when lambda_teacher_code > 0")
            teacher = teacher_code[indices].to(device, non_blocking=True)
            teacher_code_loss = F.mse_loss(mapped, teacher)
        fisher_loss = mapped.new_tensor(0.0)
        if lambda_fisher > 0:
            if fisher_basis is None:
                raise ValueError("fisher_basis is required when lambda_fisher > 0")
            coefficients = (mapped - target).matmul(fisher_basis)
            if fisher_weight is not None:
                coefficients = coefficients * fisher_weight.sqrt().view(1, -1)
            fisher_loss = coefficients.pow(2).mean()
        feature_loss = mapped.new_tensor(0.0)
        if lambda_feature > 0:
            if not hasattr(decoder, "fc_decoder"):
                raise AttributeError("decoder has no fc_decoder for feature loss")
            feature_loss = F.mse_loss(
                decoder.fc_decoder(mapped),
                decoder.fc_decoder(target).detach())
        gate_reg = mapped.new_tensor(0.0)
        if gate_l1 > 0 and hasattr(model, "gate_regularization"):
            reg = model.gate_regularization()
            if reg is not None:
                gate_reg = reg
        delta_norm_loss = mapped.new_tensor(0.0)
        if lambda_delta_norm > 0:
            z0 = get_start_code(model, source_input)
            delta_norm_loss = F.mse_loss(mapped, z0)
        need_decoder_grad = lambda_recon > 0 or lambda_encoder_consistency > 0
        encoder_consistency_loss = mapped.new_tensor(0.0)
        if need_decoder_grad:
            recon = decoder(mapped)
            recon_loss = F.mse_loss(recon, csi)
            if lambda_encoder_consistency > 0:
                if target_encoder is None:
                    raise ValueError(
                        "target_encoder is required when lambda_encoder_consistency > 0")
                reencoded = target_encoder(recon)
                if encoder_consistency_target == "mapped":
                    enc_target = mapped
                elif encoder_consistency_target == "target":
                    enc_target = target
                else:
                    raise ValueError(
                        f"Unknown encoder_consistency_target: "
                        f"{encoder_consistency_target}")
                encoder_consistency_loss = F.mse_loss(reencoded, enc_target)
            loss = (
                lambda_code * code_loss
                + lambda_teacher_code * teacher_code_loss
                + lambda_fisher * fisher_loss
                + lambda_feature * feature_loss
                + lambda_recon * recon_loss
                + lambda_encoder_consistency * encoder_consistency_loss
                + lambda_delta_norm * delta_norm_loss
                + gate_l1 * gate_reg)
        else:
            with torch.no_grad():
                recon = decoder(mapped.detach())
                recon_loss = F.mse_loss(recon, csi)
            loss = (
                lambda_code * code_loss
                + lambda_teacher_code * teacher_code_loss
                + lambda_fisher * fisher_loss
                + lambda_feature * feature_loss
                + lambda_delta_norm * delta_norm_loss
                + gate_l1 * gate_reg)
        if gradient_diagnostics and batch_index == 0:
            diagnostic_losses = {
                "code": lambda_code * code_loss if lambda_code > 0 else None,
                "recon": lambda_recon * recon_loss
                if lambda_recon > 0 else None,
                "encoder": lambda_encoder_consistency * encoder_consistency_loss
                if lambda_encoder_consistency > 0 else None,
                "fisher": lambda_fisher * fisher_loss
                if lambda_fisher > 0 else None,
            }
            gradient_metrics = loss_gradient_diagnostics(
                diagnostic_losses, model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if scheduler is not None:
            scheduler.step()
        n = source.size(0)
        cerr = mapped.detach() - target
        rerr = recon.detach() - csi
        code_err += cerr.pow(2).sum()
        code_power += target.pow(2).sum()
        recon_err += rerr.pow(2).sum()
        recon_power += csi.pow(2).sum()
        total["loss"] += float(loss.detach().cpu()) * n
        total["code_loss"] += float(code_loss.detach().cpu()) * n
        total["code_mse"] += float(code_mse.detach().cpu()) * n
        total["feature_mse"] += float(feature_loss.detach().cpu()) * n
        total["teacher_code_mse"] += float(teacher_code_loss.detach().cpu()) * n
        total["fisher_mse"] += float(fisher_loss.detach().cpu()) * n
        total["encoder_consistency_mse"] += float(
            encoder_consistency_loss.detach().cpu()) * n
        total["delta_norm_mse"] += float(delta_norm_loss.detach().cpu()) * n
        total["gate_l1"] += float(gate_reg.detach().cpu()) * n
        total["recon_mse"] += float(recon_loss.detach().cpu()) * n
        total["cos"] += float(
            F.cosine_similarity(mapped.detach(), target, dim=1).mean().cpu()) * n
        total["n"] += n
    metrics = {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}
    metrics["code_nmse"] = float((10.0 * torch.log10(
        code_err / code_power.clamp_min(1e-12))).cpu())
    metrics["code_cos"] = metrics.pop("cos")
    metrics["decoder_nmse"] = float((10.0 * torch.log10(
        recon_err / recon_power.clamp_min(1e-12))).cpu())
    metrics["decoder_mse"] = metrics["recon_mse"]
    metrics.update(finalize_delta_metrics(delta_totals))
    metrics.update(gradient_metrics)
    return metrics


@torch.no_grad()
def evaluate(model, loader, decoder, device, code_loss_type="mse",
             std_weight=None, target_encoder=None,
             encoder_consistency_target="mapped",
             fisher_basis=None, fisher_eigenvalues=None):
    model.eval()
    decoder.eval()
    code_err = torch.tensor(0.0, device=device)
    code_power = torch.tensor(0.0, device=device)
    recon_err = torch.tensor(0.0, device=device)
    recon_power = torch.tensor(0.0, device=device)
    code_loss_sum = 0.0
    code_mse_sum = 0.0
    feature_mse_sum = 0.0
    encoder_consistency_sum = 0.0
    recon_mse_sum = 0.0
    cos_sum = 0.0
    n_total = 0
    delta_totals = init_delta_totals(device)
    z0_recon_err = torch.tensor(0.0, device=device)
    target_recon_err = torch.tensor(0.0, device=device)
    decoder_delta_err = torch.tensor(0.0, device=device)
    decoder_delta_code = torch.tensor(0.0, device=device)
    reencoded_target_mse_sum = 0.0
    reencoded_mapped_mse_sum = 0.0
    reencoded_target_cos_sum = 0.0
    reencoded_mapped_cos_sum = 0.0
    fisher_sq_sum = None
    if fisher_basis is not None:
        fisher_sq_sum = torch.zeros(
            fisher_basis.size(1), device=device, dtype=torch.float64)
    for source, target, csi, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        csi = csi.to(device, non_blocking=True)
        mapped = model(source)
        z0 = get_start_code(model, source)
        recon = decoder(mapped)
        update_delta_totals(delta_totals, mapped, target, z0)
        if z0 is not None:
            z0_recon = decoder(z0)
            target_recon = decoder(target)
            z0_recon_err += (z0_recon - csi).pow(2).sum()
            target_recon_err += (target_recon - csi).pow(2).sum()
            decoder_delta_err += (recon - z0_recon).pow(2).sum()
            decoder_delta_code += (mapped - z0).pow(2).sum()
        if hasattr(decoder, "fc_decoder"):
            feature_mse = F.mse_loss(
                decoder.fc_decoder(mapped),
                decoder.fc_decoder(target))
        else:
            feature_mse = mapped.new_tensor(0.0)
        if target_encoder is not None:
            reencoded = target_encoder(recon)
            reencoded_target_mse = F.mse_loss(reencoded, target)
            reencoded_mapped_mse = F.mse_loss(reencoded, mapped)
            reencoded_target_cos = F.cosine_similarity(
                reencoded, target, dim=1).mean()
            reencoded_mapped_cos = F.cosine_similarity(
                reencoded, mapped, dim=1).mean()
            if encoder_consistency_target == "mapped":
                enc_target = mapped
            elif encoder_consistency_target == "target":
                enc_target = target
            else:
                raise ValueError(
                    f"Unknown encoder_consistency_target: "
                    f"{encoder_consistency_target}")
            encoder_consistency_mse = F.mse_loss(reencoded, enc_target)
        else:
            encoder_consistency_mse = mapped.new_tensor(0.0)
            reencoded_target_mse = mapped.new_tensor(0.0)
            reencoded_mapped_mse = mapped.new_tensor(0.0)
            reencoded_target_cos = mapped.new_tensor(0.0)
            reencoded_mapped_cos = mapped.new_tensor(0.0)
        n = source.size(0)
        cerr = mapped - target
        if fisher_sq_sum is not None:
            fisher_coeff = cerr.matmul(fisher_basis)
            fisher_sq_sum += fisher_coeff.double().pow(2).sum(dim=0)
        rerr = recon - csi
        code_loss, code_mse = compute_code_loss(
            mapped, target, code_loss_type, std_weight)
        code_err += cerr.pow(2).sum()
        code_power += target.pow(2).sum()
        recon_err += rerr.pow(2).sum()
        recon_power += csi.pow(2).sum()
        code_loss_sum += float(code_loss.cpu()) * n
        code_mse_sum += float(code_mse.cpu()) * n
        feature_mse_sum += float(feature_mse.cpu()) * n
        encoder_consistency_sum += float(encoder_consistency_mse.cpu()) * n
        reencoded_target_mse_sum += float(reencoded_target_mse.cpu()) * n
        reencoded_mapped_mse_sum += float(reencoded_mapped_mse.cpu()) * n
        reencoded_target_cos_sum += float(reencoded_target_cos.cpu()) * n
        reencoded_mapped_cos_sum += float(reencoded_mapped_cos.cpu()) * n
        recon_mse_sum += float(rerr.pow(2).mean().cpu()) * n
        cos_sum += float(F.cosine_similarity(mapped, target, dim=1).mean().cpu()) * n
        n_total += n
    metrics = {
        "code_loss": code_loss_sum / max(n_total, 1),
        "code_mse": code_mse_sum / max(n_total, 1),
        "feature_mse": feature_mse_sum / max(n_total, 1),
        "encoder_consistency_mse": encoder_consistency_sum / max(n_total, 1),
        "reencoded_target_mse": reencoded_target_mse_sum / max(n_total, 1),
        "reencoded_mapped_mse": reencoded_mapped_mse_sum / max(n_total, 1),
        "reencoded_target_cos": reencoded_target_cos_sum / max(n_total, 1),
        "reencoded_mapped_cos": reencoded_mapped_cos_sum / max(n_total, 1),
        "code_nmse": float((10.0 * torch.log10(
            code_err / code_power.clamp_min(1e-12))).cpu()),
        "code_cos": cos_sum / max(n_total, 1),
        "decoder_mse": recon_mse_sum / max(n_total, 1),
        "decoder_nmse": float((10.0 * torch.log10(
            recon_err / recon_power.clamp_min(1e-12))).cpu()),
        "n": n_total,
    }
    metrics.update(finalize_delta_metrics(delta_totals))
    if fisher_sq_sum is not None:
        dim = fisher_sq_sum.numel()
        total_projected = fisher_sq_sum.sum().clamp_min(1e-30)
        if fisher_eigenvalues is None:
            normalized_eigenvalues = torch.ones_like(fisher_sq_sum)
        else:
            normalized_eigenvalues = fisher_eigenvalues[:dim].to(
                device=device, dtype=torch.float64).clamp_min(0.0)
            normalized_eigenvalues /= normalized_eigenvalues.mean().clamp_min(1e-30)
        weighted_by_dim = fisher_sq_sum * normalized_eigenvalues
        total_weighted = weighted_by_dim.sum().clamp_min(1e-30)
        metrics["fisher_full_mse"] = float(
            (total_projected / max(n_total * dim, 1)).cpu())
        metrics["fisher_full_weighted_mse"] = float(
            (total_weighted / max(n_total * dim, 1)).cpu())
        boundaries = [0, 16, 32, 64, 128, 256, 384, dim]
        boundaries = sorted(set(min(max(value, 0), dim) for value in boundaries))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            prefix = f"fisher_band_{start:03d}_{end:03d}"
            band_sq = fisher_sq_sum[start:end].sum()
            band_weighted = weighted_by_dim[start:end].sum()
            metrics[f"{prefix}_mse"] = float(
                (band_sq / max(n_total * (end - start), 1)).cpu())
            metrics[f"{prefix}_energy_frac"] = float(
                (band_sq / total_projected).cpu())
            metrics[f"{prefix}_impact_frac"] = float(
                (band_weighted / total_weighted).cpu())
        for rank in (64, 128, 256, 384):
            if rank > dim:
                continue
            top_sq = fisher_sq_sum[:rank].sum()
            top_weighted = weighted_by_dim[:rank].sum()
            metrics[f"fisher_top{rank}_mse"] = float(
                (top_sq / max(n_total * rank, 1)).cpu())
            metrics[f"fisher_top{rank}_energy_frac"] = float(
                (top_sq / total_projected).cpu())
            metrics[f"fisher_top{rank}_impact_frac"] = float(
                (top_weighted / total_weighted).cpu())
    if delta_totals["n"] > 0:
        metrics["z0_decoder_nmse"] = float((10.0 * torch.log10(
            z0_recon_err / recon_power.clamp_min(1e-12))).cpu())
        metrics["target_decoder_nmse"] = float((10.0 * torch.log10(
            target_recon_err / recon_power.clamp_min(1e-12))).cpu())
        metrics["decoder_sensitivity"] = float(torch.sqrt(
            decoder_delta_err / decoder_delta_code.clamp_min(1e-12)).cpu())
    return metrics


@torch.no_grad()
def export_mapped_code(model, dataset, device, output_path, batch_size, workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    model.eval()
    chunks = []
    indices = []
    for source, _, _, idx in loader:
        source = source.to(device, non_blocking=True)
        chunks.append(model(source).cpu())
        indices.append(idx.cpu())
    mapped = torch.cat(chunks, dim=0)
    idx = torch.cat(indices, dim=0).long()
    if not torch.equal(idx, torch.arange(idx.numel())):
        aligned = torch.empty_like(mapped)
        aligned[idx] = mapped
        mapped = aligned
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, output_path)
    return tuple(mapped.shape)


def log_metrics(writer, prefix, metrics, step):
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters()
                    if param.requires_grad)
    return total, trainable


def add_text_json(writer, tag, payload, step=0):
    text = "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"
    writer.add_text(tag, text, step)


def split_paths(args, prefix, split):
    root = getattr(args, f"{prefix}_exp")
    explicit = getattr(args, f"{prefix}_{split}_code")
    if explicit:
        return explicit
    if not root:
        raise ValueError(f"Need --{prefix}_exp or --{prefix}_{split}_code")
    return str(Path(root) / "codewords" / f"{split}_code.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_exp", default=None)
    parser.add_argument("--target_exp", default=None)
    for prefix in ("source", "target"):
        for split in ("train", "val", "test"):
            parser.add_argument(f"--{prefix}_{split}_code", default=None)
    parser.add_argument("--train_csi", required=True)
    parser.add_argument("--val_csi", required=True)
    parser.add_argument("--test_csi", required=True)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--decoder_args_json", default=None)
    parser.add_argument("--encoder_checkpoint", default=None)
    parser.add_argument("--encoder_args_json", default=None)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--mapper_type", default="affine_residual_mlp",
                        choices=[
                            "affine_residual_mlp",
                            "affine_residual_mlp_attention",
                            "affine_iterative_residual",
                            "affine_iterative_residual_unshared",
                            "affine_sens_weighted_residual",
                            "affine_film_residual_mlp",
                            "affine_multiscale_residual_mlp",
                            "affine_lowrank_residual",
                            "affine_bottleneck_residual",
                            "affine_group_gated",
                            "affine_token_mixer",
                            "affine_tiny_transformer",
                            "affine_moe_bottleneck",
                            "affine_coupling_flow",
                            "affine_whole_residual_mlp",
                            "affine_whole_direct_mlp",
                            "legacy_mlp_adapter",
                            "affine_linear",
                            "direct_mlp",
                        ])
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--lowrank_rank", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--num_groups", type=int, default=16)
    parser.add_argument("--group_hidden", type=int, default=64)
    parser.add_argument("--gate_hidden", type=int, default=64)
    parser.add_argument("--gate_init", type=float, default=0.5)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--token_hidden", type=int, default=64)
    parser.add_argument("--channel_hidden", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--transformer_ffn_dim", type=int, default=128)
    parser.add_argument("--attention_dim", type=int, default=32)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--attention_scale", type=float, default=0.1)
    parser.add_argument(
        "--attention_input",
        default="value_delta",
        choices=["value", "delta", "value_delta"])
    parser.add_argument(
        "--attention_use_position",
        action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=128)
    parser.add_argument("--whole_mlp_dims", type=parse_int_list, default=None)
    parser.add_argument("--whole_mlp_activation", default="gelu")
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--learnable_residual_gate", action="store_true")
    parser.add_argument("--gate_max", type=float, default=0.5)
    parser.add_argument(
        "--gate_mode",
        default="block",
        choices=[
            "block",
            "none",
            "final_static",
            "final_adaptive",
            "final_unbounded",
        ])
    parser.add_argument("--final_gate_max", type=float, default=1.0)
    parser.add_argument("--final_gate_init", type=float, default=1.0)
    parser.add_argument("--adaptive_gate_hidden", type=int, default=128)
    parser.add_argument("--gate_l1", type=float, default=0.0)
    parser.add_argument("--no_block_norm", action="store_true")
    parser.add_argument("--use_final_norm", action="store_true")
    parser.add_argument("--train_affine", action="store_true")
    parser.add_argument(
        "--no_affine_alignment", action="store_true",
        help=("For legacy_mlp_adapter, bypass the fitted affine alignment and "
              "apply the residual MLP directly to source codes."))
    parser.add_argument("--align_ridge", type=float, default=1.0)
    parser.add_argument(
        "--affine_fit_splits",
        choices=["train", "train_val_test"],
        default="train",
        help=("Code splits used only to fit the initial affine alignment. "
              "train_val_test is an oracle diagnostic because it consumes "
              "validation and test target codes."))
    parser.add_argument("--lambda_code", type=float, default=1.0)
    parser.add_argument("--lambda_recon", type=float, default=0.0)
    parser.add_argument("--lambda_feature", type=float, default=0.0)
    parser.add_argument("--lambda_encoder_consistency", type=float, default=0.0)
    parser.add_argument("--lambda_delta_norm", type=float, default=0.0)
    parser.add_argument("--lambda_teacher_code", type=float, default=0.0)
    parser.add_argument("--teacher_train_code", default=None)
    parser.add_argument("--lambda_fisher", type=float, default=0.0)
    parser.add_argument("--fisher_basis_path", default=None)
    parser.add_argument("--fisher_rank", type=int, default=0)
    parser.add_argument("--fisher_weight_power", type=float, default=0.5)
    parser.add_argument("--fisher_weight_max", type=float, default=4.0)
    parser.add_argument(
        "--gradient_diagnostics_every", type=int, default=0,
        help="Log first-batch weighted loss gradient norms/cosines every N epochs")
    parser.add_argument(
        "--train_last_blocks", type=int, default=0,
        help="Freeze the mapper except for its last N residual blocks; 0 trains all")
    parser.add_argument("--code_noise_std", type=float, default=0.0)
    parser.add_argument("--stage1_epochs", type=int, default=0)
    parser.add_argument("--stage1_code_noise_std", type=float, default=0.0)
    parser.add_argument("--stage1_lambda_recon", type=float, default=0.0)
    parser.add_argument(
        "--stage1_lambda_encoder_consistency", type=float, default=0.0)
    parser.add_argument("--stage2_lr", type=float, default=None)
    parser.add_argument("--stage2_affine_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--stage2_affine_freeze_epochs", type=int, default=0)
    parser.add_argument("--stage2_recon_warmup_epochs", type=int, default=0)
    parser.add_argument("--stage2_encoder_delay_epochs", type=int, default=0)
    parser.add_argument("--stage2_encoder_warmup_epochs", type=int, default=0)
    parser.add_argument("--stage2_noise_decay_epochs", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--ema_start_epoch", type=int, default=1)
    parser.add_argument("--ema_update_every", type=int, default=1)
    parser.add_argument("--init_mapper_checkpoint", default=None)
    parser.add_argument("--init_mapper_use_ema", action="store_true")
    parser.add_argument(
        "--encoder_consistency_target",
        default="mapped",
        choices=["mapped", "target"])
    parser.add_argument("--code_loss_type", default="mse",
                        choices=[
                            "mse",
                            "clipped_std_mse",
                            "clipped_var_mse",
                            "clipped_power_mse",
                            "clipped_residual_std_mse",
                            "decoder_sensitivity_mse",
                            "decoder_jac_residual_mse",
                        ])
    parser.add_argument(
        "--sensitivity_source",
        default="jacobian",
        choices=["jacobian", "fc_decoder"],
        help="How to build decoder sensitivity weights for "
             "decoder_sensitivity_mse / decoder_jac_residual_mse")
    parser.add_argument(
        "--sensitivity_power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized sensitivity before clipping")
    parser.add_argument(
        "--sensitivity_hutchinson",
        type=int,
        default=8,
        help="Hutchinson probes for jacobian sensitivity")
    parser.add_argument(
        "--sensitivity_probe_samples",
        type=int,
        default=2048,
        help="Number of target codes used to estimate jacobian sensitivity")
    parser.add_argument("--std_weight_min", type=float, default=0.25)
    parser.add_argument("--std_weight_max", type=float, default=4.0)
    parser.add_argument("--std_weight_eps", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["const", "cosine"], default="cosine")
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--export_codewords", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--decoder", default="transnet")
    parser.add_argument("--encoder", default="transnet")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--decoder_num_blocks", type=int, default=2)
    return parser.parse_args()


def run_training(args, preloaded_data=None):
    if args.stage1_epochs < 0 or args.stage1_epochs >= args.epochs:
        if args.stage1_epochs != 0:
            raise ValueError("stage1_epochs must satisfy 0 <= stage1_epochs < epochs")
    for name in (
            "stage2_affine_freeze_epochs",
            "stage2_recon_warmup_epochs",
            "stage2_encoder_delay_epochs",
            "stage2_encoder_warmup_epochs",
            "stage2_noise_decay_epochs"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
    if args.ema_start_epoch <= 0:
        raise ValueError("ema_start_epoch must be positive")
    if args.ema_update_every <= 0:
        raise ValueError("ema_update_every must be positive")
    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    codeword_dir = exp_dir / "codewords"
    tensorboard_dir = exp_dir / "tensorboard"
    for path in (checkpoint_dir, codeword_dir, tensorboard_dir):
        path.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    (exp_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8")
    writer = SummaryWriter(str(tensorboard_dir))
    add_text_json(writer, "config/args", vars(args))

    set_seed(args.seed)
    device = resolve_device(args.gpu, args.cpu)
    log_experiment_header(args, exp_dir=exp_dir, target_logger=logger)
    logger.info("=> Device: %s", device)

    decoder, decoder_cfg = load_decoder(args, device)
    if args.code_loss_type in (
            "decoder_sensitivity_mse", "decoder_jac_residual_mse"):
        source = getattr(args, "sensitivity_source", "jacobian")
        if source == "fc_decoder":
            sensitivity = estimate_decoder_fc_sensitivity(decoder)
            logger.info(
                "=> Decoder sensitivity from fc_decoder: min=%.4f max=%.4f std=%.4f",
                float(sensitivity.min()), float(sensitivity.max()),
                float(sensitivity.std()))
        else:
            # Probe with target train codes after they are loaded below if needed;
            # use a temporary load of target train codes for estimation.
            probe_path = None
            if args.target_train_code:
                probe_path = args.target_train_code
            elif args.target_exp:
                probe_path = str(
                    Path(args.target_exp) / "codewords" / "train_code.pt")
            if probe_path is None or not Path(probe_path).exists():
                raise FileNotFoundError(
                    "jacobian sensitivity needs target train codes "
                    f"(tried {probe_path})")
            probe = load_code(
                probe_path, max_samples=args.sensitivity_probe_samples)
            sensitivity = estimate_decoder_jacobian_sensitivity(
                decoder,
                probe,
                device,
                n_hutchinson=args.sensitivity_hutchinson,
                max_samples=args.sensitivity_probe_samples,
            )
            logger.info(
                "=> Decoder jacobian sensitivity (Hutchinson=%d, n=%d): "
                "min=%.4f max=%.4f std=%.4f",
                args.sensitivity_hutchinson,
                min(probe.size(0), args.sensitivity_probe_samples),
                float(sensitivity.min()), float(sensitivity.max()),
                float(sensitivity.std()))
        args._decoder_sensitivity_weight = sensitivity
        torch.save(
            {
                "sensitivity": sensitivity,
                "source": source,
                "power": args.sensitivity_power,
            },
            exp_dir / "decoder_sensitivity.pt")
    target_encoder = None
    if (args.lambda_encoder_consistency > 0
            or args.stage1_lambda_encoder_consistency > 0):
        target_encoder = load_target_encoder(args, device)
        logger.info(
            "=> Loaded frozen target encoder for consistency loss: "
            "lambda=%s target=%s",
            args.lambda_encoder_consistency,
            args.encoder_consistency_target)
    channel, nt, nc = decoder_cfg["channel"], decoder_cfg["nt"], decoder_cfg["nc"]

    datasets = {}
    for split in ("train", "val", "test"):
        if preloaded_data is None:
            max_samples = args.max_train_samples if split == "train" else args.max_eval_samples
            source = load_code(split_paths(args, "source", split), max_samples)
            target = load_code(split_paths(args, "target", split), max_samples)
            csi_path = getattr(args, f"{split}_csi")
            csi = load_csi(csi_path, channel, nt, nc, max_samples)
        else:
            source, target, csi = preloaded_data[split]
        datasets[split] = CodeCsiDataset(source, target, csi)
        logger.info(
            "=> %s dataset: source=%s target=%s csi=%s n=%d",
            split,
            tuple(source.shape),
            tuple(target.shape),
            tuple(csi.shape),
            len(datasets[split]))
        writer.add_scalar(f"data/{split}_samples", len(datasets[split]), 0)
        writer.add_scalar(f"data/{split}_code_dim", source.size(1), 0)

    if args.affine_fit_splits == "train":
        affine_splits = ("train",)
    elif args.affine_fit_splits == "train_val_test":
        affine_splits = ("train", "val", "test")
    else:
        raise ValueError(f"Unknown affine_fit_splits={args.affine_fit_splits}")
    affine_source = torch.cat(
        [datasets[split].source for split in affine_splits], dim=0)
    affine_target = torch.cat(
        [datasets[split].target for split in affine_splits], dim=0)
    logger.info(
        "=> Fitting affine alignment on splits=%s n=%d ridge=%s%s",
        ",".join(affine_splits), len(affine_source), args.align_ridge,
        " [ORACLE: val/test target codes used]"
        if args.affine_fit_splits == "train_val_test" else "")
    weight, bias = fit_affine(
        affine_source, affine_target, ridge=args.align_ridge)
    torch.save({"weight": weight, "bias": bias}, exp_dir / "affine_alignment.pt")
    writer.add_scalar("affine/fit_samples", len(affine_source), 0)
    writer.add_text(
        "affine/fit_splits", ",".join(affine_splits), 0)
    writer.add_scalar("affine/weight_norm", float(weight.norm().item()), 0)
    writer.add_scalar("affine/bias_norm", float(bias.norm().item()), 0)
    writer.add_scalar("affine/weight_mean", float(weight.mean().item()), 0)
    writer.add_scalar("affine/weight_std", float(weight.std().item()), 0)
    writer.add_scalar("affine/bias_mean", float(bias.mean().item()), 0)
    writer.add_scalar("affine/bias_std", float(bias.std().item()), 0)
    mapper = build_mapper(
        args.mapper_type,
        weight,
        bias,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        use_block_norm=not args.no_block_norm,
        use_final_norm=args.use_final_norm,
        train_affine=args.train_affine,
        learnable_residual_gate=args.learnable_residual_gate,
        gate_max=args.gate_max,
        gate_mode=args.gate_mode,
        final_gate_max=args.final_gate_max,
        final_gate_init=args.final_gate_init,
        adaptive_gate_hidden=args.adaptive_gate_hidden,
        bottleneck_dim=args.bottleneck_dim,
        num_groups=args.num_groups,
        group_hidden=args.group_hidden,
        gate_hidden=args.gate_hidden,
        gate_init=args.gate_init,
        num_tokens=args.num_tokens,
        token_hidden=args.token_hidden,
        channel_hidden=args.channel_hidden,
        num_heads=args.num_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        attention_dim=args.attention_dim,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
        attention_scale=args.attention_scale,
        attention_input=args.attention_input,
        attention_use_position=args.attention_use_position,
        num_experts=args.num_experts,
        flow_hidden_dim=args.flow_hidden_dim,
        whole_mlp_dims=args.whole_mlp_dims,
        whole_mlp_activation=args.whole_mlp_activation,
        use_affine_alignment=not args.no_affine_alignment,
        lowrank_rank=args.lowrank_rank).to(device)
    if args.init_mapper_checkpoint:
        checkpoint = torch.load(
            args.init_mapper_checkpoint, weights_only=False, map_location="cpu")
        if args.init_mapper_use_ema:
            if "ema" not in checkpoint or "shadow" not in checkpoint["ema"]:
                raise RuntimeError(
                    "--init_mapper_use_ema requested but checkpoint has no ema.shadow")
            state_dict = checkpoint["ema"]["shadow"]
            logger.info(
                "=> Loading mapper initialization from EMA shadow: %s",
                args.init_mapper_checkpoint)
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)
            logger.info(
                "=> Loading mapper initialization from state_dict: %s",
                args.init_mapper_checkpoint)
        missing, unexpected = mapper.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"mapper checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    if args.train_last_blocks > 0:
        if not hasattr(mapper, "blocks"):
            raise AttributeError("--train_last_blocks requires mapper.blocks")
        if args.train_last_blocks > len(mapper.blocks):
            raise ValueError(
                f"train_last_blocks={args.train_last_blocks} exceeds "
                f"num_blocks={len(mapper.blocks)}")
        for parameter in mapper.parameters():
            parameter.requires_grad_(False)
        for block in mapper.blocks[-args.train_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        logger.info(
            "=> Frozen mapper except last %d/%d residual blocks",
            args.train_last_blocks, len(mapper.blocks))
    log_parameter_table(mapper, logger)
    total_params, trainable_params = count_parameters(mapper)
    writer.add_scalar("model/total_params", total_params, 0)
    writer.add_scalar("model/trainable_params", trainable_params, 0)
    writer.add_scalar("model/frozen_params", total_params - trainable_params, 0)
    add_text_json(writer, "config/model", {
        "mapper_type": args.mapper_type,
        "hidden_dim": args.hidden_dim,
        "lowrank_rank": args.lowrank_rank,
        "bottleneck_dim": args.bottleneck_dim,
        "num_groups": args.num_groups,
        "group_hidden": args.group_hidden,
        "gate_hidden": args.gate_hidden,
        "gate_init": args.gate_init,
        "num_tokens": args.num_tokens,
        "token_hidden": args.token_hidden,
        "channel_hidden": args.channel_hidden,
        "num_heads": args.num_heads,
        "transformer_ffn_dim": args.transformer_ffn_dim,
        "attention_dim": args.attention_dim,
        "attention_heads": args.attention_heads,
        "attention_dropout": args.attention_dropout,
        "attention_scale": args.attention_scale,
        "attention_input": args.attention_input,
        "attention_use_position": args.attention_use_position,
        "num_experts": args.num_experts,
        "flow_hidden_dim": args.flow_hidden_dim,
        "whole_mlp_dims": args.whole_mlp_dims,
        "whole_mlp_activation": args.whole_mlp_activation,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
        "residual_scale": args.residual_scale,
        "learnable_residual_gate": args.learnable_residual_gate,
        "gate_max": args.gate_max,
        "gate_mode": args.gate_mode,
        "final_gate_max": args.final_gate_max,
        "final_gate_init": args.final_gate_init,
        "adaptive_gate_hidden": args.adaptive_gate_hidden,
        "gate_l1": args.gate_l1,
        "use_block_norm": not args.no_block_norm,
        "use_final_norm": args.use_final_norm,
        "train_affine": args.train_affine,
        "use_affine_alignment": not args.no_affine_alignment,
        "affine_fit_splits": args.affine_fit_splits,
        "affine_fit_samples": len(affine_source),
        "lambda_encoder_consistency": args.lambda_encoder_consistency,
        "encoder_consistency_target": args.encoder_consistency_target,
        "lambda_delta_norm": args.lambda_delta_norm,
        "code_noise_std": args.code_noise_std,
        "total_params": total_params,
        "trainable_params": trainable_params,
    })

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda"),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda"),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda"),
    }
    eval_loaders = {
        key: DataLoader(
            Subset(dataset, range(min(len(dataset), args.max_eval_samples)))
            if args.max_eval_samples and key != "train" else dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda")
        for key, dataset in datasets.items()
    }

    train_affine_start = datasets["train"].source.matmul(weight) + bias
    train_residual = datasets["train"].target - train_affine_start
    std_weight_cpu, loss_scale = build_code_loss_weight(
        datasets["train"].target, args, train_residual)
    std_weight = std_weight_cpu.to(device) if std_weight_cpu is not None else None
    teacher_train_code = load_optional_code(
        args.teacher_train_code, args.max_train_samples)
    if args.lambda_teacher_code > 0:
        if teacher_train_code is None:
            raise ValueError(
                "--teacher_train_code is required when lambda_teacher_code > 0")
        if teacher_train_code.shape != datasets["train"].target.shape:
            raise ValueError(
                "teacher_train_code shape mismatch: "
                f"{tuple(teacher_train_code.shape)} vs "
                f"{tuple(datasets['train'].target.shape)}")
        logger.info("=> Loaded teacher train code: %s",
                    tuple(teacher_train_code.shape))
    fisher_basis = None
    fisher_weight = None
    fisher_eval_basis = None
    fisher_eval_eigenvalues = None
    if args.lambda_fisher > 0 and not args.fisher_basis_path:
        raise ValueError(
            "--fisher_basis_path is required when lambda_fisher > 0")
    if args.fisher_basis_path:
        fisher_state = torch.load(
            args.fisher_basis_path, weights_only=True, map_location="cpu")
        if not isinstance(fisher_state, dict) or "eigenvectors" not in fisher_state:
            raise ValueError(
                "fisher basis file must contain eigenvectors and eigenvalues")
        eigenvectors = fisher_state["eigenvectors"].float()
        eigenvalues = fisher_state.get("eigenvalues")
        fisher_eval_basis = eigenvectors.contiguous().to(device)
        if eigenvalues is not None:
            fisher_eval_eigenvalues = eigenvalues.float().contiguous().to(device)
        rank = args.fisher_rank or eigenvectors.size(1)
        if rank <= 0 or rank > eigenvectors.size(1):
            raise ValueError(
                f"invalid fisher_rank={rank} for {tuple(eigenvectors.shape)}")
        if args.lambda_fisher > 0:
            fisher_basis = eigenvectors[:, :rank].contiguous().to(device)
        if (args.lambda_fisher > 0 and eigenvalues is not None
                and args.fisher_weight_power != 0):
            fisher_weight = eigenvalues[:rank].float().clamp_min(
                args.std_weight_eps)
            fisher_weight = fisher_weight / fisher_weight.mean().clamp_min(1e-12)
            fisher_weight = fisher_weight.pow(args.fisher_weight_power)
            fisher_weight = fisher_weight / fisher_weight.mean().clamp_min(1e-12)
            fisher_weight = fisher_weight.clamp(max=args.fisher_weight_max).to(device)
        logger.info(
            "=> Fisher diagnostics/loss: path=%s eval_rank=%d train_rank=%d "
            "lambda=%s power=%s weight=[%.4f, %.4f]",
            args.fisher_basis_path, eigenvectors.size(1), rank, args.lambda_fisher,
            args.fisher_weight_power,
            float(fisher_weight.min()) if fisher_weight is not None else 1.0,
            float(fisher_weight.max()) if fisher_weight is not None else 1.0)
    if std_weight_cpu is not None:
        torch.save({
            "std_weight": std_weight_cpu,
            "loss_scale": loss_scale,
            "code_loss_type": args.code_loss_type,
            "std_weight_min": args.std_weight_min,
            "std_weight_max": args.std_weight_max,
            "std_weight_eps": args.std_weight_eps,
        }, exp_dir / "code_loss_weight.pt")
        weight_stats = {
            "mean": float(std_weight_cpu.mean()),
            "std": float(std_weight_cpu.std()),
            "min": float(std_weight_cpu.min()),
            "max": float(std_weight_cpu.max()),
            "scale_mean": float(loss_scale.mean()),
            "scale_min": float(loss_scale.min()),
            "scale_max": float(loss_scale.max()),
        }
        for key, value in weight_stats.items():
            writer.add_scalar(f"code_loss_weight/{key}", value, 0)
        logger.info("=> Code loss weight stats: %s", weight_stats)

    optimizer = build_optimizer(mapper, args.lr, args.weight_decay)
    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(
            mapper, decay=args.ema_decay,
            update_every=args.ema_update_every)
        logger.info(
            "=> EMA enabled: decay=%s start_epoch=%d update_every=%d",
            args.ema_decay, args.ema_start_epoch, args.ema_update_every)
    first_stage_epochs = args.stage1_epochs or args.epochs
    scheduler = build_scheduler(
        optimizer, args.scheduler, first_stage_epochs, len(loaders["train"]),
        args.eta_min)
    logger.info(
        "=> Optimizer: AdamW lr=%s weight_decay=%s scheduler=%s eta_min=%s",
        args.lr, args.weight_decay, args.scheduler, args.eta_min)
    logger.info(
        "=> Loss: type=%s lambda_code=%s lambda_feature=%s lambda_recon=%s "
        "lambda_encoder_consistency=%s encoder_consistency_target=%s "
        "lambda_delta_norm=%s lambda_teacher_code=%s code_noise_std=%s",
        args.code_loss_type, args.lambda_code, args.lambda_feature,
        args.lambda_recon, args.lambda_encoder_consistency,
        args.encoder_consistency_target, args.lambda_delta_norm,
        args.lambda_teacher_code,
        args.code_noise_std)
    writer.add_scalar("loss_weights/code_loss_type_is_weighted",
                      float(args.code_loss_type != "mse"), 0)
    writer.add_scalar("loss_weights/lambda_code", args.lambda_code, 0)
    writer.add_scalar("loss_weights/lambda_feature", args.lambda_feature, 0)
    writer.add_scalar("loss_weights/lambda_recon", args.lambda_recon, 0)
    writer.add_scalar(
        "loss_weights/lambda_encoder_consistency",
        args.lambda_encoder_consistency,
        0)
    writer.add_scalar("loss_weights/gate_l1", args.gate_l1, 0)
    writer.add_scalar("loss_weights/lambda_delta_norm", args.lambda_delta_norm, 0)
    writer.add_scalar("loss_weights/lambda_teacher_code", args.lambda_teacher_code, 0)
    writer.add_scalar("loss_weights/lambda_fisher", args.lambda_fisher, 0)
    writer.add_scalar("regularization/code_noise_std", args.code_noise_std, 0)

    best = {
        "val_code_mse": {"metric": math.inf, "epoch": 0},
        "val_decoder_nmse": {"metric": math.inf, "epoch": 0},
    }
    history = []

    def save_checkpoint(name, epoch, metrics):
        payload = {
            "epoch": epoch,
            "state_dict": mapper.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        }
        if ema is not None:
            payload["ema"] = ema.state_dict()
        torch.save(payload, checkpoint_dir / name)

    for epoch in range(1, args.epochs + 1):
        if args.stage1_epochs > 0 and epoch == args.stage1_epochs + 1:
            stage1_path = checkpoint_dir / "best_code_mse.pth"
            if not stage1_path.exists():
                raise RuntimeError(
                    "two-stage training requires a stage-1 best_code_mse "
                    "checkpoint; check eval_every and stage1_epochs")
            checkpoint = torch.load(
                stage1_path, weights_only=True, map_location=device)
            mapper.load_state_dict(checkpoint["state_dict"])
            if ema is not None:
                ema.reset(mapper)
            stage1_best = dict(best["val_code_mse"])
            torch.save(checkpoint, checkpoint_dir / "stage1_best_code_mse.pth")
            best["stage1_val_code_mse"] = stage1_best
            best["val_decoder_nmse"] = {"metric": math.inf, "epoch": 0}

            freeze_affine = args.stage2_affine_freeze_epochs > 0
            for name, param in mapper.named_parameters():
                if name in ("alignment_weight", "alignment_bias"):
                    param.requires_grad_(not freeze_affine)
            stage2_lr = args.stage2_lr or args.lr
            optimizer = build_optimizer(
                mapper, stage2_lr, args.weight_decay,
                affine_lr_multiplier=args.stage2_affine_lr_multiplier)
            scheduler = build_scheduler(
                optimizer, args.scheduler,
                args.epochs - args.stage1_epochs,
                len(loaders["train"]), args.eta_min)
            logger.info(
                "=> Stage 2 starts from stage-1 best code checkpoint: "
                "epoch=%d code_mse=%.6e lr=%s affine_lr_multiplier=%s "
                "affine_frozen=%s",
                checkpoint["epoch"], stage1_best["metric"], stage2_lr,
                args.stage2_affine_lr_multiplier, freeze_affine)

        unfreeze_epoch = (
            args.stage1_epochs + args.stage2_affine_freeze_epochs + 1)
        if (args.stage1_epochs > 0
                and args.stage2_affine_freeze_epochs > 0
                and epoch == unfreeze_epoch):
            for name, param in mapper.named_parameters():
                if name in ("alignment_weight", "alignment_bias"):
                    param.requires_grad_(True)
            stage2_lr = args.stage2_lr or args.lr
            optimizer = build_optimizer(
                mapper, stage2_lr, args.weight_decay,
                affine_lr_multiplier=args.stage2_affine_lr_multiplier)
            scheduler = build_scheduler(
                optimizer, args.scheduler, args.epochs - epoch + 1,
                len(loaders["train"]), args.eta_min)
            logger.info(
                "=> Stage 2 affine parameters unfrozen at epoch %d; "
                "affine lr=%s", epoch,
                stage2_lr * args.stage2_affine_lr_multiplier)

        stage_values = stage_hyperparameters(args, epoch)
        if ema is not None and epoch == args.ema_start_epoch:
            ema.reset(mapper)
        active_ema = (
            ema if ema is not None and epoch >= args.ema_start_epoch else None)
        train_metrics = train_epoch(
            mapper,
            loaders["train"],
            decoder,
            device,
            optimizer,
            scheduler,
            args.lambda_code,
            stage_values["lambda_recon"],
            args.lambda_feature,
            args.code_loss_type,
            std_weight,
            args.gate_l1,
            target_encoder,
            stage_values["lambda_encoder_consistency"],
            args.encoder_consistency_target,
            stage_values["code_noise_std"],
            args.lambda_delta_norm,
            active_ema,
            teacher_train_code,
            args.lambda_teacher_code,
            fisher_basis,
            fisher_weight,
            args.lambda_fisher,
            args.gradient_diagnostics_every > 0
            and epoch % args.gradient_diagnostics_every == 0)
        record = {"epoch": epoch, "lr": scheduler.get_lr()[0],
                  "stage": stage_values,
                  "eval_weights": "ema" if active_ema is not None else "raw",
                  "train": train_metrics}
        log_metrics(writer, "train", train_metrics, epoch)
        writer.add_scalar("train/lr", scheduler.get_lr()[0], epoch)
        writer.add_scalar("schedule/stage", stage_values["stage"], epoch)
        writer.add_scalar(
            "schedule/lambda_recon", stage_values["lambda_recon"], epoch)
        writer.add_scalar(
            "schedule/lambda_encoder_consistency",
            stage_values["lambda_encoder_consistency"], epoch)
        writer.add_scalar(
            "schedule/code_noise_std", stage_values["code_noise_std"], epoch)

        if args.eval_every and epoch % args.eval_every == 0:
            if active_ema is not None:
                active_ema.apply(mapper)
            for split in ("val", "test"):
                metrics = evaluate(
                    mapper,
                    eval_loaders[split],
                    decoder,
                    device,
                    args.code_loss_type,
                    std_weight,
                    target_encoder,
                    args.encoder_consistency_target,
                    fisher_eval_basis,
                    fisher_eval_eigenvalues)
                record[split] = metrics
                log_metrics(writer, split, metrics, epoch)
                logger.info(
                    "Epoch [%d/%d] %s code_loss=%.6e code_mse=%.6e code_nmse=%.3fdB "
                    "cos=%.6f decoder_mse=%.6e decoder_nmse=%.3fdB n=%d",
                    epoch, args.epochs, split, metrics["code_loss"],
                    metrics["code_mse"], metrics["code_nmse"], metrics["code_cos"],
                    metrics["decoder_mse"], metrics["decoder_nmse"],
                    metrics["n"])
                logger.info(
                    "Epoch [%d/%d] %s_metrics=%s",
                    epoch,
                    args.epochs,
                    split,
                    json.dumps(metrics, sort_keys=True))
            val_metrics = record["val"]
            param_stats = collect_parameter_stats(mapper)
            param_summary = summarize_parameter_stats(param_stats)
            record["param_stats"] = param_stats
            record["param_summary"] = param_summary
            log_metrics(writer, "params", param_stats, epoch)
            log_metrics(writer, "param_summary", param_summary, epoch)
            logger.info(
                "Epoch [%d/%d] param_summary=%s",
                epoch,
                args.epochs,
                json.dumps(param_summary, sort_keys=True))
            logger.info(
                "Epoch [%d/%d] param_stats=%s",
                epoch,
                args.epochs,
                json.dumps(param_stats, sort_keys=True))
            if val_metrics["code_mse"] < best["val_code_mse"]["metric"]:
                best["val_code_mse"] = {
                    "metric": val_metrics["code_mse"],
                    "epoch": epoch,
                    "metrics": val_metrics,
                }
                save_checkpoint("best_code_mse.pth", epoch, val_metrics)
                writer.add_scalar(
                    "best/val_code_mse", val_metrics["code_mse"], epoch)
                writer.add_scalar(
                    "best/val_code_nmse", val_metrics["code_nmse"], epoch)
            decoder_selection_enabled = (
                args.stage1_epochs <= 0 or epoch > args.stage1_epochs)
            if (decoder_selection_enabled
                    and val_metrics["decoder_nmse"]
                    < best["val_decoder_nmse"]["metric"]):
                best["val_decoder_nmse"] = {
                    "metric": val_metrics["decoder_nmse"],
                    "epoch": epoch,
                    "metrics": val_metrics,
                }
                save_checkpoint("best_decoder_nmse.pth", epoch, val_metrics)
                writer.add_scalar(
                    "best/val_decoder_nmse",
                    val_metrics["decoder_nmse"],
                    epoch)
                writer.add_scalar(
                    "best/val_decoder_mse",
                    val_metrics["decoder_mse"],
                    epoch)
            if active_ema is not None:
                active_ema.restore(mapper)

        if hasattr(mapper, "get_metrics"):
            adapter_metrics = mapper.get_metrics()
            record["adapter"] = adapter_metrics
            log_metrics(writer, "adapter", adapter_metrics, epoch)

        history.append(record)
        adapter_msg = ""
        if record.get("adapter"):
            keep = [
                "adapter/delta_ratio",
                "adapter/gate_mean_avg",
                "adapter/gate_max_max",
            ]
            adapter_msg = " " + " ".join(
                f"{key.split('/')[-1]}={record['adapter'][key]:.6e}"
                for key in keep
                if key in record["adapter"])
        logger.info(
            "Epoch [%d/%d] lr=%.6e train_loss=%.6e train_code_loss=%.6e "
            "train_code_mse=%.6e train_code_nmse=%.3fdB train_cos=%.6f "
            "train_recon_mse=%.6e train_decoder_nmse=%.3fdB "
            "delta_target_cos=%.6f residual_coverage=%.6f%s",
            epoch, args.epochs, scheduler.get_lr()[0],
            train_metrics["loss"], train_metrics["code_loss"],
            train_metrics["code_mse"], train_metrics["code_nmse"],
            train_metrics["code_cos"], train_metrics["recon_mse"],
            train_metrics["decoder_nmse"],
            train_metrics.get("delta_target_cos", 0.0),
            train_metrics.get("residual_coverage", 0.0),
            adapter_msg)
        logger.info(
            "Epoch [%d/%d] train_metrics=%s",
            epoch,
            args.epochs,
            json.dumps(train_metrics, sort_keys=True))

    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps(best, indent=2))
    add_text_json(writer, "result/best", best, args.epochs)
    if math.isfinite(best["val_code_mse"]["metric"]):
        writer.add_scalar(
            "result/best_val_code_mse",
            best["val_code_mse"]["metric"],
            best["val_code_mse"]["epoch"])
    if math.isfinite(best["val_decoder_nmse"]["metric"]):
        writer.add_scalar(
            "result/best_val_decoder_nmse",
            best["val_decoder_nmse"]["metric"],
            best["val_decoder_nmse"]["epoch"])
    save_checkpoint("last.pth", args.epochs, history[-1] if history else {})

    if args.export_codewords:
        ckpt_path = checkpoint_dir / "best_decoder_nmse.pth"
        if not ckpt_path.exists():
            ckpt_path = checkpoint_dir / "best_code_mse.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
            mapper.load_state_dict(ckpt["state_dict"])
            logger.info("=> Exporting mapped codewords from %s", ckpt_path)
        for split in ("train", "val", "test"):
            shape = export_mapped_code(
                mapper,
                datasets[split],
                device,
                codeword_dir / f"{split}_mapped_code.pt",
                args.batch_size,
                args.workers)
            logger.info("=> Saved %s mapped codewords %s", split, shape)

    writer.flush()
    writer.close()
    logger.info("=> Best val_code_mse: %s", best["val_code_mse"])
    logger.info("=> Best val_decoder_nmse: %s", best["val_decoder_nmse"])


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
