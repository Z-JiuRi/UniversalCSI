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


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr)


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


def build_code_loss_weight(target_code, args, residual_code=None):
    if args.code_loss_type not in (
            "clipped_std_mse",
            "clipped_var_mse",
            "clipped_power_mse",
            "clipped_residual_std_mse"):
        return None, None
    if args.code_loss_type == "clipped_residual_std_mse":
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
            "clipped_residual_std_mse"):
        if std_weight is None:
            raise ValueError(f"std_weight is required for {code_loss_type}")
        weighted = sqerr * std_weight.view(1, -1)
        return weighted.mean(), raw_mse
    raise ValueError(f"Unknown code_loss_type: {code_loss_type}")


def train_epoch(model, loader, decoder, device, optimizer, scheduler,
                lambda_code, lambda_recon, lambda_feature=0.0,
                code_loss_type="mse", std_weight=None, gate_l1=0.0,
                target_encoder=None, lambda_encoder_consistency=0.0,
                encoder_consistency_target="mapped"):
    model.train()
    decoder.eval()
    total = {
        "loss": 0.0,
        "code_loss": 0.0,
        "code_mse": 0.0,
        "feature_mse": 0.0,
        "encoder_consistency_mse": 0.0,
        "gate_l1": 0.0,
        "recon_mse": 0.0,
        "cos": 0.0,
        "n": 0,
    }
    code_err = torch.tensor(0.0, device=device)
    code_power = torch.tensor(0.0, device=device)
    recon_err = torch.tensor(0.0, device=device)
    recon_power = torch.tensor(0.0, device=device)
    for source, target, csi, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        csi = csi.to(device, non_blocking=True)
        mapped = model(source)
        code_loss, code_mse = compute_code_loss(
            mapped, target, code_loss_type, std_weight)
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
                + lambda_feature * feature_loss
                + lambda_recon * recon_loss
                + lambda_encoder_consistency * encoder_consistency_loss
                + gate_l1 * gate_reg)
        else:
            with torch.no_grad():
                recon = decoder(mapped.detach())
                recon_loss = F.mse_loss(recon, csi)
            loss = (
                lambda_code * code_loss
                + lambda_feature * feature_loss
                + gate_l1 * gate_reg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
        total["encoder_consistency_mse"] += float(
            encoder_consistency_loss.detach().cpu()) * n
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
    return metrics


@torch.no_grad()
def evaluate(model, loader, decoder, device, code_loss_type="mse",
             std_weight=None, target_encoder=None,
             encoder_consistency_target="mapped"):
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
    for source, target, csi, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        csi = csi.to(device, non_blocking=True)
        mapped = model(source)
        recon = decoder(mapped)
        if hasattr(decoder, "fc_decoder"):
            feature_mse = F.mse_loss(
                decoder.fc_decoder(mapped),
                decoder.fc_decoder(target))
        else:
            feature_mse = mapped.new_tensor(0.0)
        if target_encoder is not None:
            reencoded = target_encoder(recon)
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
        n = source.size(0)
        cerr = mapped - target
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
        recon_mse_sum += float(rerr.pow(2).mean().cpu()) * n
        cos_sum += float(F.cosine_similarity(mapped, target, dim=1).mean().cpu()) * n
        n_total += n
    return {
        "code_loss": code_loss_sum / max(n_total, 1),
        "code_mse": code_mse_sum / max(n_total, 1),
        "feature_mse": feature_mse_sum / max(n_total, 1),
        "encoder_consistency_mse": encoder_consistency_sum / max(n_total, 1),
        "code_nmse": float((10.0 * torch.log10(
            code_err / code_power.clamp_min(1e-12))).cpu()),
        "code_cos": cos_sum / max(n_total, 1),
        "decoder_mse": recon_mse_sum / max(n_total, 1),
        "decoder_nmse": float((10.0 * torch.log10(
            recon_err / recon_power.clamp_min(1e-12))).cpu()),
        "n": n_total,
    }


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
                            "affine_lowrank_residual",
                            "affine_linear",
                            "direct_mlp",
                        ])
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--lowrank_rank", type=int, default=64)
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
    parser.add_argument("--align_ridge", type=float, default=1.0)
    parser.add_argument("--lambda_code", type=float, default=1.0)
    parser.add_argument("--lambda_recon", type=float, default=0.0)
    parser.add_argument("--lambda_feature", type=float, default=0.0)
    parser.add_argument("--lambda_encoder_consistency", type=float, default=0.0)
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
                        ])
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
    target_encoder = None
    if args.lambda_encoder_consistency > 0:
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

    weight, bias = fit_affine(
        datasets["train"].source,
        datasets["train"].target,
        ridge=args.align_ridge)
    torch.save({"weight": weight, "bias": bias}, exp_dir / "affine_alignment.pt")
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
        lowrank_rank=args.lowrank_rank).to(device)
    log_parameter_table(mapper, logger)
    total_params, trainable_params = count_parameters(mapper)
    writer.add_scalar("model/total_params", total_params, 0)
    writer.add_scalar("model/trainable_params", trainable_params, 0)
    writer.add_scalar("model/frozen_params", total_params - trainable_params, 0)
    add_text_json(writer, "config/model", {
        "mapper_type": args.mapper_type,
        "hidden_dim": args.hidden_dim,
        "lowrank_rank": args.lowrank_rank,
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
        "lambda_encoder_consistency": args.lambda_encoder_consistency,
        "encoder_consistency_target": args.encoder_consistency_target,
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
    scheduler = build_scheduler(
        optimizer, args.scheduler, args.epochs, len(loaders["train"]),
        args.eta_min)
    logger.info(
        "=> Optimizer: AdamW lr=%s weight_decay=%s scheduler=%s eta_min=%s",
        args.lr, args.weight_decay, args.scheduler, args.eta_min)
    logger.info(
        "=> Loss: type=%s lambda_code=%s lambda_feature=%s lambda_recon=%s "
        "lambda_encoder_consistency=%s encoder_consistency_target=%s",
        args.code_loss_type, args.lambda_code, args.lambda_feature,
        args.lambda_recon, args.lambda_encoder_consistency,
        args.encoder_consistency_target)
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

    best = {
        "val_code_mse": {"metric": math.inf, "epoch": 0},
        "val_decoder_nmse": {"metric": math.inf, "epoch": 0},
    }
    history = []

    def save_checkpoint(name, epoch, metrics):
        torch.save({
            "epoch": epoch,
            "state_dict": mapper.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        }, checkpoint_dir / name)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            mapper,
            loaders["train"],
            decoder,
            device,
            optimizer,
            scheduler,
            args.lambda_code,
            args.lambda_recon,
            args.lambda_feature,
            args.code_loss_type,
            std_weight,
            args.gate_l1,
            target_encoder,
            args.lambda_encoder_consistency,
            args.encoder_consistency_target)
        record = {"epoch": epoch, "lr": scheduler.get_lr()[0],
                  "train": train_metrics}
        log_metrics(writer, "train", train_metrics, epoch)
        writer.add_scalar("train/lr", scheduler.get_lr()[0], epoch)

        if args.eval_every and epoch % args.eval_every == 0:
            for split in ("val", "test"):
                metrics = evaluate(
                    mapper,
                    eval_loaders[split],
                    decoder,
                    device,
                    args.code_loss_type,
                    std_weight,
                    target_encoder,
                    args.encoder_consistency_target)
                record[split] = metrics
                log_metrics(writer, split, metrics, epoch)
                logger.info(
                    "Epoch [%d/%d] %s code_loss=%.6e code_mse=%.6e code_nmse=%.3fdB "
                    "cos=%.6f decoder_mse=%.6e decoder_nmse=%.3fdB n=%d",
                    epoch, args.epochs, split, metrics["code_loss"],
                    metrics["code_mse"], metrics["code_nmse"], metrics["code_cos"],
                    metrics["decoder_mse"], metrics["decoder_nmse"],
                    metrics["n"])
            val_metrics = record["val"]
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
            if val_metrics["decoder_nmse"] < best["val_decoder_nmse"]["metric"]:
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
            "train_recon_mse=%.6e train_decoder_nmse=%.3fdB%s",
            epoch, args.epochs, scheduler.get_lr()[0],
            train_metrics["loss"], train_metrics["code_loss"],
            train_metrics["code_mse"], train_metrics["code_nmse"],
            train_metrics["code_cos"], train_metrics["recon_mse"],
            train_metrics["decoder_nmse"], adapter_msg)

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
