#!/usr/bin/env python
import argparse
import json
import math
import os
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
log_experiment_header = _logger_module.log_experiment_header
log_parameter_table = _logger_module.log_parameter_table

_scheduler_spec = importlib.util.spec_from_file_location(
    "mapper_project_scheduler",
    ROOT / "utils" / "scheduler.py")
_scheduler_module = importlib.util.module_from_spec(_scheduler_spec)
_scheduler_spec.loader.exec_module(_scheduler_module)
FakeLR = _scheduler_module.FakeLR
WarmUpCosineAnnealingLR = _scheduler_module.WarmUpCosineAnnealingLR

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
    decoder = model.decoder.to(device).eval()
    for param in decoder.parameters():
        param.requires_grad_(False)
    return decoder, {
        "channel": channel,
        "nt": nt,
        "nc": nc,
        "train_path": args.csi_path or cfg.get("train_path"),
    }


def load_csi_tensor(path, channel, nt, nc, max_samples=0):
    if not path:
        raise ValueError("csi_path is required for decoder losses/eval")
    data = torch.load(path, weights_only=True,
                      map_location=torch.device("cpu")).float()
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(
            f"{path} should have shape (N, {channel}, {nt}, {nc}), "
            f"got {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data


def nmse_db(pred, target):
    err = (pred - target).pow(2).sum()
    power = target.pow(2).sum().clamp_min(1e-12)
    return 10.0 * torch.log10(err / power)


def cosine_mean(pred, target):
    return F.cosine_similarity(pred, target, dim=1).mean()


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


class AlignedResidualMapper(torch.nn.Module):
    def __init__(self, mapper, weight, bias, condition="source_start",
                 residual_scale=1.0):
        super().__init__()
        if condition not in ("source", "start", "source_start"):
            raise ValueError(f"Unknown residual_condition: {condition}")
        self.mapper = mapper
        self.condition = condition
        self.residual_scale = residual_scale
        self.register_buffer("alignment_weight", weight)
        self.register_buffer("alignment_bias", bias)

    def start(self, source):
        return source.matmul(self.alignment_weight) + self.alignment_bias

    def build_condition(self, source, start):
        if self.condition == "source":
            return source
        if self.condition == "start":
            return start
        return torch.cat([source, start], dim=-1)

    def forward(self, source):
        start = self.start(source)
        delta = self.mapper(self.build_condition(source, start))
        return start + self.residual_scale * delta


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


def sample_tail_from_values(values, ratio):
    if ratio <= 0:
        return values.new_tensor(0.0)
    k = max(1, int(math.ceil(values.numel() * ratio)))
    k = min(k, values.numel())
    return torch.topk(values, k=k, largest=True).values.mean()


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


def run_epoch(model, loader, device, optimizer=None, lambda_cos=0.0,
              lambda_cov=0.0, lambda_smoothl1=0.0, smoothl1_beta=0.05,
              lambda_sample_tail=0.0, sample_tail_ratio=0.2,
              lambda_dim_tail=0.0, dim_tail_ratio=0.05,
              lambda_whiten=0.0, whiten_stats=None, decoder=None,
              csi_tensor=None, lambda_rec=0.0, lambda_recT=0.0,
              lambda_fc=0.0, lambda_decoder_tail=0.0,
              decoder_tail_ratio=0.2, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total = {
        "loss": 0.0,
        "mse": 0.0,
        "smoothl1": 0.0,
        "sample_tail": 0.0,
        "dim_tail": 0.0,
        "whiten": 0.0,
        "rec": 0.0,
        "recT": 0.0,
        "fc": 0.0,
        "decoder_tail": 0.0,
        "cos": 0.0,
        "nmse": 0.0,
        "start_mse": 0.0,
        "start_nmse": 0.0,
        "delta_mse": 0.0,
        "n": 0,
    }
    decoder_aware = any([
        lambda_rec,
        lambda_recT,
        lambda_fc,
        lambda_decoder_tail,
    ])
    if decoder_aware and decoder is None:
        raise ValueError("decoder-aware losses require decoder")
    if (lambda_rec or lambda_decoder_tail) and csi_tensor is None:
        raise ValueError("lambda_rec/lambda_decoder_tail require csi_tensor")
    for source, target, indices in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pred = model(source)
        if hasattr(model, "start"):
            with torch.no_grad():
                start = model.start(source)
        else:
            start = source
        mse = F.mse_loss(pred, target)
        start_mse = F.mse_loss(start, target)
        delta_mse = F.mse_loss(pred - start, target - start)
        loss = mse
        smoothl1 = pred.new_tensor(0.0)
        sample_tail = pred.new_tensor(0.0)
        dim_tail = pred.new_tensor(0.0)
        whiten = pred.new_tensor(0.0)
        rec = pred.new_tensor(0.0)
        recT = pred.new_tensor(0.0)
        fc = pred.new_tensor(0.0)
        decoder_tail = pred.new_tensor(0.0)
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
        y_pred = None
        if lambda_rec or lambda_recT or lambda_decoder_tail:
            y_pred = decoder(pred)
        if lambda_rec:
            gt = csi_tensor[indices].to(device, non_blocking=True)
            rec = F.mse_loss(y_pred, gt)
            loss = loss + lambda_rec * rec
        if lambda_recT:
            with torch.no_grad():
                y_teacher = decoder(target)
            recT = F.mse_loss(y_pred, y_teacher)
            loss = loss + lambda_recT * recT
        if lambda_fc:
            if not hasattr(decoder, "fc_decoder"):
                raise ValueError("lambda_fc requires decoder.fc_decoder")
            fc_pred = decoder.fc_decoder(pred)
            with torch.no_grad():
                fc_teacher = decoder.fc_decoder(target)
            fc = F.mse_loss(fc_pred, fc_teacher)
            loss = loss + lambda_fc * fc
        if lambda_decoder_tail:
            gt = csi_tensor[indices].to(device, non_blocking=True)
            per_sample = (y_pred - gt).pow(2).flatten(1).mean(dim=1)
            decoder_tail = sample_tail_from_values(
                per_sample,
                decoder_tail_ratio)
            loss = loss + lambda_decoder_tail * decoder_tail
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        batch_n = source.size(0)
        total["loss"] += float(loss.detach().cpu()) * batch_n
        total["mse"] += float(mse.detach().cpu()) * batch_n
        total["smoothl1"] += float(smoothl1.detach().cpu()) * batch_n
        total["sample_tail"] += float(sample_tail.detach().cpu()) * batch_n
        total["dim_tail"] += float(dim_tail.detach().cpu()) * batch_n
        total["whiten"] += float(whiten.detach().cpu()) * batch_n
        total["rec"] += float(rec.detach().cpu()) * batch_n
        total["recT"] += float(recT.detach().cpu()) * batch_n
        total["fc"] += float(fc.detach().cpu()) * batch_n
        total["decoder_tail"] += float(decoder_tail.detach().cpu()) * batch_n
        total["cos"] += float(cosine_mean(pred, target).detach().cpu()) * batch_n
        total["nmse"] += float(nmse_db(pred, target).detach().cpu()) * batch_n
        total["start_mse"] += float(start_mse.detach().cpu()) * batch_n
        total["start_nmse"] += float(nmse_db(start, target).detach().cpu()) * batch_n
        total["delta_mse"] += float(delta_mse.detach().cpu()) * batch_n
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
        mapped = model(source)
        recon = decoder(mapped)
        mse = F.mse_loss(recon, gt)
        total_error += (recon - gt).pow(2).sum()
        total_power += gt.pow(2).sum()
        total_mse += float(mse.detach().cpu()) * source.size(0)
        total_n += source.size(0)
    decoder_nmse = 10.0 * torch.log10(total_error / total_power.clamp_min(1e-12))
    return {
        "decoder_mse": total_mse / max(total_n, 1),
        "decoder_nmse": float(decoder_nmse.detach().cpu()),
        "n": total_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_code", required=True)
    parser.add_argument("--target_code", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--mapper", default="flow",
                        choices=["identity", "mlp", "deep_mlp",
                                 "residual_mlp", "flow", "coupling_flow",
                                 "hybrid", "hybrid_flow_mlp",
                                 "delta_mlp", "residual_delta_mlp",
                                 "affine_residual_mlp"])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine",
                        choices=["const", "cosine"])
    parser.add_argument("--eta_min", type=float, default=5e-5)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--flow_hidden_dim", type=int, default=1024)
    parser.add_argument("--flow_blocks", type=int, default=8)
    parser.add_argument("--flow_clamp", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual_mapping", action="store_true",
                        help="fit a fixed alignment and train mapper as residual predictor")
    parser.add_argument("--align_mode", default="identity",
                        choices=["identity", "procrustes", "affine"])
    parser.add_argument("--align_ridge", type=float, default=1e-4)
    parser.add_argument("--residual_condition", default="source_start",
                        choices=["source", "start", "source_start"])
    parser.add_argument("--residual_scale", type=float, default=1.0)
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
    parser.add_argument("--lambda_rec", type=float, default=0.0)
    parser.add_argument("--lambda_recT", type=float, default=0.0)
    parser.add_argument("--lambda_fc", type=float, default=0.0)
    parser.add_argument("--lambda_decoder_tail", type=float, default=0.0)
    parser.add_argument("--decoder_tail_ratio", type=float, default=0.2)
    parser.add_argument("--decoder_checkpoint", default=None)
    parser.add_argument("--decoder_args_json", default=None)
    parser.add_argument("--csi_path", default=None)
    parser.add_argument("--decoder_name", default=None)
    parser.add_argument("--decoder_cr", type=int, default=None)
    parser.add_argument("--decoder_d_model", type=int, default=None)
    parser.add_argument("--decoder_dim_feedforward", type=int, default=None)
    parser.add_argument("--decoder_channel", type=int, default=None)
    parser.add_argument("--decoder_nt", type=int, default=None)
    parser.add_argument("--decoder_nc", type=int, default=None)
    parser.add_argument("--decoder_hidden", type=int, default=None)
    parser.add_argument("--decoder_num_blocks", type=int, default=None)
    parser.add_argument("--eval_decoder_every", type=int, default=0,
                        help="periodically evaluate mapped code with fixed decoder; 0 disables it")
    parser.add_argument("--eval_decoder_max_samples", type=int, default=0,
                        help="max samples for periodic decoder NMSE eval; 0 means full all_set")
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

    device = resolve_device(args.gpu, args.cpu)
    set_seed(args.seed)
    log_experiment_header(args, exp_dir=exp_dir, target_logger=logger)
    logger.info(f"=> Checkpoint directory: {checkpoint_dir}")
    logger.info(f"=> Codeword directory: {codeword_dir}")
    logger.info(f"=> TensorBoard directory: {tensorboard_dir}")
    train_set = CodewordPairDataset(
        args.source_code,
        args.target_code,
        split="train",
        val_ratio=args.val_ratio,
        max_samples=args.max_samples)
    use_val = args.val_ratio > 0
    val_set = None
    if use_val:
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
    residual_input_dim = code_dim
    if args.residual_mapping and args.residual_condition == "source_start":
        residual_input_dim = code_dim * 2
        if args.mapper not in (
                "delta_mlp",
                "residual_delta_mlp",
                "affine_residual_mlp"):
            raise ValueError(
                "residual_condition=source_start doubles the mapper input "
                "dimension; use mapper=delta_mlp or set "
                "residual_condition=source for legacy mappers.")
    base_model = build_mapper(
        args.mapper,
        code_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        flow_hidden_dim=args.flow_hidden_dim,
        flow_blocks=args.flow_blocks,
        clamp=args.flow_clamp,
        dropout=args.dropout,
        input_dim=residual_input_dim)
    if args.residual_mapping:
        weight, bias = fit_alignment(
            args.align_mode,
            train_set.source,
            train_set.target,
            ridge=args.align_ridge)
        model = AlignedResidualMapper(
            base_model,
            weight,
            bias,
            condition=args.residual_condition,
            residual_scale=args.residual_scale).to(device)
    else:
        model = base_model.to(device)
    log_parameter_table(model, logger)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
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
    decoder_eval_loader = None
    if args.eval_decoder_every:
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
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        args.epochs,
        len(train_loader),
        args.eta_min)
    whiten_stats = None
    if args.lambda_whiten:
        eigvecs, inv_eig = fit_teacher_whiten_stats(
            all_set.target,
            eps_ratio=args.whiten_eps_ratio)
        whiten_stats = (eigvecs.to(device), inv_eig.to(device))
    decoder = None
    csi_tensor = None
    decoder_aware = any([
        args.lambda_rec,
        args.lambda_recT,
        args.lambda_fc,
        args.lambda_decoder_tail,
    ])
    need_decoder = decoder_aware or bool(args.eval_decoder_every)
    if need_decoder:
        if args.decoder_checkpoint is None:
            raise ValueError("decoder loss/eval requires --decoder_checkpoint")
        decoder, decoder_cfg = load_decoder_from_checkpoint(args, device)
        if args.lambda_rec or args.lambda_decoder_tail or args.eval_decoder_every:
            csi_tensor = load_csi_tensor(
                decoder_cfg["train_path"],
                decoder_cfg["channel"],
                decoder_cfg["nt"],
                decoder_cfg["nc"],
                max_samples=args.max_samples)
            if csi_tensor.size(0) < all_set.source.size(0):
                raise ValueError(
                    f"CSI tensor has fewer samples than codewords: "
                    f"{csi_tensor.size(0)} vs {all_set.source.size(0)}")
        logger.info(f"=> Loaded fixed decoder from {args.decoder_checkpoint}")
        if csi_tensor is not None:
            logger.info(f"=> Loaded CSI tensor {tuple(csi_tensor.shape)} "
                        f"from {decoder_cfg['train_path']}")
        if args.eval_decoder_every:
            logger.info(
                "=> Periodic true NMSE eval: every=%d max_samples=%s",
                args.eval_decoder_every,
                args.eval_decoder_max_samples or "full")

    logger.info(f"=> Device: {device}")
    logger.info(
        "=> Mapper: %s trainable_params=%s",
        args.mapper,
        f"{count_parameters(model):,}")
    if args.residual_mapping:
        logger.info(
            "=> Alignment: mode=%s ridge=%s weight=%s bias=%s buffers=%d",
            args.align_mode,
            args.align_ridge,
            tuple(model.alignment_weight.shape),
            tuple(model.alignment_bias.shape),
            model.alignment_weight.numel() + model.alignment_bias.numel())
        logger.info(
            "=> Residual mapping: condition=%s residual_scale=%s "
            "input_dim=%d output_dim=%d",
            args.residual_condition,
            args.residual_scale,
            residual_input_dim,
            code_dim)
    val_len = len(val_set) if use_val else 0
    logger.info(
        f"=> Dataset: train={len(train_set)} val={val_len} "
        f"all={len(all_set)} code_dim={code_dim}")
    logger.info(
        f"=> DataLoader: batch_size={args.batch_size} "
        f"workers={args.workers} pin_memory={device.type == 'cuda'}")
    logger.info(
        f"=> Optimizer: AdamW lr={args.lr} weight_decay={args.weight_decay}")
    logger.info(
        "=> Scheduler: %s eta_min=%s steps_per_epoch=%d total_steps=%d",
        args.scheduler,
        args.eta_min,
        len(train_loader),
        args.epochs * len(train_loader))
    if not use_val:
        logger.info("val_ratio<=0: skip per-epoch validation; "
                    "select best checkpoint by train loss")
    logger.info(
        "=> Objective: "
        f"mse + smoothl1*{args.lambda_smoothl1} "
        f"+ sample_tail*{args.lambda_sample_tail} "
        f"+ dim_tail*{args.lambda_dim_tail} "
        f"+ whiten*{args.lambda_whiten} "
        f"+ rec*{args.lambda_rec} "
        f"+ recT*{args.lambda_recT} "
        f"+ fc*{args.lambda_fc} "
        f"+ decoder_tail*{args.lambda_decoder_tail} "
        f"+ cos*{args.lambda_cos} + cov*{args.lambda_cov}")
    best_loss = {
        "metric": math.inf,
        "mse": math.inf,
        "loss": math.inf,
        "epoch": 0,
        "selection": "val_loss" if use_val else "train_loss",
    }
    best_mse = {
        "metric": math.inf,
        "mse": math.inf,
        "loss": math.inf,
        "epoch": 0,
        "selection": "val_mse" if use_val else "train_mse",
    }
    best_nmse = {
        "metric": math.inf,
        "decoder_mse": math.inf,
        "epoch": 0,
        "selection": "true_decoder_nmse",
    }
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scheduler=scheduler,
            lambda_cos=args.lambda_cos,
            lambda_cov=args.lambda_cov,
            lambda_smoothl1=args.lambda_smoothl1,
            smoothl1_beta=args.smoothl1_beta,
            lambda_sample_tail=args.lambda_sample_tail,
            sample_tail_ratio=args.sample_tail_ratio,
            lambda_dim_tail=args.lambda_dim_tail,
            dim_tail_ratio=args.dim_tail_ratio,
            lambda_whiten=args.lambda_whiten,
            whiten_stats=whiten_stats,
            decoder=decoder,
            csi_tensor=csi_tensor,
            lambda_rec=args.lambda_rec,
            lambda_recT=args.lambda_recT,
            lambda_fc=args.lambda_fc,
            lambda_decoder_tail=args.lambda_decoder_tail,
            decoder_tail_ratio=args.decoder_tail_ratio)
        val_metrics = None
        if use_val:
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                lambda_cos=args.lambda_cos,
                lambda_cov=args.lambda_cov,
                lambda_smoothl1=args.lambda_smoothl1,
                smoothl1_beta=args.smoothl1_beta,
                lambda_sample_tail=args.lambda_sample_tail,
                sample_tail_ratio=args.sample_tail_ratio,
                lambda_dim_tail=args.lambda_dim_tail,
                dim_tail_ratio=args.dim_tail_ratio,
                lambda_whiten=args.lambda_whiten,
                whiten_stats=whiten_stats,
                decoder=decoder,
                csi_tensor=csi_tensor,
                lambda_rec=args.lambda_rec,
                lambda_recT=args.lambda_recT,
                lambda_fc=args.lambda_fc,
                lambda_decoder_tail=args.lambda_decoder_tail,
                decoder_tail_ratio=args.decoder_tail_ratio)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row["lr"] = scheduler.get_lr()[0]
        if val_metrics is not None:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        decoder_metrics = None
        if args.eval_decoder_every and epoch % args.eval_decoder_every == 0:
            decoder_metrics = evaluate_decoder_nmse(
                model,
                decoder_eval_loader,
                csi_tensor,
                decoder,
                device)
            row.update({
                f"true_decoder_eval_{k}": v
                for k, v in decoder_metrics.items()
            })
            log_metrics_to_tensorboard(
                writer,
                "true_decoder_eval",
                decoder_metrics,
                epoch)
            logger.info(
                f"Epoch [{epoch}/{args.epochs}] true_decoder_eval "
                f"n={decoder_metrics['n']} "
                f"decoder_mse={decoder_metrics['decoder_mse']:.6e} "
                f"decoder_nmse={decoder_metrics['decoder_nmse']:.3f}dB")
        history.append(row)
        log_metrics_to_tensorboard(writer, "train", train_metrics, epoch)
        writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step=epoch)
        if val_metrics is not None:
            log_metrics_to_tensorboard(writer, "val", val_metrics, epoch)
            logger.info(
                f"Epoch [{epoch}/{args.epochs}] "
                f"lr={scheduler.get_lr()[0]:.6e} "
                f"train_opt_loss={train_metrics['loss']:.6e} "
                f"val_select_loss={val_metrics['loss']:.6e} "
                f"val_mse={val_metrics['mse']:.6e} "
                f"val_start={val_metrics['start_mse']:.6e} "
                f"val_delta={val_metrics['delta_mse']:.6e} "
                f"val_cos={val_metrics['cos']:.6f} "
                f"val_nmse={val_metrics['nmse']:.3f}dB "
                f"val_rec={val_metrics['rec']:.6e} "
                f"val_recT={val_metrics['recT']:.6e} "
                f"val_fc={val_metrics['fc']:.6e}")
        else:
            logger.info(
                f"Epoch [{epoch}/{args.epochs}] "
                f"lr={scheduler.get_lr()[0]:.6e} "
                f"train_opt_loss={train_metrics['loss']:.6e} "
                f"train_select_loss={train_metrics['loss']:.6e} "
                f"train_mse={train_metrics['mse']:.6e} "
                f"train_start={train_metrics['start_mse']:.6e} "
                f"train_delta={train_metrics['delta_mse']:.6e} "
                f"train_cos={train_metrics['cos']:.6f} "
                f"train_nmse={train_metrics['nmse']:.3f}dB "
                f"train_rec={train_metrics['rec']:.6e} "
                f"train_recT={train_metrics['recT']:.6e} "
                f"train_fc={train_metrics['fc']:.6e} "
                f"train_decoder_tail={train_metrics['decoder_tail']:.6e}")
        select_metrics = val_metrics if use_val else train_metrics
        loss_metric = -float(epoch) if args.save_last else select_metrics["loss"]
        mse_metric = -float(epoch) if args.save_last else select_metrics["mse"]
        if loss_metric < best_loss["metric"]:
            best_loss = {
                "metric": loss_metric,
                "mse": select_metrics["mse"],
                "loss": select_metrics["loss"],
                "epoch": epoch,
                "selection": "last" if args.save_last
                else ("val_loss" if use_val else "train_loss"),
            }
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best_loss,
                "best_type": "loss",
                "args": vars(args),
            }, checkpoint_dir / "best_loss.pth")
        if mse_metric < best_mse["metric"]:
            best_mse = {
                "metric": mse_metric,
                "mse": select_metrics["mse"],
                "loss": select_metrics["loss"],
                "epoch": epoch,
                "selection": "last" if args.save_last
                else ("val_mse" if use_val else "train_mse"),
            }
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best_mse,
                "best_type": "mse",
                "args": vars(args),
            }, checkpoint_dir / "best_mse.pth")
        if decoder_metrics is not None:
            nmse_metric = decoder_metrics["decoder_nmse"]
            if nmse_metric < best_nmse["metric"]:
                best_nmse = {
                    "metric": nmse_metric,
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
                    "best_type": "nmse",
                    "args": vars(args),
                }, checkpoint_dir / "best_nmse.pth")

    (exp_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    def load_checkpoint_and_eval(checkpoint_path, output_paths,
                                 eval_decoder=False):
        ckpt = torch.load(checkpoint_path,
                          weights_only=True,
                          map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = run_epoch(
            model,
            all_loader,
            device,
            lambda_cos=args.lambda_cos,
            lambda_cov=args.lambda_cov,
            lambda_smoothl1=args.lambda_smoothl1,
            smoothl1_beta=args.smoothl1_beta,
            lambda_sample_tail=args.lambda_sample_tail,
            sample_tail_ratio=args.sample_tail_ratio,
            lambda_dim_tail=args.lambda_dim_tail,
            dim_tail_ratio=args.dim_tail_ratio,
            lambda_whiten=args.lambda_whiten,
            whiten_stats=whiten_stats,
            decoder=decoder,
            csi_tensor=csi_tensor,
            lambda_rec=args.lambda_rec,
            lambda_recT=args.lambda_recT,
            lambda_fc=args.lambda_fc,
            lambda_decoder_tail=args.lambda_decoder_tail,
            decoder_tail_ratio=args.decoder_tail_ratio)
        save_outputs(model, all_loader, device, output_paths)
        decoder_metrics = None
        if eval_decoder:
            decoder_metrics = evaluate_decoder_nmse(
                model,
                all_loader,
                csi_tensor,
                decoder,
                device)
        return metrics, decoder_metrics

    final_loss_metrics = None
    final_mse_metrics = None
    final_nmse_metrics = None
    final_nmse_decoder_metrics = None
    if (checkpoint_dir / "best_loss.pth").exists():
        final_loss_metrics, _ = load_checkpoint_and_eval(
            checkpoint_dir / "best_loss.pth",
            [
                codeword_dir / "mapped_code_best_loss.pt",
                codeword_dir / "mapped_code.pt",
                exp_dir / "mapped_code_best_loss.pt",
                # Keep the old location for compatibility.
                exp_dir / "mapped_code.pt",
            ])
        log_metrics_to_tensorboard(
            writer,
            "all_best_loss",
            final_loss_metrics,
            args.epochs)
    if (checkpoint_dir / "best_mse.pth").exists():
        final_mse_metrics, _ = load_checkpoint_and_eval(
            checkpoint_dir / "best_mse.pth",
            [
                codeword_dir / "mapped_code_best_mse.pt",
                exp_dir / "mapped_code_best_mse.pt",
            ])
        log_metrics_to_tensorboard(
            writer,
            "all_best_mse",
            final_mse_metrics,
            args.epochs)
    if (checkpoint_dir / "best_nmse.pth").exists():
        final_nmse_metrics, final_nmse_decoder_metrics = load_checkpoint_and_eval(
            checkpoint_dir / "best_nmse.pth",
            [
                codeword_dir / "mapped_code_best_nmse.pt",
                exp_dir / "mapped_code_best_nmse.pt",
            ],
            eval_decoder=True)
        log_metrics_to_tensorboard(
            writer,
            "all_best_nmse_code",
            final_nmse_metrics,
            args.epochs)
        log_metrics_to_tensorboard(
            writer,
            "all_best_nmse_decoder",
            final_nmse_decoder_metrics,
            args.epochs)
    final_metrics = final_loss_metrics or final_mse_metrics
    (exp_dir / "metrics.json").write_text(
        json.dumps({
            "best": best_loss,
            "best_loss": best_loss,
            "best_mse": best_mse,
            "best_nmse": best_nmse,
            "all": final_metrics,
            "all_best_loss": final_loss_metrics,
            "all_best_mse": final_mse_metrics,
            "all_best_nmse_code": final_nmse_metrics,
            "all_best_nmse_decoder": final_nmse_decoder_metrics,
        }, indent=2),
        encoding="utf-8")
    writer.flush()
    writer.close()
    logger.info(f"best_loss_epoch={best_loss['epoch']} "
                f"best_loss_selection={best_loss['selection']} "
                f"best_loss_metric={best_loss['metric']:.6e} "
                f"best_loss_mse={best_loss['mse']:.6e} "
                f"best_loss_loss={best_loss['loss']:.6e}")
    logger.info(f"best_mse_epoch={best_mse['epoch']} "
                f"best_mse_selection={best_mse['selection']} "
                f"best_mse_metric={best_mse['metric']:.6e} "
                f"best_mse_mse={best_mse['mse']:.6e} "
                f"best_mse_loss={best_mse['loss']:.6e}")
    if best_nmse["epoch"]:
        logger.info(f"best_nmse_epoch={best_nmse['epoch']} "
                    f"best_nmse_selection={best_nmse['selection']} "
                    f"best_nmse_metric={best_nmse['metric']:.3f}dB "
                    f"best_nmse_decoder_mse={best_nmse['decoder_mse']:.6e}")
    if final_loss_metrics is not None:
        logger.info(f"all_best_loss_mse={final_loss_metrics['mse']:.6e} "
                    f"all_best_loss_cos={final_loss_metrics['cos']:.6f} "
                    f"all_best_loss_nmse={final_loss_metrics['nmse']:.3f}dB")
    if final_mse_metrics is not None:
        logger.info(f"all_best_mse_mse={final_mse_metrics['mse']:.6e} "
                    f"all_best_mse_cos={final_mse_metrics['cos']:.6f} "
                    f"all_best_mse_nmse={final_mse_metrics['nmse']:.3f}dB")
    if final_nmse_decoder_metrics is not None:
        logger.info(
            f"all_best_nmse_decoder_mse="
            f"{final_nmse_decoder_metrics['decoder_mse']:.6e} "
            f"all_best_nmse_decoder_nmse="
            f"{final_nmse_decoder_metrics['decoder_nmse']:.3f}dB")


if __name__ == "__main__":
    main()
