#!/usr/bin/env python
import argparse
import json
import math
import os
import uuid
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
        raise ValueError("csi_path is required for decoder-aware losses")
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


def run_epoch(model, loader, device, optimizer=None, lambda_cos=0.0,
              lambda_cov=0.0, lambda_smoothl1=0.0, smoothl1_beta=0.05,
              lambda_sample_tail=0.0, sample_tail_ratio=0.2,
              lambda_dim_tail=0.0, dim_tail_ratio=0.05,
              lambda_whiten=0.0, whiten_stats=None, decoder=None,
              csi_tensor=None, lambda_rec=0.0, lambda_recT=0.0,
              lambda_fc=0.0, lambda_decoder_tail=0.0,
              decoder_tail_ratio=0.2):
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
        mse = F.mse_loss(pred, target)
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
    if decoder_aware:
        if args.decoder_checkpoint is None:
            raise ValueError("decoder-aware loss requires --decoder_checkpoint")
        decoder, decoder_cfg = load_decoder_from_checkpoint(args, device)
        if args.lambda_rec or args.lambda_decoder_tail:
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

    logger.info(f"device={device}")
    logger.info(f"mapper={args.mapper}, params={count_parameters(model)}")
    val_len = len(val_set) if use_val else 0
    logger.info(f"train={len(train_set)}, val={val_len}, code_dim={code_dim}")
    if not use_val:
        logger.info("val_ratio<=0: skip per-epoch validation; "
                    "select best checkpoint by train loss")
    logger.info(
        "loss="
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
        if val_metrics is not None:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        log_metrics_to_tensorboard(writer, "train", train_metrics, epoch)
        if val_metrics is not None:
            log_metrics_to_tensorboard(writer, "val", val_metrics, epoch)
            logger.info(
                f"epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.6e} "
                f"train_mse={train_metrics['mse']:.6e} "
                f"train_cos={train_metrics['cos']:.6f} "
                f"train_nmse={train_metrics['nmse']:.3f}dB "
                f"train_rec={train_metrics['rec']:.6e} "
                f"train_recT={train_metrics['recT']:.6e} "
                f"val_mse={val_metrics['mse']:.6e} "
                f"val_rec={val_metrics['rec']:.6e} "
                f"val_cos={val_metrics['cos']:.6f} "
                f"val_nmse={val_metrics['nmse']:.3f}dB")
        else:
            logger.info(
                f"epoch={epoch:04d} "
                f"train_loss={train_metrics['loss']:.6e} "
                f"train_mse={train_metrics['mse']:.6e} "
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
                "best": best_loss,
                "best_type": "loss",
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
                "best": best_mse,
                "best_type": "mse",
            }, checkpoint_dir / "best_mse.pth")

    (exp_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    def load_checkpoint_and_eval(checkpoint_path, output_paths):
        ckpt = torch.load(checkpoint_path,
                          weights_only=True,
                          map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        metrics = run_epoch(
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
            whiten_stats=whiten_stats,
            decoder=decoder,
            csi_tensor=csi_tensor,
            lambda_rec=args.lambda_rec,
            lambda_recT=args.lambda_recT,
            lambda_fc=args.lambda_fc,
            lambda_decoder_tail=args.lambda_decoder_tail,
            decoder_tail_ratio=args.decoder_tail_ratio)
        save_outputs(model, all_loader, device, output_paths)
        return metrics

    final_loss_metrics = None
    final_mse_metrics = None
    if (checkpoint_dir / "best_loss.pth").exists():
        final_loss_metrics = load_checkpoint_and_eval(
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
        final_mse_metrics = load_checkpoint_and_eval(
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
    final_metrics = final_loss_metrics or final_mse_metrics
    (exp_dir / "metrics.json").write_text(
        json.dumps({
            "best": best_loss,
            "best_loss": best_loss,
            "best_mse": best_mse,
            "all": final_metrics,
            "all_best_loss": final_loss_metrics,
            "all_best_mse": final_mse_metrics,
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
    if final_loss_metrics is not None:
        logger.info(f"all_best_loss_mse={final_loss_metrics['mse']:.6e} "
                    f"all_best_loss_cos={final_loss_metrics['cos']:.6f} "
                    f"all_best_loss_nmse={final_loss_metrics['nmse']:.3f}dB")
    if final_mse_metrics is not None:
        logger.info(f"all_best_mse_mse={final_mse_metrics['mse']:.6e} "
                    f"all_best_mse_cos={final_mse_metrics['cos']:.6f} "
                    f"all_best_mse_nmse={final_mse_metrics['nmse']:.3f}dB")


if __name__ == "__main__":
    main()
