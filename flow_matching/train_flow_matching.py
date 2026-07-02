#!/usr/bin/env python
import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard.writer import SummaryWriter

FLOW_DIR = Path(__file__).resolve().parent
ROOT = FLOW_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(FLOW_DIR) not in sys.path:
    sys.path.insert(0, str(FLOW_DIR))

import importlib.util

_logger_spec = importlib.util.spec_from_file_location(
    "flow_matching_project_logger",
    ROOT / "utils" / "logger.py")
_logger_module = importlib.util.module_from_spec(_logger_spec)
_logger_spec.loader.exec_module(_logger_module)
logger = _logger_module.logger
setup_logging = _logger_module.setup_logging
log_experiment_header = _logger_module.log_experiment_header
log_parameter_table = _logger_module.log_parameter_table
_scheduler_spec = importlib.util.spec_from_file_location(
    "flow_matching_project_scheduler",
    ROOT / "utils" / "scheduler.py")
_scheduler_module = importlib.util.module_from_spec(_scheduler_spec)
_scheduler_spec.loader.exec_module(_scheduler_module)
FakeLR = _scheduler_module.FakeLR
WarmUpCosineAnnealingLR = _scheduler_module.WarmUpCosineAnnealingLR

from dataset import CodewordPairDataset
from models import FlowMatchingTranslator, count_parameters


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
    main_models = load_main_models_package()
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
        "cr": cr,
        "decoder": decoder_name,
    }


def load_csi_tensor(path, channel, nt, nc, max_samples=0):
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


def code_metrics(pred, target):
    mse = F.mse_loss(pred, target)
    return {
        "mse": float(mse.detach().cpu()),
        "rmse": float(mse.sqrt().detach().cpu()),
        "cos": float(cosine_mean(pred, target).detach().cpu()),
        "nmse": float(nmse_db(pred, target).detach().cpu()),
    }


def decoder_nmse_db(pred, target):
    err = (pred - target).pow(2).sum()
    power = target.pow(2).sum().clamp_min(1e-12)
    return 10.0 * torch.log10(err / power)


def nmse_db_from_sums(error_sum, power_sum):
    return 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))


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


def run_epoch(model, loader, device, optimizer=None, t_eps=1e-4,
              lambda_endpoint=0.0, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total = {
        "loss": 0.0,
        "velocity_mse": 0.0,
        "endpoint_mse": 0.0,
        "start_mse": 0.0,
        "n": 0,
    }
    for source, target, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        start = model.start(source)
        t = torch.rand(source.size(0), device=device)
        if t_eps:
            t = t * (1.0 - 2.0 * t_eps) + t_eps
        view_t = t.view(-1, 1)
        velocity_target = target - start
        x_t = (1.0 - view_t) * start + view_t * target
        velocity_pred = model.velocity(x_t, t, source, start)
        velocity_mse = F.mse_loss(velocity_pred, velocity_target)
        endpoint_pred = x_t + (1.0 - view_t) * velocity_pred
        endpoint_mse = F.mse_loss(endpoint_pred, target)
        loss = velocity_mse + lambda_endpoint * endpoint_mse
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        batch_n = source.size(0)
        total["loss"] += float(loss.detach().cpu()) * batch_n
        total["velocity_mse"] += float(velocity_mse.detach().cpu()) * batch_n
        total["endpoint_mse"] += float(endpoint_mse.detach().cpu()) * batch_n
        total["start_mse"] += float(F.mse_loss(start, target).detach().cpu()) * batch_n
        total["n"] += batch_n
    return {k: v / max(total["n"], 1) for k, v in total.items() if k != "n"}


@torch.no_grad()
def evaluate_ode(model, loader, device, ode_steps=16, ode_method="euler"):
    model.eval()
    preds = []
    targets = []
    starts = []
    for source, target, _ in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        starts.append(model.start(source).cpu())
        preds.append(model.sample(source, steps=ode_steps, method=ode_method).cpu())
        targets.append(target.cpu())
    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    start = torch.cat(starts, dim=0)
    metrics = code_metrics(pred, target)
    for key, value in code_metrics(start, target).items():
        metrics[f"start_{key}"] = value
    return metrics


@torch.no_grad()
def evaluate_decoder_ode(model, loader, decoder, csi_tensor, device,
                         ode_steps=16, ode_method="euler"):
    model.eval()
    decoder.eval()
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    code_preds = []
    code_targets = []
    for source, target, indices in loader:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        gt = csi_tensor[indices].to(device, non_blocking=True)
        mapped = model.sample(source, steps=ode_steps, method=ode_method)
        recon = decoder(mapped)
        mse = F.mse_loss(recon, gt)
        total_error += (recon - gt).pow(2).sum()
        total_power += gt.pow(2).sum()
        total_mse += float(mse.detach().cpu()) * source.size(0)
        total_n += source.size(0)
        code_preds.append(mapped.cpu())
        code_targets.append(target.cpu())
    code_pred = torch.cat(code_preds, dim=0)
    code_target = torch.cat(code_targets, dim=0)
    metrics = code_metrics(code_pred, code_target)
    metrics = {f"code_{key}": value for key, value in metrics.items()}
    metrics.update({
        "decoder_mse": total_mse / max(total_n, 1),
        "decoder_nmse": float(nmse_db_from_sums(
            total_error,
            total_power).detach().cpu()),
        "n": total_n,
    })
    return metrics


@torch.no_grad()
def save_outputs(model, loader, device, output_paths, ode_steps=16,
                 ode_method="euler"):
    if isinstance(output_paths, (str, Path)):
        output_paths = [output_paths]
    output_paths = [Path(path) for path in output_paths]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    outs = []
    for source, _, _ in loader:
        source = source.to(device, non_blocking=True)
        outs.append(model.sample(source, steps=ode_steps, method=ode_method).cpu())
    outputs = torch.cat(outs, dim=0)
    for output_path in output_paths:
        torch.save(outputs, output_path)
    return outputs


def write_json(path, obj):
    with Path(path).open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def log_metrics(writer, prefix, metrics, epoch):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, global_step=epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_code", required=True)
    parser.add_argument("--target_code", required=True)
    parser.add_argument("--exp_dir", required=True)
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
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--condition", default="source_start",
                        choices=["source", "start", "source_start", "none"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--align_mode", default="identity",
                        choices=["identity", "procrustes", "affine"])
    parser.add_argument("--align_ridge", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--t_eps", type=float, default=1e-4)
    parser.add_argument("--lambda_endpoint", type=float, default=0.0)
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--ode_method", default="euler",
                        choices=["euler", "heun"])
    parser.add_argument("--eval_ode_every", type=int, default=0)
    parser.add_argument("--eval_decoder_every", type=int, default=0,
                        help="compute true fixed-decoder NMSE every N epochs; 0 disables it")
    parser.add_argument("--eval_decoder_max_samples", type=int, default=0,
                        help="limit samples for periodic decoder NMSE; 0 means full set")
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
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    codeword_dir = exp_dir / "codewords"
    tensorboard_dir = exp_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    codeword_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    write_json(exp_dir / "args.json", vars(args))

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
    model = FlowMatchingTranslator(
        code_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        time_dim=args.time_dim,
        condition=args.condition,
        dropout=args.dropout)
    weight, bias = fit_alignment(
        args.align_mode,
        train_set.source,
        train_set.target,
        ridge=args.align_ridge)
    model.alignment.set_transform(weight, bias)
    model.to(device)
    log_parameter_table(model, logger)
    logger.info(
        "=> Alignment buffers: weight=%s bias=%s numel=%d",
        tuple(model.alignment.weight.shape),
        tuple(model.alignment.bias.shape),
        model.alignment.weight.numel() + model.alignment.bias.numel())
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
    decoder = None
    csi_tensor = None
    decoder_eval_loader = None
    if args.eval_decoder_every:
        if not args.decoder_checkpoint:
            raise ValueError("--eval_decoder_every requires --decoder_checkpoint")
        if not args.csi_path:
            raise ValueError("--eval_decoder_every requires --csi_path")
        decoder, decoder_cfg = load_decoder_from_checkpoint(args, device)
        csi_tensor = load_csi_tensor(
            args.csi_path,
            decoder_cfg["channel"],
            decoder_cfg["nt"],
            decoder_cfg["nc"],
            max_samples=args.max_samples)
        if csi_tensor.size(0) < len(all_set):
            raise ValueError(
                f"CSI tensor has fewer samples than codewords: "
                f"{csi_tensor.size(0)} vs {len(all_set)}")
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

    logger.info(f"=> Device: {device}")
    logger.info(f"=> Source code: {args.source_code}")
    logger.info(f"=> Target code: {args.target_code}")
    logger.info(f"=> Code shape: {tuple(all_set.source.shape)}")
    logger.info(f"=> Align mode: {args.align_mode}")
    logger.info(f"=> Parameters: {count_parameters(model):,}")
    logger.info(
        "=> Dataset sizes: train=%d, val=%d, all=%d, code_dim=%d",
        len(train_set),
        len(val_set) if use_val else 0,
        len(all_set),
        code_dim)
    logger.info(
        "=> Loader config: batch_size=%d, workers=%d, pin_memory=%s",
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
        "=> Flow objective: velocity_mse + endpoint*%s; "
        "ode_steps=%s ode_method=%s eval_ode_every=%s",
        args.lambda_endpoint,
        args.ode_steps,
        args.ode_method,
        args.eval_ode_every)
    if args.eval_decoder_every:
        logger.info(
            "=> Periodic decoder eval: every=%d max_samples=%s "
            "decoder_checkpoint=%s csi_path=%s",
            args.eval_decoder_every,
            args.eval_decoder_max_samples or "full",
            args.decoder_checkpoint,
            args.csi_path)

    history = []
    best_loss = float("inf")
    best_endpoint = float("inf")
    best_loss_path = checkpoint_dir / "best_loss.pth"
    best_mse_path = checkpoint_dir / "best_mse.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            t_eps=args.t_eps,
            lambda_endpoint=args.lambda_endpoint,
            scheduler=scheduler)
        if use_val:
            eval_loader = val_loader
            eval_prefix = "val"
            eval_metrics = run_epoch(
                model,
                eval_loader,
                device,
                optimizer=None,
                t_eps=args.t_eps,
                lambda_endpoint=args.lambda_endpoint)
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": eval_metrics,
                "lr": scheduler.get_lr()[0],
            }
        else:
            eval_loader = train_loader
            eval_prefix = "train"
            eval_metrics = train_metrics
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "lr": scheduler.get_lr()[0],
            }
        if args.eval_ode_every and epoch % args.eval_ode_every == 0:
            ode_metrics = evaluate_ode(
                model,
                eval_loader,
                device,
                ode_steps=args.ode_steps,
                ode_method=args.ode_method)
            record[f"{eval_prefix}_ode"] = ode_metrics
            log_metrics(writer, f"{eval_prefix}_ode", ode_metrics, epoch)
        if args.eval_decoder_every and epoch % args.eval_decoder_every == 0:
            decoder_metrics = evaluate_decoder_ode(
                model,
                decoder_eval_loader,
                decoder,
                csi_tensor,
                device,
                ode_steps=args.ode_steps,
                ode_method=args.ode_method)
            record["decoder_ode"] = decoder_metrics
            log_metrics(writer, "decoder_ode", decoder_metrics, epoch)
            logger.info(
                f"Epoch [{epoch}/{args.epochs}] true_decoder_eval "
                f"n={decoder_metrics['n']} "
                f"code_mse={decoder_metrics['code_mse']:.6e} "
                f"code_nmse={decoder_metrics['code_nmse']:.3f}dB "
                f"decoder_mse={decoder_metrics['decoder_mse']:.6e} "
                f"decoder_nmse={decoder_metrics['decoder_nmse']:.3f}dB")
        history.append(record)
        log_metrics(writer, "train", train_metrics, epoch)
        log_metrics(writer, eval_prefix, eval_metrics, epoch)
        writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step=epoch)

        selected_loss = eval_metrics["loss"]
        selected_endpoint = eval_metrics["endpoint_mse"]
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"lr={scheduler.get_lr()[0]:.6e} "
            f"train_opt_loss={train_metrics['loss']:.6e} "
            f"{eval_prefix}_select_loss={selected_loss:.6e} "
            f"{eval_prefix}_endpoint={selected_endpoint:.6e} "
            f"{eval_prefix}_start={eval_metrics['start_mse']:.6e}")

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": min(best_loss, selected_loss),
            "best_endpoint_mse": min(best_endpoint, selected_endpoint),
            "args": vars(args),
        }
        if args.save_last or selected_loss < best_loss:
            best_loss = selected_loss
            state["best"] = best_loss
            torch.save(state, best_loss_path)
        if args.save_last or selected_endpoint < best_endpoint:
            best_endpoint = selected_endpoint
            state["best"] = best_endpoint
            torch.save(state, best_mse_path)

    write_json(exp_dir / "history.json", history)

    metrics = {}
    for tag, checkpoint_path, output_paths in [
        ("best_loss", best_loss_path, [
            codeword_dir / "mapped_code_best_loss.pt",
        ]),
        ("best_mse", best_mse_path, [
            codeword_dir / "mapped_code_best_mse.pt",
            codeword_dir / "mapped_code.pt",
            exp_dir / "mapped_code.pt",
        ]),
    ]:
        ckpt = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location=torch.device("cpu"))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        save_outputs(
            model,
            all_loader,
            device,
            output_paths,
            ode_steps=args.ode_steps,
            ode_method=args.ode_method)
        metrics[tag] = evaluate_ode(
            model,
            all_loader,
            device,
            ode_steps=args.ode_steps,
            ode_method=args.ode_method)
        metrics[tag]["epoch"] = ckpt.get("epoch")
    write_json(exp_dir / "metrics.json", metrics)
    writer.close()
    logger.info(f"=> Final metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
