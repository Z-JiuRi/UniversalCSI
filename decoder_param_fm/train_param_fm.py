import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoder_param_fm.models import (  # noqa: E402
    ConditionEncoder,
    ConditionExtractor,
    DecoderParamFMSystem,
    ParamFM,
)
from decoder_param_fm.param_utils import (  # noqa: E402
    build_param_meta,
    build_decoder_from_args,
    compute_global_norm_stats,
    compute_norm_stats,
    extract_decoder_state,
    load_codes,
    load_param_pairs,
    make_base_and_target,
    masked_mse,
    normalize_target_delta,
    save_artifacts,
    state_to_tokens,
)
from utils.logger import (  # noqa: E402
    count_parameters,
    log_experiment_header,
    log_parameter_table,
    logger,
    setup_logging,
)
from utils.scheduler import WarmUpCosineAnnealingLR  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train flow matching to generate full decoder parameters.")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--decoder_args_json", type=str, required=True)
    parser.add_argument("--target_checkpoint", type=str, default="")
    parser.add_argument("--guide_code_path", type=str, default="")
    parser.add_argument(
        "--data_txt", type=str, default="",
        help="Optional CSV-like file: code_path,target_checkpoint per line.")
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--token_size", type=int, default=512)
    parser.add_argument("--param_norm", choices=["rms", "zscore"], default="rms")
    parser.add_argument(
        "--condition_extract",
        choices=["random", "svd", "set_transformer"],
        default="random")
    parser.add_argument(
        "--condition_inject",
        choices=["film", "cross_attention", "hyper_lora"],
        default="film")
    parser.add_argument("--condition_tokens", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--set_layers", type=int, default=2)
    parser.add_argument("--hyper_lora_rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda_endpoint", type=float, default=1.0)
    parser.add_argument("--max_guide_codes", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_system(args, code_dim, meta):
    condition_extractor = ConditionExtractor(
        method=args.condition_extract,
        code_dim=code_dim,
        num_tokens=args.condition_tokens,
        d_model=args.cond_dim,
        num_heads=args.num_heads,
        num_layers=args.set_layers,
    )
    condition_encoder = ConditionEncoder(
        code_dim=code_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
    )
    param_fm = ParamFM(
        num_tensors=meta["num_tensors"],
        max_layer_id=meta["max_layer_id"],
        max_token_offset=meta["max_token_offset"],
        token_size=args.token_size,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        time_dim=args.time_dim,
        cond_dim=args.cond_dim,
        condition_inject=args.condition_inject,
        num_heads=args.num_heads,
        hyper_lora_rank=args.hyper_lora_rank,
        dropout=args.dropout,
    )
    return DecoderParamFMSystem(condition_extractor, condition_encoder, param_fm)


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, step,
                    loss, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "step": step,
        "loss": float(loss),
        "meta": meta,
    }, path)


def build_base_state(args):
    base_decoder, decoder_cfg = build_decoder_from_args(
        args.decoder_args_json, seed=args.base_seed)
    base_state = {
        key: value.detach().cpu().float()
        for key, value in base_decoder.state_dict().items()
    }
    return base_state, decoder_cfg


def validate_target_state(base_state, target_state, source):
    missing = sorted(set(base_state) - set(target_state))
    unexpected = sorted(set(target_state) - set(base_state))
    if missing or unexpected:
        raise ValueError(
            f"decoder state mismatch for {source}: "
            f"missing={missing}, unexpected={unexpected}")
    for key in base_state:
        if tuple(base_state[key].shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"shape mismatch for {key} in {source}: "
                f"{tuple(base_state[key].shape)} vs "
                f"{tuple(target_state[key].shape)}")


def load_training_examples(args, device):
    if args.data_txt:
        pairs = load_param_pairs(args.data_txt)
    else:
        if not args.guide_code_path or not args.target_checkpoint:
            raise ValueError(
                "Either --data_txt or both --guide_code_path and "
                "--target_checkpoint must be provided")
        pairs = [{
            "code_path": args.guide_code_path,
            "target_checkpoint": args.target_checkpoint,
        }]

    base_state, decoder_cfg = build_base_state(args)
    target_states = []
    for pair in pairs:
        target_state = extract_decoder_state(pair["target_checkpoint"])
        validate_target_state(base_state, target_state, pair["target_checkpoint"])
        target_states.append(target_state)

    meta = build_param_meta(target_states[0], args.token_size)
    if len(target_states) == 1:
        norm_stats = compute_norm_stats(
            base_state, target_states[0], method=args.param_norm)
    else:
        norm_stats = compute_global_norm_stats(
            base_state, target_states, method=args.param_norm)

    examples = []
    code_dim = None
    for idx, (pair, target_state) in enumerate(zip(pairs, target_states)):
        norm_target_state = normalize_target_delta(
            base_state, target_state, norm_stats)
        theta1, token_mask, meta_tensors = state_to_tokens(
            norm_target_state, meta, device="cpu")
        guide_codes = load_codes(
            pair["code_path"], max_samples=args.max_guide_codes)
        if code_dim is None:
            code_dim = guide_codes.size(1)
        elif code_dim != guide_codes.size(1):
            raise ValueError(
                f"code_dim mismatch for {pair['code_path']}: "
                f"{guide_codes.size(1)} vs {code_dim}")
        examples.append({
            "index": idx,
            "code_path": pair["code_path"],
            "target_checkpoint": pair["target_checkpoint"],
            "guide_codes": guide_codes,
            "theta1": theta1,
            "token_mask": token_mask,
            "meta_tensors": meta_tensors,
            "num_codes": int(guide_codes.size(0)),
        })

    # Move static metadata once; per-example tensors stay on CPU until sampled.
    meta_tensors = {
        key: value.to(device) for key, value in examples[0]["meta_tensors"].items()
    }
    return base_state, target_states, decoder_cfg, meta, norm_stats, examples, code_dim, meta_tensors


def summarize_decoder_tensors(state):
    total = sum(value.numel() for value in state.values())
    largest = sorted(
        ((name, tuple(value.shape), value.numel())
         for name, value in state.items()),
        key=lambda item: item[2],
        reverse=True)[:8]
    lines = [
        "=> Target decoder tensors:",
        f"   tensors={len(state)} total_params={total:,}",
        "   largest tensors:",
    ]
    for name, shape, numel in largest:
        lines.append(f"     {name:<55} shape={shape} numel={numel:,}")
    logger.info("\n%s", "\n".join(lines))


def summarize_norm_stats(norm_stats):
    methods = sorted({stat["method"] for stat in norm_stats.values()})
    lines = [
        "=> Parameter normalization:",
        f"   method={','.join(methods)} tensors={len(norm_stats)}",
    ]
    for name, stat in list(norm_stats.items())[:8]:
        if stat["method"] == "rms":
            value = stat["scale"].item()
            lines.append(f"   {name:<55} scale={value:.6e}")
        else:
            mean = stat["mean"].item()
            std = stat["std"].item()
            lines.append(f"   {name:<55} mean={mean:.6e} std={std:.6e}")
    if len(norm_stats) > 8:
        lines.append(f"   ... {len(norm_stats) - 8} more tensors")
    logger.info("\n%s", "\n".join(lines))


def log_training_preamble(args, device, decoder_cfg, meta, examples, model,
                          total_steps, warmup_steps):
    total, trainable, frozen = count_parameters(model)
    logger.info("=> Device: %s", device)
    logger.info("=> Data txt: %s", args.data_txt or "<single pair>")
    logger.info("=> Target checkpoint: %s", args.target_checkpoint or "<from data_txt>")
    logger.info("=> Decoder args json: %s", args.decoder_args_json)
    logger.info("=> Guide code path: %s", args.guide_code_path or "<from data_txt>")
    logger.info(
        "=> Decoder: type=%s cr=%s channel=%s nt=%s nc=%s d_model=%s "
        "dim_feedforward=%s",
        decoder_cfg.get("decoder", "transnet"),
        decoder_cfg.get("cr"),
        decoder_cfg.get("channel"),
        decoder_cfg.get("nt"),
        decoder_cfg.get("nc"),
        decoder_cfg.get("d_model"),
        decoder_cfg.get("dim_feedforward"))
    logger.info(
        "=> Condition: extract=%s tokens=%d inject=%s cond_dim=%d "
        "heads=%d set_layers=%d hyper_lora_rank=%d",
        args.condition_extract,
        args.condition_tokens,
        args.condition_inject,
        args.cond_dim,
        args.num_heads,
        args.set_layers,
        args.hyper_lora_rank)
    logger.info(
        "=> Parameter tokens: tensors=%d tokens=%d token_size=%d "
        "valid_elements=%d padded_elements=%d",
        meta["num_tensors"],
        meta["num_tokens"],
        meta["token_size"],
        sum(token["valid_elements"] for token in meta["tokens"]),
        meta["num_tokens"] * meta["token_size"]
        - sum(token["valid_elements"] for token in meta["tokens"]))
    logger.info(
        "=> Training pairs: count=%d max_guide_codes=%d",
        len(examples),
        args.max_guide_codes)
    for item in examples[:8]:
        logger.info(
            "   pair[%d] codes=%s target=%s code_shape=(%d,%d)",
            item["index"],
            item["code_path"],
            item["target_checkpoint"],
            item["num_codes"],
            item["guide_codes"].size(1))
    if len(examples) > 8:
        logger.info("   ... %d more pairs", len(examples) - 8)
    logger.info(
        "=> Optimizer: AdamW lr=%g eta_min=%g weight_decay=%g "
        "grad_clip=%g",
        args.lr,
        args.eta_min,
        args.weight_decay,
        args.grad_clip)
    logger.info(
        "=> Scheduler: warmup_cosine epochs=%d steps_per_epoch=%d "
        "total_steps=%d warmup_steps=%d warmup_ratio=%g",
        args.epochs,
        args.steps_per_epoch,
        total_steps,
        warmup_steps,
        args.warmup_ratio)
    logger.info(
        "=> Loss: velocity_mse=1.0 endpoint_param_mse=%g",
        args.lambda_endpoint)
    logger.info(
        "=> Parameters: total=%s trainable=%s frozen=%s",
        f"{total:,}", f"{trainable:,}", f"{frozen:,}")
    log_parameter_table(model)


def main():
    args = parse_args()
    seed_everything(args.seed)
    exp_dir = Path(args.exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    artifact_dir = exp_dir / "artifacts"
    tensorboard_dir = exp_dir / "tensorboard"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    (exp_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    log_experiment_header(args, exp_dir=exp_dir)
    logger.info("=> TensorBoard directory: %s", tensorboard_dir)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("config/args", json.dumps(
        vars(args), indent=2, sort_keys=True), global_step=0)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu
        else "cpu")

    (base_state, target_states, decoder_cfg, meta, norm_stats, examples,
     code_dim, meta_tensors) = load_training_examples(args, device)
    summarize_decoder_tensors(target_states[0])
    summarize_norm_stats(norm_stats)
    if len(target_states) == 1:
        save_artifacts(
            artifact_dir, base_state, target_states[0], meta, norm_stats,
            decoder_cfg)
    else:
        save_artifacts(
            artifact_dir, base_state, target_states[0], meta, norm_stats,
            decoder_cfg)
        (artifact_dir / "data_pairs.json").write_text(
            json.dumps([{
                "code_path": item["code_path"],
                "target_checkpoint": item["target_checkpoint"],
                "num_codes": item["num_codes"],
            } for item in examples], indent=2),
            encoding="utf-8")
    logger.info(
        "Prepared %d training pairs, %d tensors, %d parameter tokens, "
        "token_size=%d",
        len(examples), meta["num_tensors"], meta["num_tokens"],
        meta["token_size"])

    model = build_system(args, code_dim, meta).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(2, args.epochs * args.steps_per_epoch)
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = max(1, min(warmup_steps, total_steps - 1))
    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=total_steps, T_warmup=warmup_steps,
        eta_min=args.eta_min)
    log_training_preamble(
        args, device, decoder_cfg, meta, examples, model, total_steps,
        warmup_steps)
    total_params, trainable_params, frozen_params = count_parameters(model)
    writer.add_scalar("meta/decoder_target_params",
                      sum(value.numel() for value in target_states[0].values()), 0)
    writer.add_scalar("meta/training_pairs", len(examples), 0)
    writer.add_scalar("meta/param_tokens", meta["num_tokens"], 0)
    writer.add_scalar("meta/valid_param_elements",
                      sum(token["valid_elements"] for token in meta["tokens"]),
                      0)
    writer.add_scalar("meta/padded_param_elements",
                      meta["num_tokens"] * meta["token_size"]
                      - sum(token["valid_elements"]
                            for token in meta["tokens"]), 0)
    writer.add_scalar("meta/model_total_params", total_params, 0)
    writer.add_scalar("meta/model_trainable_params", trainable_params, 0)
    writer.add_scalar("meta/model_frozen_params", frozen_params, 0)

    best_loss = float("inf")
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_velocity_loss = 0.0
            epoch_endpoint_loss = 0.0
            for _ in range(args.steps_per_epoch):
                global_step += 1
                example = random.choice(examples)
                theta1 = example["theta1"].to(device, non_blocking=True)
                token_mask = example["token_mask"].to(device, non_blocking=True)
                guide_codes = example["guide_codes"].to(device, non_blocking=True)
                theta0 = torch.zeros_like(theta1)
                velocity_target = theta1
                t = torch.rand((), device=device).clamp_(1e-4, 1.0 - 1e-4)
                theta_t = theta0 + t * (theta1 - theta0)
                pred_v = model(theta_t, t, meta_tensors, guide_codes)
                velocity_loss = masked_mse(pred_v, velocity_target, token_mask)
                endpoint = theta_t + (1.0 - t) * pred_v
                endpoint_loss = masked_mse(endpoint, theta1, token_mask)
                loss = velocity_loss + args.lambda_endpoint * endpoint_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = None
                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                lr = optimizer.param_groups[0]["lr"]

                loss_value = loss.item()
                velocity_value = velocity_loss.item()
                endpoint_value = endpoint_loss.item()
                epoch_loss += loss_value
                epoch_velocity_loss += velocity_value
                epoch_endpoint_loss += endpoint_value
                writer.add_scalar("train_step/loss", loss_value, global_step)
                writer.add_scalar(
                    "train_step/velocity_mse", velocity_value, global_step)
                writer.add_scalar(
                    "train_step/endpoint_param_mse",
                    endpoint_value,
                    global_step)
                writer.add_scalar("train_step/lr", lr, global_step)
                writer.add_scalar("train_step/t", t.item(), global_step)
                writer.add_scalar(
                    "train_step/pair_index", example["index"], global_step)
                if grad_norm is not None:
                    writer.add_scalar(
                        "train_step/grad_norm",
                        float(grad_norm.detach().cpu()),
                        global_step)

            denom = max(args.steps_per_epoch, 1)
            avg_loss = epoch_loss / denom
            avg_velocity_loss = epoch_velocity_loss / denom
            avg_endpoint_loss = epoch_endpoint_loss / denom
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch [%d/%d] loss=%.8f velocity=%.8f endpoint=%.8f lr=%.6g",
                epoch, args.epochs, avg_loss, avg_velocity_loss,
                avg_endpoint_loss, lr)
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/velocity_mse", avg_velocity_loss, epoch)
            writer.add_scalar(
                "train/endpoint_param_mse", avg_endpoint_loss, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            save_checkpoint(
                ckpt_dir / "last.pth", model, optimizer, scheduler, args,
                epoch, global_step, avg_loss, meta)
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    ckpt_dir / "best_loss.pth", model, optimizer, scheduler,
                    args, epoch, global_step, avg_loss, meta)
            writer.add_scalar("train/best_loss", best_loss, epoch)
            if args.save_every and epoch % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch}.pth", model, optimizer,
                    scheduler, args, epoch, global_step, avg_loss, meta)
        logger.info("Finished training. best_loss=%.8f", best_loss)
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
