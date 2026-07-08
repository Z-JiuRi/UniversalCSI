import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoder_generalization_fm.dataset import parse_data_txt, split_entries  # noqa: E402
from decoder_generalization_fm.models import (  # noqa: E402
    ConditionEncoder,
    ConditionExtractor,
    DecoderGeneralizationFM,
    ParamFM,
)
from decoder_generalization_fm.param_utils import (  # noqa: E402
    build_param_meta,
    clone_decoder_with_state,
    denormalize_state,
    extract_decoder_state,
    load_codes,
    load_csi,
    load_json,
    load_or_compute_stats,
    masked_mse,
    meta_tensors_from_meta,
    normalize_state,
    state_to_tokens,
    tokens_to_state,
    validate_compatible_states,
    write_json,
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
        description="Train FM to generate full transnet decoder parameters.")
    parser.add_argument("--data_txt", default="decoder_generalization_fm/data/data.txt")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--stats_cache", default="")
    parser.add_argument("--csi_path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="number of experiment-level samples per optimizer step")
    parser.add_argument("--steps_per_epoch", type=int, default=0,
                        help="random-sampling steps per epoch; 0=iterate all examples")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--token_size", type=int, default=512)
    parser.add_argument("--condition_extract", choices=["random", "svd", "set_transformer"],
                        default="svd")
    parser.add_argument("--condition_inject", choices=["film", "cross_attention", "hyper_lora"],
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
    parser.add_argument("--t_eps", type=float, default=1e-4)
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--max_condition_codes", type=int, default=0)

    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--eval_max_entries", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    logger.info("Random seed set to %s", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(args):
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.is_available() and not args.cpu:
        return torch.device("cuda")
    return torch.device("cpu")


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
    return DecoderGeneralizationFM(
        condition_extractor, condition_encoder, param_fm)


def prepare_examples(entries, stats, meta, reference_state, max_condition_codes):
    examples = []
    code_dim = None
    for idx, entry in enumerate(entries):
        state = extract_decoder_state(entry.checkpoint_path)
        validate_compatible_states(reference_state, state, entry.exp_dir)
        norm_state = normalize_state(state, stats)
        theta1, token_mask, meta_tensors = state_to_tokens(norm_state, meta)
        codes = load_codes(entry.code_path, max_samples=max_condition_codes)
        if code_dim is None:
            code_dim = codes.size(1)
        elif code_dim != codes.size(1):
            raise ValueError(
                f"code dim mismatch for {entry.code_path}: "
                f"{codes.size(1)} vs {code_dim}")
        cfg = load_json(entry.args_json)
        examples.append({
            "index": idx,
            "split": entry.split,
            "exp_dir": str(entry.exp_dir),
            "encoder": entry.encoder,
            "seed": entry.seed,
            "args_json": str(entry.args_json),
            "code_path": str(entry.code_path),
            "checkpoint_path": str(entry.checkpoint_path),
            "condition_codes": codes,
            "theta1": theta1,
            "token_mask": token_mask,
            "meta_tensors": meta_tensors,
            "num_codes": int(codes.size(0)),
            "cfg": cfg,
        })
    return examples, code_dim


@torch.no_grad()
def sample_tokens(model, meta_tensors, condition_codes, num_tokens, token_size,
                  steps, device):
    theta = torch.randn(num_tokens, token_size, device=device)
    dt = 1.0 / max(steps, 1)
    model.eval()
    cond_tokens, cond_mask, global_cond = model.encode_condition(condition_codes)
    for step in range(steps):
        t = torch.tensor((step + 0.5) * dt, device=device)
        velocity = model.param_fm(
            theta, t, meta_tensors, cond_tokens, cond_mask, global_cond)
        theta = theta + dt * velocity
    return theta


@torch.no_grad()
def evaluate_decoder_nmse(decoder, codes, csi, batch_size, device):
    decoder.eval()
    error_sum = torch.zeros((), dtype=torch.float64, device=device)
    power_sum = torch.zeros((), dtype=torch.float64, device=device)
    n = min(codes.size(0), csi.size(0))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        code_batch = codes[start:end].to(device, non_blocking=True)
        csi_batch = csi[start:end].to(device, non_blocking=True)
        pred = decoder(code_batch)
        error_sum += (pred - csi_batch).double().pow(2).sum()
        power_sum += csi_batch.double().pow(2).sum()
    nmse = 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))
    return float(nmse.detach().cpu())


@torch.no_grad()
def evaluate_split(model, examples, stats, meta, meta_tensors, args, device):
    if args.eval_max_entries:
        examples = examples[:args.eval_max_entries]
    rows = []
    for example in examples:
        condition_codes = example["condition_codes"].to(device, non_blocking=True)
        theta_tokens = sample_tokens(
            model, meta_tensors, condition_codes, meta["num_tokens"],
            meta["token_size"], args.ode_steps, device)
        norm_state = tokens_to_state(theta_tokens, meta)
        generated_state = denormalize_state(norm_state, stats)
        decoder = clone_decoder_with_state(
            example["args_json"], generated_state, device)
        cfg = example["cfg"]
        csi_path = args.csi_path or cfg.get("train_path")
        csi = load_csi(
            csi_path,
            cfg.get("channel", 2),
            cfg.get("nt", 32),
            cfg.get("nc", 32),
            max_samples=args.eval_max_samples)
        codes = load_codes(
            example["code_path"],
            max_samples=args.eval_max_samples)
        nmse = evaluate_decoder_nmse(
            decoder, codes, csi, args.eval_batch_size, device)
        rows.append({
            "exp_dir": example["exp_dir"],
            "encoder": example["encoder"],
            "seed": example["seed"],
            "nmse_db": nmse,
        })
    mean = sum(row["nmse_db"] for row in rows) / max(len(rows), 1)
    return {"mean_nmse_db": mean, "n": len(rows), "rows": rows}


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


def log_preamble(args, train_examples, test_examples, meta, model,
                 stats_cache, stats_from_cache, batches_per_epoch,
                 total_steps):
    total, trainable, frozen = count_parameters(model)
    logger.info("=> Data txt: %s", args.data_txt)
    logger.info("=> Stats cache: %s from_cache=%s", stats_cache, stats_from_cache)
    logger.info("=> Train examples: %d", len(train_examples))
    logger.info("=> Test examples: %d", len(test_examples))
    logger.info("=> Training: batch_size=%d steps_per_epoch=%d batches_per_epoch=%d total_steps=%d mode=%s",
                args.batch_size, args.steps_per_epoch, batches_per_epoch, total_steps,
                "random_sampling" if args.steps_per_epoch > 0 else "iterate_all")
    logger.info("=> Condition: extract=%s tokens=%d inject=%s",
                args.condition_extract, args.condition_tokens,
                args.condition_inject)
    logger.info("=> Param tokens: tensors=%d tokens=%d token_size=%d",
                meta["num_tensors"], meta["num_tokens"], meta["token_size"])
    logger.info("=> Parameters: total=%s trainable=%s frozen=%s",
                f"{total:,}", f"{trainable:,}", f"{frozen:,}")
    for item in train_examples[:8]:
        logger.info("   train %s encoder=%s seed=%s codes=%d",
                    item["exp_dir"], item["encoder"], item["seed"],
                    item["num_codes"])
    for item in test_examples[:8]:
        logger.info("   test  %s encoder=%s seed=%s codes=%d",
                    item["exp_dir"], item["encoder"], item["seed"],
                    item["num_codes"])
    log_parameter_table(model)


def main():
    args = parse_args()
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    exp_dir = Path(args.exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    artifact_dir = exp_dir / "artifacts"
    tensorboard_dir = exp_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)
    write_json(exp_dir / "args.json", vars(args))
    log_experiment_header(args, exp_dir=exp_dir)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("config/args", json.dumps(vars(args), indent=2), 0)
    seed_everything(args.seed)
    device = resolve_device(args)
    logger.info("=> Device: %s", device)

    entries = parse_data_txt(args.data_txt)
    train_entries, test_entries = split_entries(entries)
    train_states = [extract_decoder_state(item.checkpoint_path)
                    for item in train_entries]
    reference_state = train_states[0]
    for entry, state in zip(train_entries, train_states):
        validate_compatible_states(reference_state, state, entry.exp_dir)
    for entry in test_entries:
        validate_compatible_states(
            reference_state, extract_decoder_state(entry.checkpoint_path),
            entry.exp_dir)

    stats_cache = args.stats_cache or str(
        artifact_dir / "train_tensor_zscore_stats.pt")
    stats, stats_from_cache = load_or_compute_stats(stats_cache, train_states)
    torch.save(stats, artifact_dir / "train_tensor_zscore_stats.pt")
    meta = build_param_meta(reference_state, args.token_size)
    write_json(artifact_dir / "param_meta.json", meta)
    write_json(artifact_dir / "data_entries.json", [{
        "split": item.split,
        "exp_dir": str(item.exp_dir),
        "encoder": item.encoder,
        "seed": item.seed,
    } for item in entries])

    train_examples, code_dim = prepare_examples(
        train_entries, stats, meta, reference_state, args.max_condition_codes)
    test_examples, test_code_dim = prepare_examples(
        test_entries, stats, meta, reference_state, args.max_condition_codes)
    if code_dim != test_code_dim:
        raise ValueError(f"train/test code_dim mismatch: {code_dim} vs {test_code_dim}")
    meta_tensors = meta_tensors_from_meta(meta, device=device)

    model = build_system(args, code_dim, meta).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_random = args.steps_per_epoch > 0
    if use_random:
        batches_per_epoch = args.steps_per_epoch
    else:
        batches_per_epoch = max(
            1, (len(train_examples) + args.batch_size - 1) // args.batch_size)
    total_steps = max(2, args.epochs * batches_per_epoch)
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = max(1, min(warmup_steps, total_steps - 1))
    scheduler = WarmUpCosineAnnealingLR(
        optimizer, T_max=total_steps, T_warmup=warmup_steps,
        eta_min=args.eta_min)
    log_preamble(args, train_examples, test_examples, meta, model,
                 stats_cache, stats_from_cache, batches_per_epoch,
                 total_steps)

    best_loss = float("inf")
    best_nmse = float("inf")
    global_step = 0
    history = []
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_velocity = 0.0
            epoch_endpoint = 0.0
            epoch_examples = list(train_examples)
            random.shuffle(epoch_examples)
            num_batches = 0
            if use_random:
                steps = args.steps_per_epoch
            else:
                steps = len(epoch_examples)
                if args.batch_size > 1:
                    steps = (steps + args.batch_size - 1) // args.batch_size
            for step_i in range(steps):
                global_step += 1
                if use_random:
                    example = random.choice(epoch_examples)
                    batch_examples = [example]
                else:
                    start_idx = step_i * args.batch_size
                    batch_examples = epoch_examples[start_idx:start_idx + args.batch_size]
                batch_losses = []
                batch_velocity_losses = []
                batch_endpoint_losses = []
                for example in batch_examples:
                    theta1 = example["theta1"].to(device, non_blocking=True)
                    token_mask = example["token_mask"].to(device, non_blocking=True)
                    condition_codes = example["condition_codes"].to(device, non_blocking=True)
                    theta0 = torch.randn_like(theta1)
                    t = torch.rand((), device=device)
                    t = t * (1.0 - 2.0 * args.t_eps) + args.t_eps
                    theta_t = (1.0 - t) * theta0 + t * theta1
                    velocity_target = theta1 - theta0
                    pred_v = model(theta_t, t, meta_tensors, condition_codes)
                    velocity_loss = masked_mse(pred_v, velocity_target, token_mask)
                    endpoint = theta_t + (1.0 - t) * pred_v
                    endpoint_loss = masked_mse(endpoint, theta1, token_mask)
                    sample_loss = velocity_loss + args.lambda_endpoint * endpoint_loss
                    batch_losses.append(sample_loss)
                    batch_velocity_losses.append(velocity_loss.detach())
                    batch_endpoint_losses.append(endpoint_loss.detach())
                loss = torch.stack(batch_losses).mean()
                velocity_loss_value = torch.stack(batch_velocity_losses).mean()
                endpoint_loss_value = torch.stack(batch_endpoint_losses).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = None
                if args.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                lr = optimizer.param_groups[0]["lr"]

                num_batches += 1
                epoch_loss += loss.item()
                epoch_velocity += velocity_loss_value.item()
                epoch_endpoint += endpoint_loss_value.item()
                writer.add_scalar("train_step/loss", loss.item(), global_step)
                writer.add_scalar(
                    "train_step/velocity_mse", velocity_loss_value.item(), global_step)
                writer.add_scalar(
                    "train_step/endpoint_mse", endpoint_loss_value.item(), global_step)
                writer.add_scalar("train_step/lr", lr, global_step)
                writer.add_scalar(
                    "train_step/batch_size", len(batch_examples), global_step)
                if grad_norm is not None:
                    writer.add_scalar(
                        "train_step/grad_norm",
                        float(grad_norm.detach().cpu()), global_step)

            denom = max(num_batches, 1)
            avg_loss = epoch_loss / denom
            avg_velocity = epoch_velocity / denom
            avg_endpoint = epoch_endpoint / denom
            lr = optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "loss": avg_loss,
                "velocity_mse": avg_velocity,
                "endpoint_mse": avg_endpoint,
                "lr": lr,
            }
            logger.info(
                "Epoch [%d/%d] loss=%.6e velocity=%.6e endpoint=%.6e lr=%.6e",
                epoch, args.epochs, avg_loss, avg_velocity, avg_endpoint, lr)
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/velocity_mse", avg_velocity, epoch)
            writer.add_scalar("train/endpoint_mse", avg_endpoint, epoch)
            writer.add_scalar("train/lr", lr, epoch)

            save_checkpoint(
                ckpt_dir / "last.pth", model, optimizer, scheduler, args,
                epoch, global_step, avg_loss, meta)
            if args.save_every and epoch % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch}.pth", model, optimizer,
                    scheduler, args, epoch, global_step, avg_loss, meta)
            if args.eval_every and epoch % args.eval_every == 0:
                test_eval = evaluate_split(
                    model, test_examples, stats, meta, meta_tensors, args, device)
                record["eval"] = {"test": test_eval}
                writer.add_scalar(
                    "eval/test_mean_nmse_db",
                    test_eval["mean_nmse_db"], epoch)
                logger.info(
                    "Epoch [%d/%d] generated_decoder_nmse "
                    "test_mean=%.6e dB n=%d",
                    epoch, args.epochs,
                    test_eval["mean_nmse_db"], test_eval["n"])
                for row in test_eval["rows"]:
                    logger.info("   test_nmse %.6e dB %s",
                                row["nmse_db"], row["exp_dir"])
                if test_eval["mean_nmse_db"] < best_nmse:
                    best_nmse = test_eval["mean_nmse_db"]
                    best_loss = avg_loss
                    save_checkpoint(
                        ckpt_dir / "best_nmse.pth", model, optimizer, scheduler,
                        args, epoch, global_step, avg_loss, meta)
                    logger.info("=> New best test_nmse: %.6e", best_nmse)
                trace_entry = {"metrics": test_eval}
                record["eval_trace"] = trace_entry
                write_json(exp_dir / "eval_trace.json", trace_entry)
            history.append(record)
        write_json(exp_dir / "history.json", history)
        logger.info("Finished training. best_loss=%.6e best_test_nmse=%.6e",
                    best_loss, best_nmse)
    finally:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
