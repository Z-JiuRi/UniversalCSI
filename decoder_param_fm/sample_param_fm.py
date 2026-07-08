import argparse
import json
import sys
from pathlib import Path

import torch

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
    denormalize_state,
    extract_decoder_state,
    load_codes,
    load_param_meta,
    masked_mse,
    meta_tensors_from_meta,
    normalize_target_delta,
    tokens_to_norm_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample generated decoder parameters from a trained FM.")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--guide_code_path", type=str, default="")
    parser.add_argument("--target_checkpoint", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_guide_codes", type=int, default=0)
    return parser.parse_args()


def build_system(train_args, code_dim, meta):
    condition_extractor = ConditionExtractor(
        method=train_args["condition_extract"],
        code_dim=code_dim,
        num_tokens=train_args["condition_tokens"],
        d_model=train_args["cond_dim"],
        num_heads=train_args["num_heads"],
        num_layers=train_args["set_layers"],
    )
    condition_encoder = ConditionEncoder(
        code_dim=code_dim,
        hidden_dim=train_args["hidden_dim"],
        cond_dim=train_args["cond_dim"],
    )
    param_fm = ParamFM(
        num_tensors=meta["num_tensors"],
        max_layer_id=meta["max_layer_id"],
        max_token_offset=meta["max_token_offset"],
        token_size=train_args["token_size"],
        hidden_dim=train_args["hidden_dim"],
        num_blocks=train_args["num_blocks"],
        time_dim=train_args["time_dim"],
        cond_dim=train_args["cond_dim"],
        condition_inject=train_args["condition_inject"],
        num_heads=train_args["num_heads"],
        hyper_lora_rank=train_args["hyper_lora_rank"],
        dropout=train_args["dropout"],
    )
    return DecoderParamFMSystem(condition_extractor, condition_encoder, param_fm)


@torch.no_grad()
def euler_sample(model, meta_tensors, guide_codes, num_tokens, token_size,
                 ode_steps, device):
    theta = torch.zeros(num_tokens, token_size, device=device)
    dt = 1.0 / ode_steps
    model.eval()
    cond_tokens, cond_mask, global_cond = model.encode_condition(guide_codes)
    for step in range(ode_steps):
        t = torch.tensor((step + 0.5) * dt, device=device)
        velocity = model.param_fm(
            theta, t, meta_tensors, cond_tokens, cond_mask, global_cond)
        theta = theta + dt * velocity
    return theta


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    train_args = json.loads((exp_dir / "args.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        exp_dir / "checkpoints" / "best_loss.pth")
    output_path = Path(args.output) if args.output else (
        exp_dir / "generated" / "generated_decoder.pth")
    guide_code_path = args.guide_code_path or train_args.get("guide_code_path", "")
    if not guide_code_path:
        raise ValueError(
            "--guide_code_path is required when the FM was trained with "
            "--data_txt")
    artifact_dir = exp_dir / "artifacts"

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu
        else "cpu")
    base_state = torch.load(
        artifact_dir / "theta_base.pt", weights_only=True, map_location="cpu")
    if args.target_checkpoint:
        target_state = extract_decoder_state(args.target_checkpoint)
    elif train_args.get("target_checkpoint"):
        target_state = torch.load(
            artifact_dir / "theta_star.pt",
            weights_only=True,
            map_location="cpu")
    else:
        target_state = None
    norm_stats = torch.load(
        artifact_dir / "norm_stats.pt", weights_only=True, map_location="cpu")
    meta = load_param_meta(artifact_dir / "param_meta.json")
    guide_codes = load_codes(
        guide_code_path, max_samples=args.max_guide_codes).to(device)

    model = build_system(train_args, guide_codes.size(1), meta).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    meta_tensors = meta_tensors_from_meta(meta, device=device)

    theta_tokens = euler_sample(
        model, meta_tensors, guide_codes, meta["num_tokens"],
        meta["token_size"], args.ode_steps, device)
    norm_state = tokens_to_norm_state(theta_tokens, meta)
    generated_state = denormalize_state(base_state, norm_state, norm_stats)

    param_mse = None
    if target_state is not None:
        target_norm = normalize_target_delta(base_state, target_state, norm_stats)
        target_tokens = []
        masks = []
        for token in meta["tokens"]:
            name = token["tensor_name"]
            flat = target_norm[name].flatten().to(device)
            start = token["token_offset"] * meta["token_size"]
            valid = token["valid_elements"]
            tok = theta_tokens.new_zeros(meta["token_size"])
            mask = theta_tokens.new_zeros(meta["token_size"])
            tok[:valid] = flat[start:start + valid]
            mask[:valid] = 1.0
            target_tokens.append(tok)
            masks.append(mask)
        target_tokens = torch.stack(target_tokens)
        masks = torch.stack(masks)
        param_mse = masked_mse(theta_tokens, target_tokens, masks).item()

    output = {
        "decoder_state_dict": {
            key: value.detach().cpu() for key, value in generated_state.items()
        },
        "state_dict": {
            f"decoder.{key}": value.detach().cpu()
            for key, value in generated_state.items()
        },
        "source_checkpoint": str(checkpoint_path),
        "guide_code_path": str(guide_code_path),
        "target_checkpoint": str(args.target_checkpoint),
        "ode_steps": args.ode_steps,
        "train_args": train_args,
    }
    if param_mse is not None:
        output["param_mse_normalized"] = param_mse
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"saved={output_path}")
    if param_mse is not None:
        print(f"param_mse_normalized={param_mse:.8f}")


if __name__ == "__main__":
    main()
