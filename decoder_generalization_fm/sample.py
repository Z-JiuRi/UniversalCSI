import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoder_generalization_fm.models import (  # noqa: E402
    ConditionEncoder,
    ConditionExtractor,
    DecoderGeneralizationFM,
    ParamFM,
)
from decoder_generalization_fm.param_utils import (  # noqa: E402
    denormalize_state,
    load_codes,
    meta_tensors_from_meta,
    tokens_to_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample full decoder parameters from a trained FM.")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--condition_exp_dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--ode_steps", type=int, default=16)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--max_condition_codes", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
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
    return DecoderGeneralizationFM(condition_extractor, condition_encoder, param_fm)


@torch.no_grad()
def sample_tokens(model, meta_tensors, condition_codes, meta, ode_steps, device):
    theta = torch.randn(
        meta["num_tokens"], meta["token_size"], device=device)
    dt = 1.0 / max(ode_steps, 1)
    model.eval()
    cond_tokens, cond_mask, global_cond = model.encode_condition(condition_codes)
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        exp_dir / "checkpoints" / "best_loss.pth")
    output = Path(args.output) if args.output else (
        exp_dir / "generated" / f"{Path(args.condition_exp_dir).name}_decoder.pth")
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.sample_seed)
    torch.cuda.manual_seed_all(args.sample_seed)

    meta = json.loads(
        (exp_dir / "artifacts" / "param_meta.json").read_text(encoding="utf-8"))
    stats = torch.load(
        exp_dir / "artifacts" / "train_tensor_zscore_stats.pt",
        weights_only=True,
        map_location="cpu")
    condition_exp_dir = Path(args.condition_exp_dir)
    condition_codes = load_codes(
        condition_exp_dir / "codewords" / "train_code.pt",
        max_samples=args.max_condition_codes
        or train_args.get("max_condition_codes", 0))
    model = build_system(train_args, condition_codes.size(1), meta).to(device)
    ckpt = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    meta_tensors = meta_tensors_from_meta(meta, device=device)
    theta_tokens = sample_tokens(
        model, meta_tensors, condition_codes.to(device), meta, args.ode_steps, device)
    norm_state = tokens_to_state(theta_tokens, meta)
    generated_state = denormalize_state(norm_state, stats)
    out = {
        "decoder_state_dict": {
            key: value.detach().cpu() for key, value in generated_state.items()
        },
        "state_dict": {
            f"decoder.{key}": value.detach().cpu()
            for key, value in generated_state.items()
        },
        "source_checkpoint": str(checkpoint),
        "condition_exp_dir": str(condition_exp_dir),
        "ode_steps": args.ode_steps,
        "sample_seed": args.sample_seed,
        "train_args": train_args,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
