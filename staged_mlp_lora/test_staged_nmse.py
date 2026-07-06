#!/usr/bin/env python
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from decoder_lora.train_decoder_lora import (  # noqa: E402
    DecoderLoraSystem,
    build_code_adapter,
    inject_decoder_lora,
    load_decoder_from_checkpoint,
    load_lora_state,
)
from staged_mlp_lora.train_affine_mlp_mapper import (  # noqa: E402
    AffineResidualMLPMapper,
)
from utils.statics import evaluator, nmse_from_sums  # noqa: E402


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tensor(path):
    return torch.load(path, weights_only=True, map_location="cpu").float()


def load_csi(path, channel, nt, nc):
    csi = load_tensor(path)
    if csi.ndim == 2:
        csi = csi.view(-1, channel, nt, nc)
    if csi.ndim != 4 or tuple(csi.shape[1:]) != (channel, nt, nc):
        raise ValueError(
            f"CSI should have shape (N,{channel},{nt},{nc}), got "
            f"{tuple(csi.shape)}")
    return csi


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def decoder_args(args):
    return Namespace(
        decoder_checkpoint=args.decoder_checkpoint,
        decoder_args_json=args.decoder_args_json,
        decoder_name=args.decoder_name,
        decoder_cr=args.decoder_cr,
        decoder_d_model=args.decoder_d_model,
        decoder_dim_feedforward=args.decoder_dim_feedforward,
        decoder_channel=args.decoder_channel,
        decoder_nt=args.decoder_nt,
        decoder_nc=args.decoder_nc,
        decoder_hidden=args.decoder_hidden,
        decoder_num_blocks=args.decoder_num_blocks,
    )


def lora_args_from_checkpoint(path):
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    saved = ckpt.get("args", {})
    return Namespace(
        lora_target=saved.get("lora_target", "fc_ffn"),
        lora_rank=saved.get("lora_rank", 8),
        lora_alpha=saved.get("lora_alpha"),
        lora_dropout=saved.get("lora_dropout", 0.0),
        fc_lora_rank=saved.get("fc_lora_rank"),
        ffn_lora_rank=saved.get("ffn_lora_rank"),
        fc_lora_alpha=saved.get("fc_lora_alpha"),
        ffn_lora_alpha=saved.get("ffn_lora_alpha"),
        code_adapter=saved.get("code_adapter", "none"),
        code_lowrank_rank=saved.get("code_lowrank_rank", 0),
        code_mlp_hidden=saved.get("code_mlp_hidden", 0),
        code_gate_lr=saved.get("code_gate_lr", 0.1),
        code_gate_mlp=saved.get("code_gate_mlp", 0.1),
        code_adapter_dropout=saved.get("code_adapter_dropout", 0.0),
    )


def load_mapper(path, device):
    ckpt = torch.load(path, weights_only=True, map_location=device)
    saved = ckpt.get("args", {})
    state = ckpt["state_dict"]
    weight = state["alignment_weight"].to(device)
    bias = state["alignment_bias"].to(device)
    model = AffineResidualMLPMapper(
        weight=weight,
        bias=bias,
        hidden_dim=int(saved.get("hidden_dim", 1024)),
        num_blocks=int(saved.get("num_blocks", 4)),
        dropout=float(saved.get("dropout", 0.0)),
        residual_scale=float(saved.get("residual_scale", 1.0)),
        use_final_norm=not bool(saved.get("no_final_norm", False)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, saved, ckpt.get("best", {})


def load_base_decoder(args, device):
    decoder, cfg = load_decoder_from_checkpoint(decoder_args(args), device)
    return decoder.eval(), cfg


def load_lora_decoder(args, lora_checkpoint, code_dim, device):
    lora_cfg = lora_args_from_checkpoint(lora_checkpoint)
    decoder, dec_cfg = load_base_decoder(args, device)
    inject_decoder_lora(
        decoder,
        target=lora_cfg.lora_target,
        rank=lora_cfg.lora_rank,
        alpha=lora_cfg.lora_alpha,
        dropout=lora_cfg.lora_dropout,
        fc_rank=lora_cfg.fc_lora_rank,
        ffn_rank=lora_cfg.ffn_lora_rank,
        fc_alpha=lora_cfg.fc_lora_alpha,
        ffn_alpha=lora_cfg.ffn_lora_alpha)
    code_adapter = build_code_adapter(lora_cfg, dim=code_dim, device=device)
    model = DecoderLoraSystem(decoder, code_adapter).to(device).eval()
    ckpt, missing = load_lora_state(model, lora_checkpoint, device)
    return model, dec_cfg, lora_cfg, ckpt.get("best", {}), missing


def transform_code(mapper, source_code, batch_size, workers, device):
    loader = DataLoader(
        TensorDataset(source_code),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    chunks = []
    with torch.no_grad():
        for (code,) in loader:
            code = code.to(device, non_blocking=True)
            chunks.append(mapper(code).detach().cpu())
    return torch.cat(chunks, dim=0)


def evaluate_decoder(name, model, code, csi, batch_size, workers, device):
    loader = DataLoader(
        TensorDataset(code, csi),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    with torch.no_grad():
        for batch_code, batch_csi in loader:
            batch_code = batch_code.to(device, non_blocking=True)
            batch_csi = batch_csi.to(device, non_blocking=True)
            pred = model(batch_code)
            mse = F.mse_loss(pred, batch_csi)
            error_sum, power_sum = evaluator(pred, batch_csi)
            total_error += error_sum
            total_power += power_sum
            total_mse += float(mse.detach().cpu()) * batch_code.size(0)
            total_n += batch_code.size(0)
    nmse_db = nmse_from_sums(total_error, total_power)
    nmse_linear = total_error / total_power.clamp_min(1e-12)
    return {
        "name": name,
        "n": int(total_n),
        "mse_loss": total_mse / max(total_n, 1),
        "error_sum": float(total_error.detach().cpu()),
        "power_sum": float(total_power.detach().cpu()),
        "nmse_linear": float(nmse_linear.detach().cpu()),
        "nmse_db": float(nmse_db.detach().cpu()),
    }


def code_metrics(mapped, target):
    mse = F.mse_loss(mapped, target).item()
    cos = F.cosine_similarity(mapped, target, dim=1).mean().item()
    return {
        "mapped_to_teacher_mse": mse,
        "mapped_to_teacher_cos": cos,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate raw code -> mapper -> LoRA decoder true NMSE.")
    parser.add_argument("--source_code", required=True)
    parser.add_argument("--target_code",
                        default="exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt")
    parser.add_argument("--csi_path",
                        default="/storage/hujiacong/zxd/datasets/cost2100/in_train.pt")
    parser.add_argument("--mapper_checkpoint", required=True)
    parser.add_argument("--lora_checkpoint", required=True)
    parser.add_argument("--decoder_checkpoint",
                        default="exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth")
    parser.add_argument("--decoder_args_json",
                        default="exps/COST2100/in/seed42/transnet_transnet/args.json")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--save_mapped_code", default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--decoder_name", default=None)
    parser.add_argument("--decoder_cr", type=int, default=None)
    parser.add_argument("--decoder_d_model", type=int, default=None)
    parser.add_argument("--decoder_dim_feedforward", type=int, default=None)
    parser.add_argument("--decoder_channel", type=int, default=None)
    parser.add_argument("--decoder_nt", type=int, default=None)
    parser.add_argument("--decoder_nc", type=int, default=None)
    parser.add_argument("--decoder_hidden", type=int, default=None)
    parser.add_argument("--decoder_num_blocks", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device(args.gpu, args.cpu)
    dec_cfg_json = load_json(args.decoder_args_json)
    channel = args.decoder_channel or dec_cfg_json.get("channel", 2)
    nt = args.decoder_nt or dec_cfg_json.get("nt", 32)
    nc = args.decoder_nc or dec_cfg_json.get("nc", 32)

    source_code = load_tensor(args.source_code)
    target_code = load_tensor(args.target_code)
    csi = load_csi(args.csi_path, channel, nt, nc)
    n = min(source_code.size(0), target_code.size(0), csi.size(0))
    if args.max_samples and args.max_samples > 0:
        n = min(n, args.max_samples)
    source_code = source_code[:n].contiguous()
    target_code = target_code[:n].contiguous()
    csi = csi[:n].contiguous()
    if source_code.ndim != 2 or target_code.ndim != 2:
        raise ValueError("source_code and target_code must be 2D tensors")
    if source_code.shape != target_code.shape:
        raise ValueError(
            f"source/target shape mismatch: {source_code.shape} vs "
            f"{target_code.shape}")

    mapper, mapper_args, mapper_best = load_mapper(args.mapper_checkpoint, device)
    mapped_code = transform_code(
        mapper, source_code, args.batch_size, args.workers, device)
    if args.save_mapped_code:
        save_path = Path(args.save_mapped_code)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mapped_code, save_path)

    base_decoder, dec_cfg = load_base_decoder(args, device)
    lora_decoder, _lora_dec_cfg, lora_args, lora_best, missing = load_lora_decoder(
        args, args.lora_checkpoint, source_code.size(1), device)

    raw_metrics = evaluate_decoder(
        "raw_source_code_base_decoder",
        base_decoder,
        source_code,
        csi,
        args.batch_size,
        args.workers,
        device)
    mapped_base_metrics = evaluate_decoder(
        "mapped_code_base_decoder",
        base_decoder,
        mapped_code,
        csi,
        args.batch_size,
        args.workers,
        device)
    mapped_lora_metrics = evaluate_decoder(
        "mapped_code_lora_decoder",
        lora_decoder,
        mapped_code,
        csi,
        args.batch_size,
        args.workers,
        device)

    result = {
        "source_code": args.source_code,
        "target_code": args.target_code,
        "csi_path": args.csi_path,
        "mapper_checkpoint": args.mapper_checkpoint,
        "lora_checkpoint": args.lora_checkpoint,
        "decoder_checkpoint": args.decoder_checkpoint,
        "n": int(n),
        "code_dim": int(source_code.size(1)),
        "decoder_config": dec_cfg,
        "mapper_args": mapper_args,
        "mapper_best": mapper_best,
        "lora_args": vars(lora_args),
        "lora_best": lora_best,
        "lora_missing_keys_count": len(missing),
        "code_metrics": code_metrics(mapped_code, target_code),
        "nmse": {
            raw_metrics["name"]: raw_metrics,
            mapped_base_metrics["name"]: mapped_base_metrics,
            mapped_lora_metrics["name"]: mapped_lora_metrics,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8")


if __name__ == "__main__":
    main()
