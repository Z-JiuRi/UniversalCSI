#!/usr/bin/env python
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DECODER_LORA_DIR = ROOT / "decoder_lora"
if str(DECODER_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(DECODER_LORA_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train_decoder_lora import (  # noqa: E402
    DecoderLoraSystem,
    AlignedCodeCsiDataset,
    build_code_adapter,
    evaluate_decoder,
    exp_dir_from_code_path,
    exp_dir_from_checkpoint_path,
    fit_alignment,
    load_alignment_code_pair,
    inject_decoder_lora,
    load_decoder_from_checkpoint,
    load_full_model_from_exp,
    load_lora_state,
)


def load_csi(path, channel, nt, nc, max_samples=0):
    data = torch.load(path, weights_only=True, map_location="cpu").float()
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(f"{path} got shape {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data.contiguous()


def build_system(exp_args, ckpt_path, device):
    base_decoder, decoder_cfg = load_decoder_from_checkpoint(exp_args, device)
    inject_decoder_lora(
        base_decoder,
        target=exp_args.lora_target,
        rank=exp_args.lora_rank,
        alpha=exp_args.lora_alpha,
        dropout=exp_args.lora_dropout,
        fc_rank=exp_args.fc_lora_rank,
        ffn_rank=exp_args.ffn_lora_rank,
        fc_alpha=exp_args.fc_lora_alpha,
        ffn_alpha=exp_args.ffn_lora_alpha,
    )
    code_dim = (
        decoder_cfg["channel"] * decoder_cfg["nt"] * decoder_cfg["nc"]
        // decoder_cfg["cr"]
    )
    code_adapter = build_code_adapter(exp_args, code_dim, device)
    system = DecoderLoraSystem(base_decoder, code_adapter).to(device).eval()
    load_lora_state(system, ckpt_path, device)
    return system, decoder_cfg


def canonicalize_lora_module(module):
    down = module.lora_down.weight.detach()
    up = module.lora_up.weight.detach()
    scaling = float(module.scaling)
    rank = int(module.rank)

    delta = up.matmul(down) * scaling
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    u = u[:, :rank].contiguous()
    s = s[:rank].contiguous()
    vh = vh[:rank, :].contiguous()

    for idx in range(rank):
        pivot = torch.argmax(torch.abs(u[:, idx]))
        if u[pivot, idx] < 0:
            u[:, idx].mul_(-1)
            vh[idx, :].mul_(-1)

    sqrt_s = torch.sqrt(s.clamp_min(0.0))
    up_eff = u * sqrt_s.unsqueeze(0)
    down_eff = sqrt_s.unsqueeze(1) * vh

    # Keep the existing LoRALinear scaling in forward(), so store factors for
    # delta / scaling. This preserves the effective merged DeltaW.
    module.lora_up.weight.data.copy_((up_eff / scaling).to(module.lora_up.weight))
    module.lora_down.weight.data.copy_(down_eff.to(module.lora_down.weight))

    recon = module.lora_up.weight.detach().float().matmul(
        module.lora_down.weight.detach().float()) * scaling
    rel_err = (recon - delta.float()).norm() / delta.float().norm().clamp_min(1e-12)
    return float(rel_err.cpu())


def canonicalize_all_lora(system):
    errors = {}
    for name, module in system.named_modules():
        if hasattr(module, "lora_down") and hasattr(module, "lora_up"):
            errors[name] = canonicalize_lora_module(module)
    return errors


@torch.no_grad()
def evaluate_external(system, source_model, weight, bias, csi_path, decoder_cfg,
                      batch_size, workers, device, max_samples=0):
    csi = load_csi(
        csi_path,
        decoder_cfg["channel"],
        decoder_cfg["nt"],
        decoder_cfg["nc"],
        max_samples=max_samples,
    )
    loader = DataLoader(
        TensorDataset(csi),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    for (gt,) in loader:
        gt = gt.to(device, non_blocking=True)
        source_code = source_model.encode(gt)
        z0 = source_code.matmul(weight) + bias
        pred = system(z0)
        total_error += (pred - gt).pow(2).sum()
        total_power += gt.pow(2).sum()
        total_mse += float(F.mse_loss(pred, gt).detach().cpu()) * gt.size(0)
        total_n += gt.size(0)
    nmse = 10.0 * torch.log10(total_error / total_power.clamp_min(1e-12))
    return {
        "decoder_mse": total_mse / max(total_n, 1),
        "decoder_nmse": float(nmse.detach().cpu()),
        "n": total_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--checkpoint_name", default="best_nmse.pth")
    parser.add_argument("--val_csi_path",
                        default="/storage/hujiacong/zxd/datasets/cost2100/in_val.pt")
    parser.add_argument("--test_csi_path",
                        default="/storage/hujiacong/zxd/datasets/cost2100/in_test.pt")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    exp_dir = Path(args.exp_dir)
    exp_args = argparse.Namespace(**json.loads((exp_dir / "args.json").read_text()))
    ckpt_path = exp_dir / "checkpoints" / args.checkpoint_name

    source_fit, target_fit = load_alignment_code_pair(
        exp_args.source_code,
        exp_args.target_code,
        source_extra_paths=getattr(exp_args, "source_align_code", []),
        target_extra_paths=getattr(exp_args, "target_align_code", []),
        max_samples=exp_args.max_samples,
    )
    weight, bias = fit_alignment(
        exp_args.align_mode,
        source_fit,
        target_fit,
        ridge=exp_args.align_ridge,
    )
    weight = weight.to(device)
    bias = bias.to(device)

    source_exp_dir = (
        Path(exp_args.source_args_json).parent
        if hasattr(exp_args, "source_args_json") and exp_args.source_args_json
        else exp_dir_from_code_path(exp_args.source_code)
    )
    source_checkpoint = (
        exp_args.source_checkpoint
        if hasattr(exp_args, "source_checkpoint") and exp_args.source_checkpoint
        else str(source_exp_dir / "checkpoints" / "best_nmse.pth")
    )
    source_model, _ = load_full_model_from_exp(
        source_exp_dir, source_checkpoint, device)

    original, decoder_cfg = build_system(exp_args, ckpt_path, device)
    svd_system = copy.deepcopy(original).eval()
    svd_errors = canonicalize_all_lora(svd_system)

    train_set = AlignedCodeCsiDataset(
        exp_args.source_code,
        exp_args.target_code,
        exp_args.csi_path,
        weight.detach().cpu(),
        bias.detach().cpu(),
        channel=decoder_cfg["channel"],
        nt=decoder_cfg["nt"],
        nc=decoder_cfg["nc"],
        split="all",
        val_ratio=0.0,
        max_samples=args.max_samples or exp_args.max_samples,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    results = {
        "exp_dir": str(exp_dir),
        "checkpoint": str(ckpt_path),
        "svd_relative_delta_errors": svd_errors,
        "original": {
            "train": evaluate_decoder(original, train_loader, device),
            "val": evaluate_external(
                original, source_model, weight, bias, args.val_csi_path,
                decoder_cfg, args.batch_size, args.workers, device,
                max_samples=args.max_samples),
            "test": evaluate_external(
                original, source_model, weight, bias, args.test_csi_path,
                decoder_cfg, args.batch_size, args.workers, device,
                max_samples=args.max_samples),
        },
        "svd_canonical": {
            "train": evaluate_decoder(svd_system, train_loader, device),
            "val": evaluate_external(
                svd_system, source_model, weight, bias, args.val_csi_path,
                decoder_cfg, args.batch_size, args.workers, device,
                max_samples=args.max_samples),
            "test": evaluate_external(
                svd_system, source_model, weight, bias, args.test_csi_path,
                decoder_cfg, args.batch_size, args.workers, device,
                max_samples=args.max_samples),
        },
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
