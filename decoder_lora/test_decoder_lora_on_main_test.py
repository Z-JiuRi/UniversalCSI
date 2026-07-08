#!/usr/bin/env python
import argparse
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DECODER_LORA_DIR = Path(__file__).resolve().parent
ROOT = DECODER_LORA_DIR.parent
if str(DECODER_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(DECODER_LORA_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from train_decoder_lora import (DecoderLoraSystem, build_code_adapter,  # noqa: E402
                                fit_alignment, inject_decoder_lora,
                                load_alignment_code_pair,
                                load_decoder_from_checkpoint,
                                load_lora_state)


def evaluator(pred, gt):
    return (pred - gt).pow(2).sum(), gt.pow(2).sum()


def nmse_from_sums(error_sum, power_sum):
    return 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_main_models_package():
    package_name = f"decoder_lora_main_models_{uuid.uuid4().hex}"
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


def load_csi(path, channel, nt, nc, max_samples=0):
    data = torch.load(path, weights_only=True, map_location="cpu").float()
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(f"{path} got shape {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data.contiguous()


def source_exp_from_name(source_name):
    parts = source_name.split("_")
    if len(parts) < 3 or not parts[0].startswith("seed"):
        raise ValueError(f"Cannot infer source exp from source_name={source_name}")
    seed = parts[0]
    encoder = parts[1]
    decoder = "_".join(parts[2:])
    return ROOT / "exps" / "COST2100" / "in" / seed / f"{encoder}_{decoder}"


def load_source_model(source_name, device):
    exp_dir = source_exp_from_name(source_name)
    args_path = exp_dir / "args.json"
    ckpt_path = exp_dir / "checkpoints" / "best_nmse.pth"
    if not args_path.exists():
        raise FileNotFoundError(args_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    cfg = json.loads(args_path.read_text())
    main_models = load_main_models_package()
    model = main_models.universal_csi(
        encoder_name=cfg.get("encoder", "transnet"),
        decoder_name=cfg.get("decoder", "transnet"),
        reduction=cfg.get("cr", 4),
        d_model=cfg.get("d_model", 64),
        channel=cfg.get("channel", 2),
        nt=cfg.get("nt", 32),
        nc=cfg.get("nc", 32),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        hidden=cfg.get("hidden", 16),
        num_blocks=cfg.get("num_blocks", 2))
    state_dict = clean_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [
        key for key in unexpected
        if not (key.endswith("total_ops") or key.endswith("total_params"))
    ]
    if missing or unexpected:
        raise ValueError(
            f"source checkpoint mismatch {exp_dir}: missing={missing}, "
            f"unexpected={unexpected}")
    model.to(device).eval()
    return model, {
        "source_exp_dir": str(exp_dir),
        "source_checkpoint": str(ckpt_path),
        "source_args_json": str(args_path),
        "encoder": cfg.get("encoder", "transnet"),
        "decoder": cfg.get("decoder", "transnet"),
    }


def build_decoder_lora_system(exp_dir, checkpoint_name, device):
    exp_dir = Path(exp_dir)
    args = argparse.Namespace(**json.loads((exp_dir / "args.json").read_text()))
    for key, value in {
            "lora_alpha": None,
            "fc_lora_rank": None,
            "ffn_lora_rank": None,
            "fc_lora_alpha": None,
            "ffn_lora_alpha": None,
            "lora_dropout": 0.0,
            "code_adapter": "none",
            "code_lowrank_rank": 0,
            "code_mlp_hidden": 0,
            "code_gate_lr": 0.1,
            "code_gate_mlp": 0.1,
            "code_adapter_dropout": 0.0,
            "lambda_delta": 0.0,
            "source_align_code": [],
            "target_align_code": [],
    }.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    ckpt_path = exp_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    base_decoder, decoder_cfg = load_decoder_from_checkpoint(args, device)
    inject_decoder_lora(
        base_decoder,
        target=args.lora_target,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        fc_rank=args.fc_lora_rank,
        ffn_rank=args.ffn_lora_rank,
        fc_alpha=args.fc_lora_alpha,
        ffn_alpha=args.ffn_lora_alpha)
    code_dim = decoder_cfg["channel"] * decoder_cfg["nt"] * decoder_cfg["nc"] // decoder_cfg["cr"]
    code_adapter = build_code_adapter(args, dim=code_dim, device=device)
    system = DecoderLoraSystem(base_decoder, code_adapter).to(device).eval()
    load_compatible_lora_state(system, ckpt_path, device)
    return system, args, decoder_cfg, str(ckpt_path)


def load_compatible_lora_state(system, ckpt_path, device):
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    state_dict = ckpt["state_dict"]
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("code_adapter."):
            converted[key] = value
        elif key.startswith("decoder.fc_decoder.") or key.startswith("decoder.decoder."):
            converted[key] = value
        else:
            converted[f"decoder.{key}"] = value
    state_dict = converted
    missing, unexpected = system.load_state_dict(state_dict, strict=False)
    unexpected = [
        key for key in unexpected
        if "lora_" in key or key.startswith("code_adapter.")
    ]
    if unexpected:
        raise ValueError(f"Unexpected LoRA keys: {unexpected}")
    return ckpt, missing


@torch.no_grad()
def evaluate_exp(exp_dir, data_path, checkpoint_name, device, batch_size,
                 workers, max_samples):
    system, args, decoder_cfg, ckpt_path = build_decoder_lora_system(
        exp_dir,
        checkpoint_name,
        device)
    source_model, source_info = load_source_model(args.source_name, device)
    csi = load_csi(
        data_path,
        decoder_cfg["channel"],
        decoder_cfg["nt"],
        decoder_cfg["nc"],
        max_samples=max_samples)

    source_fit, target_fit = load_alignment_code_pair(
        args.source_code,
        args.target_code,
        source_extra_paths=args.source_align_code,
        target_extra_paths=args.target_align_code,
        max_samples=args.max_samples)
    weight, bias = fit_alignment(
        args.align_mode,
        source_fit,
        target_fit,
        ridge=args.align_ridge)
    weight = weight.to(device)
    bias = bias.to(device)

    loader = DataLoader(
        TensorDataset(csi),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    code_mse_sum = 0.0
    delta_mse_sum = 0.0
    for (gt,) in loader:
        gt = gt.to(device, non_blocking=True)
        source_code = source_model.encode(gt)
        z0 = source_code.matmul(weight) + bias
        z1 = system.adapt_code(z0)
        pred = system.decoder(z1)
        mse = F.mse_loss(pred, gt)
        err, power = evaluator(pred, gt)
        total_error += err
        total_power += power
        n = gt.size(0)
        total_mse += float(mse.detach().cpu()) * n
        total_n += n
        code_mse_sum += float(F.mse_loss(z1, z0).detach().cpu()) * n
        delta_mse_sum += float(F.mse_loss(z1, z0).detach().cpu()) * n

    nmse_db = nmse_from_sums(total_error, total_power)
    return {
        "exp_dir": str(exp_dir),
        "checkpoint": ckpt_path,
        "checkpoint_name": checkpoint_name,
        "source_name": args.source_name,
        **source_info,
        "data_path": str(data_path),
        "n": int(total_n),
        "mse_loss": total_mse / max(total_n, 1),
        "error_sum": float(total_error.detach().cpu()),
        "power_sum": float(total_power.detach().cpu()),
        "nmse_linear": float((total_error / total_power.clamp_min(1e-12)).detach().cpu()),
        "nmse_db": float(nmse_db.detach().cpu()),
        "z1_minus_z0_mse": delta_mse_sum / max(total_n, 1),
        "train_code_adapter": args.code_adapter,
        "code_mlp_hidden": args.code_mlp_hidden,
        "code_lowrank_rank": args.code_lowrank_rank,
        "lambda_code": args.lambda_code,
        "lambda_delta": getattr(args, "lambda_delta", None),
        "lora_target": args.lora_target,
        "fc_lora_rank": args.fc_lora_rank,
        "ffn_lora_rank": args.ffn_lora_rank,
        "lr": args.lr,
        "eta_min": args.eta_min,
        "epochs": args.epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="decoder_lora/exps")
    parser.add_argument("--exp_dir", default=None)
    parser.add_argument("--data_path", default="/storage/hujiacong/zxd/datasets/cost2100/in_test.pt")
    parser.add_argument("--checkpoint_name", default="best_nmse.pth")
    parser.add_argument("--output_json", default="decoder_lora/reports/test_nmse_on_main_test.json")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.gpu, args.cpu)
    if args.exp_dir:
        exp_dirs = [Path(args.exp_dir)]
    else:
        exp_dirs = sorted(
            path.parent.parent
            for path in Path(args.root).glob(f"*/*/checkpoints/{args.checkpoint_name}"))
    if not exp_dirs:
        raise FileNotFoundError(f"No {args.checkpoint_name} under {args.root}")

    results = []
    for i, exp_dir in enumerate(exp_dirs, 1):
        print(f"[{i}/{len(exp_dirs)}] testing {exp_dir}", flush=True)
        try:
            result = evaluate_exp(
                exp_dir,
                args.data_path,
                args.checkpoint_name,
                device,
                args.batch_size,
                args.workers,
                args.max_samples)
            results.append(result)
            print(
                f"  nmse={result['nmse_db']:.3f}dB "
                f"source={result['source_name']}",
                flush=True)
        except Exception as exc:
            results.append({
                "exp_dir": str(exp_dir),
                "error": repr(exc),
            })
            print(f"  ERROR: {exc}", flush=True)

    results_sorted = sorted(
        results,
        key=lambda item: item.get("nmse_db", 999.0))
    output = {
        "data_path": args.data_path,
        "checkpoint_name": args.checkpoint_name,
        "num_exps": len(exp_dirs),
        "results": results_sorted,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("\nSorted by main test NMSE:")
    for item in results_sorted:
        if "error" in item:
            print(f"ERR {item['exp_dir']}: {item['error']}")
        else:
            print(
                f"{item['nmse_db']:.3f} dB | {item['source_name']} | "
                f"adapter={item['train_code_adapter']} h={item['code_mlp_hidden']} "
                f"lc={item['lambda_code']} ld={item['lambda_delta']} | "
                f"{item['exp_dir']}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
