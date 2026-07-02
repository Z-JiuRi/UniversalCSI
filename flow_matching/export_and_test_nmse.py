#!/usr/bin/env python
import argparse
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

FLOW_DIR = Path(__file__).resolve().parent
ROOT = FLOW_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.statics import evaluator, nmse_from_sums  # noqa: E402


def load_flow_models_module():
    spec = importlib.util.spec_from_file_location(
        "flow_matching_local_models",
        FLOW_DIR / "models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tensor(path):
    return torch.load(path, weights_only=True, map_location="cpu").float()


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


def build_flow_model(cfg, dim):
    fm_models = load_flow_models_module()
    return fm_models.FlowMatchingTranslator(
        dim,
        hidden_dim=cfg["hidden_dim"],
        num_blocks=cfg["num_blocks"],
        time_dim=cfg["time_dim"],
        condition=cfg["condition"],
        dropout=cfg["dropout"])


def load_decoder(args, device):
    cfg = json.loads(Path(args.decoder_args_json).read_text())
    decoder_name = args.decoder or cfg.get("decoder", "transnet")
    cr = args.cr or cfg.get("cr", 4)
    d_model = args.d_model or cfg.get("d_model", 64)
    dim_feedforward = (
        args.dim_feedforward or cfg.get("dim_feedforward", 2048))
    channel = args.channel or cfg.get("channel", 2)
    nt = args.nt or cfg.get("nt", 32)
    nc = args.nc or cfg.get("nc", 32)
    hidden = args.hidden or cfg.get("hidden", 16)
    num_blocks = args.num_blocks or cfg.get("num_blocks", 2)

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
    return model.decoder.to(device).eval(), {
        "decoder": decoder_name,
        "cr": cr,
        "channel": channel,
        "nt": nt,
        "nc": nc,
    }


def load_csi(path, channel, nt, nc, max_samples=0):
    data = load_tensor(path)
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    expected = (channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != expected:
        raise ValueError(f"{path} got shape {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data


def code_metrics(pred, target):
    mse = F.mse_loss(pred, target)
    power = target.pow(2).sum().clamp_min(1e-12)
    err = (pred - target).pow(2).sum()
    return {
        "code_mse": float(mse),
        "code_rmse": float(mse.sqrt()),
        "code_cos": float(F.cosine_similarity(pred, target, dim=1).mean()),
        "code_nmse": float(10.0 * torch.log10(err / power)),
    }


@torch.no_grad()
def export_code(exp_dir, checkpoint_path, device, batch_size, workers,
                force=False):
    exp_dir = Path(exp_dir)
    cfg = json.loads((exp_dir / "args.json").read_text())
    output_path = exp_dir / "codewords" / "mapped_code.pt"
    root_output_path = exp_dir / "mapped_code.pt"
    if output_path.exists() and not force:
        mapped = load_tensor(output_path)
        target = load_tensor(cfg["target_code"])
        max_samples = cfg.get("max_samples", 0)
        if max_samples and target.size(0) > max_samples:
            target = target[:max_samples].contiguous()
        return mapped, target, {"export_skipped": True}

    source = load_tensor(cfg["source_code"])
    target = load_tensor(cfg["target_code"])
    max_samples = cfg.get("max_samples", 0)
    if max_samples and source.size(0) > max_samples:
        source = source[:max_samples].contiguous()
        target = target[:max_samples].contiguous()

    ckpt = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=torch.device("cpu"))
    model = build_flow_model(cfg, source.size(1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    loader = DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda")
    outputs = []
    total_batches = len(loader)
    start_time = time.time()
    for i, batch in enumerate(loader, 1):
        batch = batch.to(device, non_blocking=True)
        outputs.append(model.sample(
            batch,
            steps=cfg["ode_steps"],
            method=cfg["ode_method"]).cpu())
        if i == 1 or i % 5 == 0 or i == total_batches:
            elapsed = time.time() - start_time
            print(
                f"[export] {exp_dir.name} batch {i}/{total_batches} "
                f"elapsed={elapsed:.1f}s",
                flush=True)
    mapped = torch.cat(outputs, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mapped, output_path)
    torch.save(mapped, root_output_path)
    return mapped, target, {
        "export_skipped": False,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best": ckpt.get("best"),
    }


@torch.no_grad()
def decoder_nmse(decoder, code, csi, device, batch_size, workers):
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
    for batch_code, batch_gt in loader:
        batch_code = batch_code.to(device, non_blocking=True)
        batch_gt = batch_gt.to(device, non_blocking=True)
        pred = decoder(batch_code)
        mse = F.mse_loss(pred, batch_gt)
        error_sum, power_sum = evaluator(pred, batch_gt)
        total_error += error_sum
        total_power += power_sum
        total_mse += float(mse.detach().cpu()) * batch_code.size(0)
        total_n += batch_code.size(0)
    return {
        "decoder_mse_loss": total_mse / max(total_n, 1),
        "decoder_nmse_linear": float(
            (total_error / total_power.clamp_min(1e-12)).detach().cpu()),
        "decoder_nmse_db": float(
            nmse_from_sums(total_error, total_power).detach().cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="flow_matching/exps")
    parser.add_argument("--pattern", default="checkpoints/best_mse.pth")
    parser.add_argument("--output_json",
                        default="flow_matching/reports/code_only_nmse.json")
    parser.add_argument("--decoder_checkpoint",
                        default="exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth")
    parser.add_argument("--decoder_args_json",
                        default="exps/COST2100/in/seed42/transnet_transnet/args.json")
    parser.add_argument("--data_path",
                        default="/storage/hujiacong/zxd/datasets/cost2100/in_train.pt")
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--cr", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--nc", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--decoder_batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_exps", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.gpu, args.cpu)
    checkpoints = sorted(Path(args.root).rglob(args.pattern))
    if args.max_exps:
        checkpoints = checkpoints[:args.max_exps]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {args.root}")

    decoder, decoder_cfg = load_decoder(args, device)
    csi = load_csi(
        args.data_path,
        decoder_cfg["channel"],
        decoder_cfg["nt"],
        decoder_cfg["nc"])

    results = []
    print(f"device={device} checkpoints={len(checkpoints)}", flush=True)
    for checkpoint_path in checkpoints:
        exp_dir = checkpoint_path.parents[1]
        print(f"\n==> {exp_dir}", flush=True)
        mapped, target, extra = export_code(
            exp_dir,
            checkpoint_path,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
            force=args.force)
        if csi.size(0) != mapped.size(0):
            csi_eval = csi[:mapped.size(0)].contiguous()
        else:
            csi_eval = csi
        row = {
            "exp_dir": str(exp_dir),
            "checkpoint": str(checkpoint_path),
            **extra,
            **code_metrics(mapped, target),
            **decoder_nmse(
                decoder,
                mapped,
                csi_eval,
                device,
                batch_size=args.decoder_batch_size,
                workers=args.workers),
        }
        results.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nsaved={output_path}", flush=True)


if __name__ == "__main__":
    main()
