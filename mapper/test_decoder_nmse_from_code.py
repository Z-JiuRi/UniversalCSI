#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import universal_csi  # noqa: E402
from utils.statics import evaluator, nmse_from_sums  # noqa: E402


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tensor(path):
    return torch.load(path, weights_only=True, map_location="cpu").to(torch.float32)


def load_csi(path, channel, nt, nc):
    data = load_tensor(path)
    expected = (channel, nt, nc)
    if data.ndim == 2:
        data = data.view(-1, *expected)
    if data.ndim != 4 or tuple(data.shape[1:]) != expected:
        raise ValueError(
            f"{path} should have shape (N, {channel}, {nt}, {nc}), "
            f"got {tuple(data.shape)}")
    return data


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


def load_decoder(model, checkpoint_path):
    state_dict = clean_state_dict(checkpoint_path)
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


def config_from_json(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="Decode saved codewords with a fixed decoder and compute NMSE.")
    parser.add_argument("--code_path", required=True,
                        help="mapped_code.pt or train_code.pt to decode")
    parser.add_argument("--decoder_checkpoint", required=True,
                        help="checkpoint containing decoder.* weights")
    parser.add_argument("--data_path", required=True,
                        help="CSI tensor matching codeword order")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--decoder_args_json", default=None,
                        help="args.json from the decoder training exp")
    parser.add_argument("--encoder", default="transnet",
                        help="dummy encoder name used only to build UniversalCSI")
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--cr", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--nc", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = config_from_json(args.decoder_args_json)
    decoder = args.decoder or cfg.get("decoder", "transnet")
    cr = args.cr or cfg.get("cr", 4)
    d_model = args.d_model or cfg.get("d_model", 64)
    dim_feedforward = args.dim_feedforward or cfg.get("dim_feedforward", 2048)
    channel = args.channel or cfg.get("channel", 2)
    nt = args.nt or cfg.get("nt", 32)
    nc = args.nc or cfg.get("nc", 32)
    hidden = args.hidden or cfg.get("hidden", 16)
    num_blocks = args.num_blocks or cfg.get("num_blocks", 2)

    device = resolve_device(args.gpu, args.cpu)
    code = load_tensor(args.code_path)
    if code.ndim != 2:
        raise ValueError(f"code_path should be 2D, got {tuple(code.shape)}")
    csi = load_csi(args.data_path, channel, nt, nc)
    if args.max_samples and args.max_samples > 0:
        code = code[:args.max_samples]
        csi = csi[:args.max_samples]
    if code.size(0) != csi.size(0):
        raise ValueError(
            f"code/data N mismatch: {code.size(0)} vs {csi.size(0)}")
    expected_code_dim = channel * nt * nc // cr
    if code.size(1) != expected_code_dim:
        raise ValueError(
            f"code dim mismatch: expected {expected_code_dim}, "
            f"got {code.size(1)}")

    model = universal_csi(
        encoder_name=args.encoder,
        decoder_name=decoder,
        reduction=cr,
        d_model=d_model,
        channel=channel,
        nt=nt,
        nc=nc,
        dim_feedforward=dim_feedforward,
        hidden=hidden,
        num_blocks=num_blocks)
    load_decoder(model, args.decoder_checkpoint)
    decoder_model = model.decoder.to(device).eval()

    loader = DataLoader(
        TensorDataset(code, csi),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")

    total_error = torch.tensor(0.0, device=device)
    total_power = torch.tensor(0.0, device=device)
    total_mse = 0.0
    total_n = 0
    with torch.no_grad():
        for batch_code, batch_gt in loader:
            batch_code = batch_code.to(device, non_blocking=True)
            batch_gt = batch_gt.to(device, non_blocking=True)
            pred = decoder_model(batch_code)
            mse = F.mse_loss(pred, batch_gt)
            error_sum, power_sum = evaluator(pred, batch_gt)
            total_error += error_sum
            total_power += power_sum
            total_mse += float(mse.detach().cpu()) * batch_code.size(0)
            total_n += batch_code.size(0)

    nmse_linear = total_error / total_power.clamp_min(1e-12)
    nmse_db = nmse_from_sums(total_error, total_power)
    result = {
        "code_path": str(args.code_path),
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "data_path": str(args.data_path),
        "n": int(total_n),
        "code_dim": int(code.size(1)),
        "decoder": decoder,
        "cr": int(cr),
        "mse_loss": total_mse / max(total_n, 1),
        "error_sum": float(total_error.detach().cpu()),
        "power_sum": float(total_power.detach().cpu()),
        "nmse_linear": float(nmse_linear.detach().cpu()),
        "nmse_db": float(nmse_db.detach().cpu()),
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
