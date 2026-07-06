#!/usr/bin/env python
import argparse
import csv
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
    if gpu is not None and not cpu:
        raise RuntimeError(
            f"CUDA is not available after selecting gpu={gpu}; "
            "refusing to run on CPU.")
    return torch.device("cpu")


def load_tensor(path):
    return torch.load(path, weights_only=True, map_location="cpu").float()


def load_csi(path, channel, nt, nc, max_samples=0):
    data = load_tensor(path)
    expected = (channel, nt, nc)
    if data.ndim == 2:
        data = data.view(-1, *expected)
    if data.ndim != 4 or tuple(data.shape[1:]) != expected:
        raise ValueError(
            f"{path} should have shape (N, {channel}, {nt}, {nc}), "
            f"got {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
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


def load_decoder_config(path):
    cfg = json.loads(Path(path).read_text()) if path else {}
    return {
        "decoder": cfg.get("decoder", "transnet"),
        "cr": cfg.get("cr", 4),
        "d_model": cfg.get("d_model", 64),
        "dim_feedforward": cfg.get("dim_feedforward", 2048),
        "channel": cfg.get("channel", 2),
        "nt": cfg.get("nt", 32),
        "nc": cfg.get("nc", 32),
        "hidden": cfg.get("hidden", 16),
        "num_blocks": cfg.get("num_blocks", 2),
    }


def find_code_paths(root):
    root = Path(root)
    paths = sorted(root.glob("**/alignaffine*/**/codewords/mapped_code*.pt"))
    return [
        path for path in paths
        if "codewords" in path.parts and path.is_file()
    ]


def parse_exp_info(path, root):
    path = Path(path)
    rel = path.relative_to(root)
    parts = rel.parts
    codeword_idx = parts.index("codewords")
    exp_dir = root.joinpath(*parts[:codeword_idx])
    return {
        "path": str(path),
        "relative_path": str(rel),
        "exp_dir": str(exp_dir),
        "mapper": parts[1] if len(parts) > 1 else "",
        "align_tag": next((p for p in parts if p.startswith("alignaffine")), ""),
        "source_exp": parts[codeword_idx - 1] if codeword_idx >= 1 else "",
        "code_file": path.name,
    }


@torch.no_grad()
def evaluate_code(decoder, code_path, csi, expected_code_dim, batch_size,
                  workers, device, max_samples=0):
    code = load_tensor(code_path)
    if max_samples and code.size(0) > max_samples:
        code = code[:max_samples].contiguous()
    if code.ndim != 2:
        raise ValueError(f"{code_path} should be 2D, got {tuple(code.shape)}")
    if code.size(1) != expected_code_dim:
        raise ValueError(
            f"{code_path} dim mismatch: expected {expected_code_dim}, "
            f"got {code.size(1)}")
    if code.size(0) != csi.size(0):
        raise ValueError(
            f"{code_path} N mismatch: code={code.size(0)} csi={csi.size(0)}")

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
    nmse_linear = total_error / total_power.clamp_min(1e-12)
    nmse_db = nmse_from_sums(total_error, total_power)
    return {
        "n": int(total_n),
        "mse_loss": total_mse / max(total_n, 1),
        "error_sum": float(total_error.detach().cpu()),
        "power_sum": float(total_power.detach().cpu()),
        "nmse_linear": float(nmse_linear.detach().cpu()),
        "nmse_db": float(nmse_db.detach().cpu()),
    }


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all affine mapper codewords with a fixed decoder.")
    parser.add_argument("--root", default="mapper")
    parser.add_argument("--decoder_checkpoint",
                        default="exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth")
    parser.add_argument("--decoder_args_json",
                        default="exps/COST2100/in/seed42/transnet_transnet/args.json")
    parser.add_argument("--data_path",
                        default="/storage/hujiacong/zxd/datasets/cost2100/in_train.pt")
    parser.add_argument("--output_json",
                        default="mapper/reports/affine_true_nmse/affine_code_nmse.json")
    parser.add_argument("--output_csv",
                        default="mapper/reports/affine_true_nmse/affine_code_nmse.csv")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    device = resolve_device(args.gpu, args.cpu)
    cfg = load_decoder_config(args.decoder_args_json)
    model = universal_csi(
        encoder_name="transnet",
        decoder_name=cfg["decoder"],
        reduction=cfg["cr"],
        d_model=cfg["d_model"],
        channel=cfg["channel"],
        nt=cfg["nt"],
        nc=cfg["nc"],
        dim_feedforward=cfg["dim_feedforward"],
        hidden=cfg["hidden"],
        num_blocks=cfg["num_blocks"])
    load_decoder(model, args.decoder_checkpoint)
    decoder = model.decoder.to(device).eval()

    csi = load_csi(
        args.data_path,
        cfg["channel"],
        cfg["nt"],
        cfg["nc"],
        max_samples=args.max_samples)
    expected_code_dim = cfg["channel"] * cfg["nt"] * cfg["nc"] // cfg["cr"]
    all_code_paths = find_code_paths(args.root)
    code_paths = [
        path for idx, path in enumerate(all_code_paths)
        if idx % args.num_shards == args.shard_id
    ]
    print(
        f"device={device} affine_code_files={len(all_code_paths)} "
        f"shard={args.shard_id}/{args.num_shards} shard_files={len(code_paths)}")

    rows = []
    for idx, code_path in enumerate(code_paths, 1):
        info = parse_exp_info(code_path, Path(args.root))
        try:
            metrics = evaluate_code(
                decoder,
                code_path,
                csi,
                expected_code_dim,
                args.batch_size,
                args.workers,
                device,
                max_samples=args.max_samples)
            row = {**info, **metrics, "error": ""}
            print(
                f"[{idx}/{len(code_paths)}] {info['mapper']} "
                f"{info['source_exp']} {info['code_file']} "
                f"nmse={row['nmse_db']:.3f}dB")
        except Exception as exc:
            row = {**info, "error": str(exc)}
            print(
                f"[{idx}/{len(code_paths)}] ERROR {info['relative_path']}: {exc}")
        rows.append(row)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8")
    write_csv(rows, args.output_csv)
    ok_rows = [row for row in rows if not row.get("error")]
    if ok_rows:
        best = min(ok_rows, key=lambda row: row["nmse_db"])
        print(
            "best: "
            f"{best['nmse_db']:.3f}dB {best['relative_path']}")
    print(f"saved_json={output_json}")
    print(f"saved_csv={args.output_csv}")


if __name__ == "__main__":
    main()
