#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DECODER_LORA_DIR = ROOT / "decoder_lora"
if str(DECODER_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(DECODER_LORA_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from decoder_lora.train_decoder_lora import (  # noqa: E402
    encode_csi,
    load_csi_tensor,
    load_full_model_from_exp,
)


def is_pure_transnet_decoder_exp(exp_dir, strict_pure=True):
    args_path = exp_dir / "args.json"
    ckpt_path = exp_dir / "checkpoints" / "best_nmse.pth"
    if not args_path.exists() or not ckpt_path.exists():
        return False, None
    cfg = json.loads(args_path.read_text(encoding="utf-8"))
    if cfg.get("decoder") != "transnet":
        return False, cfg
    if strict_pure:
        if cfg.get("adapter") not in (None, "none"):
            return False, cfg
        if cfg.get("canonical_head", "none") != "none":
            return False, cfg
        if cfg.get("pretrained") or cfg.get("pretrained_decoder") or cfg.get("pretrained_encoder"):
            return False, cfg
    return True, cfg


def iter_experiments(root, strict_pure=True):
    for args_path in sorted(Path(root).glob("**/args.json")):
        exp_dir = args_path.parent
        ok, cfg = is_pure_transnet_decoder_exp(exp_dir, strict_pure=strict_pure)
        if ok:
            yield exp_dir, cfg


def resolve_device(gpu=None, cpu=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def export_split(model, cfg, split, exp_dir, args, device):
    path_key = f"{split}_path"
    override_path = args.val_path if split == "val" else args.test_path
    csi_path = override_path or cfg.get(path_key)
    if not csi_path:
        raise ValueError(f"{exp_dir} missing {path_key} in args.json")
    output_path = exp_dir / "codewords" / f"{split}_code.pt"
    if args.dry_run:
        print(f"would export {split}: {csi_path} -> {output_path}")
        return
    if output_path.exists() and not args.overwrite:
        print(f"skip existing {output_path}")
        return
    csi = load_csi_tensor(
        csi_path,
        cfg.get("channel", 2),
        cfg.get("nt", 32),
        cfg.get("nc", 32),
        max_samples=args.max_samples)
    codes = encode_csi(
        model,
        csi,
        device,
        batch_size=args.batch_size,
        workers=args.workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codes, output_path)
    print(f"saved {output_path} shape={tuple(codes.shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export val/test codewords for base pure transnet-decoder experiments.")
    parser.add_argument("--root", default="exps/COST2100/in/base")
    parser.add_argument("--splits", default="val,test",
                        help="comma-separated splits to export: val,test")
    parser.add_argument("--val_path", default="",
                        help="override validation CSI path for every experiment")
    parser.add_argument("--test_path", default="",
                        help="override test CSI path for every experiment")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--include_non_pure", action="store_true",
                        help="only require decoder=transnet; do not filter adapters/pretrained flags")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.gpu, args.cpu)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise ValueError("splits must not be empty")

    count = 0
    for exp_dir, cfg in iter_experiments(
            args.root,
            strict_pure=not args.include_non_pure):
        count += 1
        if args.limit and count > args.limit:
            break
        print(f"[{count}] {exp_dir}")
        if args.dry_run:
            for split in splits:
                export_split(None, cfg, split, exp_dir, args, device)
            continue
        model, _ = load_full_model_from_exp(
            exp_dir,
            exp_dir / "checkpoints" / "best_nmse.pth",
            device)
        for split in splits:
            export_split(model, cfg, split, exp_dir, args, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"done experiments={min(count, args.limit) if args.limit else count}")


if __name__ == "__main__":
    main()
