"""
从 CSI 整模型 checkpoint 中提取所有 decoder 参数，保存为独立的 .pt 文件。

用法:
    python extract_decoder_params.py <checkpoint_path> [--out_dir OUTPUT_DIR]

默认从 checkpoint 所在实验目录读取 args.json 获取 seed / encoder / decoder 名称，
如果找不到 args.json，通过 --seed / --encoder / --decoder 手动指定。

保存的文件名格式: <seed>_<encoder>_<decoder>.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="从 checkpoint 提取 decoder 参数并保存为独立 .pt 文件"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="checkpoint .pth 文件路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="输出目录 (默认当前目录)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="手动指定 seed (默认从 args.json 读取)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="手动指定 encoder 名称 (默认从 args.json 读取)",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=None,
        help="手动指定 decoder 名称 (默认从 args.json 读取)",
    )
    return parser.parse_args(argv)


def find_args_json(ckpt_path: str) -> str | None:
    """在 checkpoint 的父级实验目录中查找 args.json。"""
    ckpt_dir = Path(ckpt_path).resolve().parent  # e.g. .../checkpoints/
    experiment_dir = ckpt_dir.parent  # e.g. .../transnet_transnet/
    candidates = [
        experiment_dir / "args.json",
        experiment_dir.parent / "args.json",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def load_args_from_json(args_json_path: str) -> dict:
    with open(args_json_path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"[ERROR] checkpoint 不存在: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # ---------- 1. 获取 seed / encoder / decoder 元信息 ----------
    seed = args.seed
    encoder_name = args.encoder
    decoder_name = args.decoder

    if seed is None or encoder_name is None or decoder_name is None:
        args_json = find_args_json(str(ckpt_path))
        if args_json is None:
            print(
                "[ERROR] 找不到 args.json，请通过 --seed / --encoder / --decoder 手动指定",
                file=sys.stderr,
            )
            sys.exit(1)
        exp_args = load_args_from_json(args_json)
        if seed is None:
            seed = exp_args.get("seed")
        if encoder_name is None:
            encoder_name = exp_args.get("encoder")
        if decoder_name is None:
            decoder_name = exp_args.get("decoder")
        print(f"[INFO] 从 {args_json} 读取配置")

    if not all([seed, encoder_name, decoder_name]):
        print(
            "[ERROR] seed / encoder / decoder 未能完整确定，请检查 args.json 或手动提供",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---------- 2. 加载 checkpoint 并提取 decoder 参数 ----------
    print(f"[INFO] 加载 checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    state_dict = checkpoint["state_dict"]
    decoder_state_dict = {}
    prefix = "decoder."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # 去掉 "decoder." 前缀，还原成子模块内部的键名
            new_key = key[len(prefix):]
            decoder_state_dict[new_key] = value

    if not decoder_state_dict:
        print("[ERROR] 未找到任何 decoder. 前缀的键", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 提取 decoder 参数: {len(decoder_state_dict)} 个键")

    # ---------- 3. 保存 ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"seed{seed}_{encoder_name}_{decoder_name}.pt"
    out_path = out_dir / filename

    torch.save(decoder_state_dict, str(out_path))
    print(f"[INFO] 保存到: {out_path.resolve()} ({os.path.getsize(out_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
