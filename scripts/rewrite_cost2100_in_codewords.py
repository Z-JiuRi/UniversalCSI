#!/usr/bin/env python3
'''
脚本功能：
    批量重写 `exps/COST2100/in` 下所有实验的 `codewords/train_code.pt`。
    旧的 `train_code.pt` 可能是在 `train_loader shuffle=True` 时直接拼接保存的，
    因而文件第 i 行不一定对应训练集第 i 个样本。本脚本会重新读取每个实验的
    `args.json` 和 `checkpoints/best_nmse.pth`，只加载并使用 `encoder.*` 权重，
    对原始 train set 做前向编码，然后按样本 index 回填保存，保证：

        train_code.pt[i] == train_dataset[i] 经过该实验 encoder 后得到的原始码字

    注意：
    - 无论实验有没有 LoRA，本脚本都不会使用 LoRA 权重，因为 LoRA 作用在 decoder。
    - 无论实验有没有 code_adapter，本脚本都不会使用 code_adapter 权重；保存的是
      encoder 后、adapter 前的原始压缩码字。
    - decoder 权重也不会参与导出。
    - 默认是 dry-run，只检查和导出到内存，不覆盖文件；必须传 `--apply` 才会替换
      原来的 `codewords/train_code.pt`。

参数说明：
    --root:
        要扫描的实验根目录，默认是 `exps/COST2100/in`。脚本会递归查找其中所有
        包含 `args.json` 的实验目录。
    --checkpoint:
        每个实验 `checkpoints/` 目录下要加载的 checkpoint 文件名，默认
        `best_nmse.pth`。
    --batch_size:
        导出码字时使用的 batch size。默认读取各实验 `args.json` 里的
        `batch_size`；传入该参数可统一覆盖。
    --workers:
        导出 DataLoader 的 worker 数，默认 0。为了稳定排查，建议先用 0。
    --device:
        导出使用的设备，例如 `cpu`、`cuda` 或 `cuda:0`。默认 `cpu`。
    --shuffle:
        导出时故意让 DataLoader shuffle，用于压力测试 index 回填逻辑。即使打开
        该选项，最终保存顺序仍应回到原始 dataset 顺序。
    --apply:
        真正覆盖 `codewords/train_code.pt`。不传该参数时只是 dry-run。
    --backup:
        覆盖前额外保存 `train_code.pt.bak`。每个 COST2100 train code 大约 200MB，
        全量备份会额外占用大量磁盘，因此默认不备份。
    --limit:
        只处理前 N 个匹配实验，便于小范围测试。
    --only:
        只处理实验目录路径中包含该字符串的实验，便于筛选某一类实验。
'''

import argparse
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import universal_csi


DEFAULT_ROOT = "exps/COST2100/in"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate COST2100/in train_code.pt files in dataset order. "
            "The script reads each experiment args.json, loads its checkpoint, "
            "runs the raw encoder on train data, and writes "
            "codewords/train_code.pt aligned by sample index."
        )
    )
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="experiment root to scan")
    parser.add_argument("--checkpoint", default="best_nmse.pth",
                        help="checkpoint filename under each checkpoints/ dir")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="override args.json batch_size for export")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers for export")
    parser.add_argument("--device", default="cpu",
                        help="cpu, cuda, or cuda:N")
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle export loader to stress-test index realignment")
    parser.add_argument("--apply", action="store_true",
                        help="overwrite train_code.pt; without this, only dry-run")
    parser.add_argument("--backup", action="store_true",
                        help="keep train_code.pt.bak before overwriting")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most this many experiments")
    parser.add_argument("--only", default=None,
                        help="substring filter for experiment directory")
    return parser.parse_args()


def load_args(path):
    with open(path, "r") as f:
        data = json.load(f)
    return SimpleNamespace(**data), data


def find_experiments(root, checkpoint_name, only=None):
    experiments = []
    for dirpath, _, filenames in os.walk(root):
        if "args.json" not in filenames:
            continue
        if only and only not in dirpath:
            continue
        checkpoint = os.path.join(dirpath, "checkpoints", checkpoint_name)
        code_path = os.path.join(dirpath, "codewords", "train_code.pt")
        experiments.append((dirpath, checkpoint, code_path))
    return sorted(experiments)


def load_train_tensor(path, channel, nt, nc):
    data = torch.load(path, weights_only=True, map_location="cpu")
    data = data.to(torch.float32)
    expected_shape = (channel, nt, nc)
    if data.ndim == 2:
        data = data.view(-1, *expected_shape)
    if data.ndim != 4 or tuple(data.shape[1:]) != expected_shape:
        raise ValueError(
            f"{path} should have shape (N, {channel}, {nt}, {nc}), "
            f"got {tuple(data.shape)}"
        )
    return data


def clean_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True,
                            map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.endswith("total_ops") or key.endswith("total_params"):
            continue
        cleaned[key] = value
    return cleaned


def build_model(exp_args):
    return universal_csi(
        encoder_name=exp_args.encoder,
        decoder_name=exp_args.decoder,
        reduction=exp_args.cr,
        channel=getattr(exp_args, "channel", 2),
        nt=exp_args.nt,
        nc=exp_args.nc,
        d_model=getattr(exp_args, "d_model", 64),
        dim_feedforward=getattr(exp_args, "dim_feedforward", 2048),
        code_adapter=getattr(exp_args, "code_adapter", False),
        hidden=getattr(exp_args, "hidden", 16),
        num_blocks=getattr(exp_args, "num_blocks", 2),
    )


def load_encoder_weights(model, state_dict, checkpoint_path):
    model_state = model.state_dict()
    loadable = {}
    for key, value in state_dict.items():
        if not key.startswith("encoder."):
            continue
        if key not in model_state:
            raise ValueError(
                f"{checkpoint_path} has encoder tensor {key}, but current "
                "model structure does not")
        if tuple(value.shape) != tuple(model_state[key].shape):
            raise ValueError(
                f"{checkpoint_path} encoder tensor {key} shape "
                f"{tuple(value.shape)} does not match current model "
                f"{tuple(model_state[key].shape)}")
        loadable[key] = value

    encoder_keys = [key for key in model_state if key.startswith("encoder.")]
    missing_encoder = [key for key in encoder_keys if key not in loadable]
    if missing_encoder:
        raise ValueError(
            f"{checkpoint_path} cannot fully restore encoder; "
            f"missing {len(missing_encoder)} tensors, first={missing_encoder[0]}"
        )

    extra_encoder = [key for key in loadable if key not in encoder_keys]
    if extra_encoder:
        raise ValueError(
            f"{checkpoint_path} has unexpected encoder tensors, "
            f"first={extra_encoder[0]}")

    model_state.update(loadable)
    model.load_state_dict(model_state, strict=True)
    return len(loadable)


def export_codes(model, train_tensor, batch_size, workers, device, shuffle):
    indices = torch.arange(train_tensor.size(0), dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_tensor, indices),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(0) if shuffle else None,
    )

    model.to(device)
    model.eval()
    aligned_codes = None
    seen = torch.zeros(train_tensor.size(0), dtype=torch.bool)
    with torch.no_grad():
        for sparse_gt, batch_indices in loader:
            sparse_gt = sparse_gt.to(device, non_blocking=True)
            code = model.encoder(sparse_gt).detach().cpu()
            if aligned_codes is None:
                aligned_codes = torch.empty(
                    train_tensor.size(0), code.size(1), dtype=code.dtype)
            aligned_codes[batch_indices] = code
            seen[batch_indices] = True

    if aligned_codes is None:
        raise ValueError("cannot export empty train set")
    if not bool(seen.all()):
        missing = torch.nonzero(~seen, as_tuple=False).view(-1)
        raise ValueError(f"missing exported codes for indices {missing[:10].tolist()}")
    return aligned_codes


def atomic_save(codes, code_path, apply, backup):
    if not apply:
        return None
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    backup_path = None
    if backup and os.path.exists(code_path):
        backup_path = code_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(code_path, backup_path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".train_code.", suffix=".pt",
        dir=os.path.dirname(code_path))
    os.close(fd)
    try:
        torch.save(codes, tmp_path)
        os.replace(tmp_path, code_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return backup_path


def format_optional(value):
    if value is None:
        return "None"
    return str(value)


def main():
    args = parse_args()
    device = torch.device(args.device)
    experiments = find_experiments(args.root, args.checkpoint, args.only)
    if args.limit is not None:
        experiments = experiments[:args.limit]

    print("=" * 80)
    print("COST2100/in codeword rewrite")
    print(f"mode              : {'apply' if args.apply else 'dry-run'}")
    print(f"root              : {args.root}")
    print(f"checkpoint         : {args.checkpoint}")
    print(f"experiments        : {len(experiments)}")
    print(f"device            : {device}")
    print(f"batch_size override: {format_optional(args.batch_size)}")
    print(f"workers           : {args.workers}")
    print(f"shuffle export    : {args.shuffle}")
    print(f"backup on apply   : {args.backup}")
    print("code definition   : raw encoder output, before code_adapter/decoder/LoRA")
    print("=" * 80)
    processed = 0
    skipped = 0
    failed = 0

    for idx, (exp_dir, checkpoint_path, code_path) in enumerate(experiments, 1):
        rel = os.path.relpath(exp_dir, args.root)
        try:
            print("-" * 80)
            print(f"[{idx}/{len(experiments)}] experiment: {rel}")
            print(f"  args.json       : {os.path.join(exp_dir, 'args.json')}")
            print(f"  checkpoint      : {checkpoint_path}")
            print(f"  output code     : {code_path}")
            if not os.path.isfile(checkpoint_path):
                skipped += 1
                print("  status          : SKIP")
                print("  reason          : missing checkpoint")
                continue
            if not os.path.isfile(code_path):
                skipped += 1
                print("  status          : SKIP")
                print("  reason          : missing existing train_code.pt")
                continue

            exp_args, _ = load_args(os.path.join(exp_dir, "args.json"))
            batch_size = args.batch_size or exp_args.batch_size
            print("  model arch      : "
                  f"encoder={exp_args.encoder}, decoder={exp_args.decoder}, "
                  f"code_adapter={getattr(exp_args, 'code_adapter', False)}, "
                  f"lora_component={format_optional(getattr(exp_args, 'lora_component', None))}")
            print("  model dims      : "
                  f"channel={getattr(exp_args, 'channel', 2)}, "
                  f"nt={exp_args.nt}, nc={exp_args.nc}, cr={exp_args.cr}, "
                  f"d_model={getattr(exp_args, 'd_model', 64)}, "
                  f"dim_feedforward={getattr(exp_args, 'dim_feedforward', 2048)}")
            print(f"  train path      : {exp_args.train_path}")
            print(f"  export batch    : {batch_size}")
            print("  pretrained      : "
                  f"full={format_optional(getattr(exp_args, 'pretrained', None))}, "
                  f"encoder={format_optional(getattr(exp_args, 'pretrained_encoder', None))}, "
                  f"decoder={format_optional(getattr(exp_args, 'pretrained_decoder', None))}")
            print("  teacher code    : "
                  f"{format_optional(getattr(exp_args, 'teacher_code', None))}")

            train_tensor = load_train_tensor(
                exp_args.train_path,
                getattr(exp_args, "channel", 2),
                exp_args.nt,
                exp_args.nc,
            )
            print(f"  train tensor    : shape={tuple(train_tensor.shape)}")
            model = build_model(exp_args)
            print(f"  model class     : {model.__class__.__name__}")
            state_dict = clean_state_dict(checkpoint_path)
            checkpoint_encoder_keys = [
                key for key in state_dict if key.startswith("encoder.")]
            print("  checkpoint keys : "
                  f"total={len(state_dict)}, encoder={len(checkpoint_encoder_keys)}")
            loaded = load_encoder_weights(model, state_dict, checkpoint_path)
            print(f"  loaded weights  : encoder tensors={loaded} (strict name/shape match)")
            codes = export_codes(
                model, train_tensor, batch_size, args.workers,
                device, args.shuffle)
            old_codes = torch.load(code_path, weights_only=True,
                                   map_location="cpu")
            print(f"  old code shape  : {tuple(old_codes.shape)}")
            print(f"  new code shape  : {tuple(codes.shape)}")
            if tuple(old_codes.shape) != tuple(codes.shape):
                raise ValueError(
                    f"new code shape {tuple(codes.shape)} differs from old "
                    f"{tuple(old_codes.shape)}")
            backup_path = atomic_save(codes, code_path, args.apply, args.backup)
            processed += 1
            print("  write action    : "
                  f"{'overwritten' if args.apply else 'dry-run, not written'}")
            if backup_path is not None:
                print(f"  backup path     : {backup_path}")
            print("  status          : OK")
        except Exception as exc:
            failed += 1
            print("  status          : FAIL")
            print(f"  reason          : {exc}")

    print("=" * 80)
    print(f"summary: processed={processed} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
