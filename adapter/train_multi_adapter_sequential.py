#!/usr/bin/env python
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter.train_multi_adapter import (  # noqa: E402
    load_json,
    load_shared_data,
    normalize_config,
    to_namespace,
)
from adapter.train_adapter import run_training  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--gpu_filter", type=int, default=None)
    parser.add_argument("--torch_num_threads", type=int, default=4)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def set_threads(num_threads):
    if num_threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def main():
    args = parse_args()
    set_threads(args.torch_num_threads)
    paths = sorted(Path(args.config_dir).glob("*.json"))
    configs = [normalize_config(load_json(path), path) for path in paths]
    if args.gpu_filter is not None:
        configs = [cfg for cfg in configs if cfg.get("gpu") == args.gpu_filter]
    if not configs:
        raise FileNotFoundError(
            f"No matching configs in {args.config_dir} for gpu={args.gpu_filter}")

    for idx, cfg in enumerate(configs, 1):
        exp_dir = Path(cfg["exp_dir"])
        metrics_path = exp_dir / "metrics.json"
        if args.skip_completed and metrics_path.exists():
            print(f"[{idx}/{len(configs)}] skip completed {exp_dir}", flush=True)
            continue

        print(
            f"[{idx}/{len(configs)}] run {cfg['_config_path']} "
            f"gpu={cfg['gpu']} exp_dir={exp_dir}",
            flush=True)
        if args.dry_run:
            continue

        # Load per-config to keep memory bounded for long sequential workers.
        shared_data = load_shared_data(cfg)
        run_training(to_namespace(deepcopy(cfg)), preloaded_data=shared_data)


if __name__ == "__main__":
    main()
