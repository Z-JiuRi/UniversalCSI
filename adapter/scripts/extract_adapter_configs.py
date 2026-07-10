#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_root",
        default="adapter/exps/affine_residual_mlp/seed1014/transnet")
    parser.add_argument("--out_dir", default="adapter/configs/exp1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for args_path in sorted(exp_root.glob("*/args.json")):
        exp_dir = args_path.parent
        cfg = json.loads(args_path.read_text())
        cfg["exp_dir"] = str(exp_dir)
        cfg.setdefault("exp_name", exp_dir.name)
        cfg.setdefault("exp_seed", "seed1014")
        cfg.setdefault("exp_arch", "transnet")
        out_path = out_dir / f"{count + 1:03d}_{safe_name(exp_dir.name)}.json"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite")
        out_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        count += 1

    print(f"saved {count} config(s) to {out_dir}")


if __name__ == "__main__":
    main()
