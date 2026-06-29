#!/usr/bin/env python
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="exps/COST2100/in/encoder_canonical")
    parser.add_argument("--out-dir", default="analysis_outputs/encoder_canonical")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for codeword tensor analysis.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index used by codeword analysis.")
    parser.add_argument("--skip-codewords", action="store_true",
                        help="Only parse logs and draw log-based figures.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    py = sys.executable

    run([
        py, str(here / "parse_logs.py"),
        "--root", args.root,
        "--out-dir", args.out_dir,
    ])

    if not args.skip_codewords:
        run([
            py, str(here / "analyze_codewords.py"),
            "--log-summary", str(Path(args.out_dir) / "experiment_log_summary.csv"),
            "--out-dir", args.out_dir,
            "--sample-size", str(args.sample_size),
            "--device", args.device,
            "--gpu", str(args.gpu),
        ])

    run([
        py, str(here / "plot_summary.py"),
        "--in-dir", args.out_dir,
        "--fig-dir", str(Path(args.out_dir) / "figures"),
    ])


if __name__ == "__main__":
    main()
