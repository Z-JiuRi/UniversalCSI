import argparse
import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_ENCODERS = [
    "csinet",
    "cnn",
    "cbam_cnn",
    "crnet",
    "clnet",
    "transnet",
    "resnet",
    "dscnn",
    "convnext",
    "mlp_mixer",
    "attention_cnn",
    "swin",
    "mlp_ae",
    "sparse_resnet",
]

DEFAULT_DECODERS = [
    "transnet",
    "cnn_residual",
    "hybrid",
]


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def task_done(exp_dir):
    run_log = exp_dir / "run.log"
    codewords = exp_dir / "codewords"
    required_codewords = [
        codewords / "train_code.pt",
        codewords / "val_code.pt",
        codewords / "test_code.pt",
    ]
    if not run_log.is_file():
        return False
    if not all(path.is_file() for path in required_codewords):
        return False
    return "Final test loss" in run_log.read_text(errors="ignore")


def build_command(args, encoder, decoder):
    exp_name = f"{args.exp_root}/{encoder}_{decoder}"
    cmd = [
        args.python,
        "main.py",
        "--exp_name", exp_name,
        "--train_path", args.train_path,
        "--val_path", args.val_path,
        "--test_path", args.test_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--workers", str(args.workers),
        "--cr", str(args.cr),
        "--nt", str(args.nt),
        "--nc", str(args.nc),
        "--channel", str(args.channel),
        "--d_model", str(args.d_model),
        "--dim_feedforward", str(args.dim_feedforward),
        "--scheduler", args.scheduler,
        "--lr_init", str(args.lr_init),
        "--weight_decay", str(args.weight_decay),
        "--encoder", encoder,
        "--decoder", decoder,
        "--gpu", str(args.gpu),
        "--seed", str(args.seed),
    ]
    if args.code_adapter:
        cmd.append("--code_adapter")
    return exp_name, cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run the full UniversalCSI encoder x decoder matrix.")
    parser.add_argument("--python", default="/home/z-jiuri/.envs/miniconda3/envs/torch/bin/python")
    parser.add_argument("--train_path", default="/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_train.pt")
    parser.add_argument("--val_path", default="/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_val.pt")
    parser.add_argument("--test_path", default="/home/z-jiuri/workspace/Huawei/TransNet/data/COST2100/in_test.pt")
    parser.add_argument("--exp_root", default="real_matrix_2epoch")
    parser.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS))
    parser.add_argument("--decoders", default=",".join(DEFAULT_DECODERS))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--scheduler", choices=["const", "cosine"], default="const")
    parser.add_argument("--lr_init", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--code_adapter", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    encoders = parse_csv_list(args.encoders)
    decoders = parse_csv_list(args.decoders)
    all_tasks = [build_command(args, encoder, decoder)
                 for encoder in encoders for decoder in decoders]

    exp_base = Path("exps") / args.exp_root
    exp_base.mkdir(parents=True, exist_ok=True)
    status_path = exp_base / "matrix_status.json"

    pending = []
    skipped = []
    for exp_name, cmd in all_tasks:
        exp_dir = Path("exps") / exp_name
        if not args.force and task_done(exp_dir):
            skipped.append(exp_name)
        else:
            pending.append((exp_name, cmd))

    status = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "parallel": args.parallel,
        "skipped": skipped,
        "finished": [],
        "failed": [],
        "pending_total": len(pending),
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))

    running = []
    while pending or running:
        while pending and len(running) < args.parallel:
            exp_name, cmd = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"[start] {exp_name}", flush=True)
            exp_dir = Path("exps") / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = exp_dir / "matrix_stdout.log"
            stdout_file = stdout_path.open("w")
            process = subprocess.Popen(cmd, cwd=os.getcwd(), env=env,
                                       stdout=stdout_file,
                                       stderr=subprocess.STDOUT)
            running.append((exp_name, process, time.time(), stdout_file))

        time.sleep(10)

        still_running = []
        for exp_name, process, start_time, stdout_file in running:
            ret = process.poll()
            if ret is None:
                still_running.append((exp_name, process, start_time, stdout_file))
                continue
            stdout_file.close()
            elapsed = time.time() - start_time
            exp_dir = Path("exps") / exp_name
            record = {
                "exp_name": exp_name,
                "returncode": ret,
                "elapsed_seconds": elapsed,
            }
            if ret == 0 and task_done(exp_dir):
                print(f"[done] {exp_name} ({elapsed:.1f}s)", flush=True)
                status["finished"].append(record)
            else:
                print(f"[failed] {exp_name} returncode={ret} ({elapsed:.1f}s)",
                      flush=True)
                status["failed"].append(record)
            status_path.write_text(json.dumps(status, indent=2,
                                              ensure_ascii=False))
        running = still_running

    status["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    print(f"finished={len(status['finished'])} skipped={len(status['skipped'])} "
          f"failed={len(status['failed'])}")
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
