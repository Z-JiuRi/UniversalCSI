#!/usr/bin/env python3
"""Generate exp25's all-split affine-initialization oracle diagnostic."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "branches"
    output.mkdir(parents=True, exist_ok=True)
    config = {
        # Exact exp25 D_s20_ltc1_lc0 architecture and training settings.
        "mapper_type": "affine_residual_mlp",
        "hidden_dim": 512,
        "num_blocks": 4,
        "residual_scale": 0.4,
        "train_affine": True,
        "align_ridge": 1.0,
        "affine_fit_splits": "train_val_test",
        "lambda_code": 0.0,
        "lambda_teacher_code": 1.0,
        "teacher_train_code": (
            "adapter/exps/exp16_pure_middle/gen_s20_train/"
            "train_refined_code.pt"),
        "lambda_recon": 1000.0,
        "lambda_encoder_consistency": 2.0,
        "encoder_consistency_target": "target",
        "code_noise_std": 0.02,
        "ema_decay": 0.995,
        "ema_start_epoch": 1,
        "ema_update_every": 1,
        "epochs": 400,
        "batch_size": 1024,
        "lr": 5e-4,
        "eta_min": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "eval_every": 10,
        "export_codewords": True,
        "seed": 1024,
        "source_seed": 1014,
        "target_seed": 1024,
        "source_encoder": "transnet",
        "source_decoder": "transnet",
        "target_encoder": "transnet",
        "target_decoder": "transnet",
        "gpu": 0,
        "exp_name": "01_s20_ltc1_lc0_affine_train_val_test",
        "exp_dir": (
            "adapter/exps/exp34_oracle_affine/"
            "01_s20_ltc1_lc0_affine_train_val_test"),
        "description": (
            "ORACLE DIAGNOSTIC ONLY. Exp25 D_s20_ltc1_lc0, with the "
            "initial least-squares affine fit using train+val+test source/target "
            "code pairs before normal training."),
    }
    path = output / "01_s20_ltc1_lc0_affine_train_val_test.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    (ROOT / "manifest.json").write_text(
        json.dumps([{"name": config["exp_name"], "gpu": config["gpu"],
                     "config": str(path), "exp_dir": config["exp_dir"]}],
                   indent=2) + "\n",
        encoding="utf-8")
    print(f"Generated exp34 oracle-affine config: {path}")


if __name__ == "__main__":
    main()
