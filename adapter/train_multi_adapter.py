#!/usr/bin/env python
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter.train_adapter import load_code, load_csi, run_training, split_paths


DEFAULTS = {
    "source_seed": 1014,
    "target_seed": 1024,
    "source_encoder": "transnet",
    "source_decoder": "transnet",
    "target_encoder": "transnet",
    "target_decoder": "transnet",
    "mapper_type": "affine_residual_mlp",
    "hidden_dim": 512,
    "lowrank_rank": 64,
    "bottleneck_dim": 128,
    "num_groups": 16,
    "group_hidden": 64,
    "gate_hidden": 64,
    "gate_init": 0.5,
    "num_tokens": 16,
    "token_hidden": 64,
    "channel_hidden": 64,
    "num_heads": 2,
    "transformer_ffn_dim": 128,
    "attention_dim": 32,
    "attention_heads": 4,
    "attention_dropout": 0.0,
    "attention_scale": 0.1,
    "attention_input": "value_delta",
    "attention_use_position": True,
    "num_experts": 4,
    "flow_hidden_dim": 128,
    "whole_mlp_dims": None,
    "whole_mlp_activation": "gelu",
    "num_blocks": 4,
    "dropout": 0.0,
    "residual_scale": 0.1,
    "learnable_residual_gate": False,
    "gate_max": 0.5,
    "gate_mode": "block",
    "final_gate_max": 1.0,
    "final_gate_init": 1.0,
    "adaptive_gate_hidden": 128,
    "gate_l1": 0.0,
    "no_block_norm": False,
    "use_final_norm": False,
    "train_affine": False,
    "no_affine_alignment": False,
    "align_ridge": 1.0,
    "affine_fit_splits": "train",
    "lambda_code": 1.0,
    "lambda_recon": 0.0,
    "lambda_feature": 0.0,
    "lambda_encoder_consistency": 0.0,
    "encoder_consistency_target": "mapped",
    "lambda_delta_norm": 0.0,
    "lambda_teacher_code": 0.0,
    "teacher_train_code": None,
    "lambda_fisher": 0.0,
    "fisher_basis_path": None,
    "fisher_rank": 0,
    "fisher_weight_power": 0.5,
    "fisher_weight_max": 4.0,
    "gradient_diagnostics_every": 0,
    "train_last_blocks": 0,
    "init_mapper_checkpoint": None,
    "init_mapper_use_ema": False,
    "code_noise_std": 0.0,
    "stage1_epochs": 0,
    "stage1_code_noise_std": 0.0,
    "stage1_lambda_recon": 0.0,
    "stage1_lambda_encoder_consistency": 0.0,
    "stage2_lr": None,
    "stage2_affine_lr_multiplier": 1.0,
    "stage2_affine_freeze_epochs": 0,
    "stage2_recon_warmup_epochs": 0,
    "stage2_encoder_delay_epochs": 0,
    "stage2_encoder_warmup_epochs": 0,
    "stage2_noise_decay_epochs": 0,
    "ema_decay": 0.0,
    "ema_start_epoch": 1,
    "ema_update_every": 1,
    "code_loss_type": "mse",
    "sensitivity_source": "jacobian",
    "sensitivity_power": 1.0,
    "sensitivity_hutchinson": 8,
    "sensitivity_probe_samples": 2048,
    "std_weight_min": 0.25,
    "std_weight_max": 4.0,
    "std_weight_eps": 1e-6,
    "epochs": 100,
    "batch_size": 256,
    "workers": 0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "eta_min": 1e-4,
    "eval_every": 10,
    "export_codewords": False,
    "max_train_samples": 0,
    "max_eval_samples": 0,
    "seed": 2026,
    "gpu": None,
    "cpu": False,
    "channel": 2,
    "nt": 32,
    "nc": 32,
    "decoder": "transnet",
    "encoder": "transnet",
    "cr": 4,
    "d_model": 64,
    "dim_feedforward": 2048,
    "hidden": 16,
    "decoder_num_blocks": 2,
    "train_csi": "/nfs5/zxd/Huawei/datasets/COST2100/in_train.pt",
    "val_csi": "/nfs5/zxd/Huawei/datasets/COST2100/in_val.pt",
    "test_csi": "/nfs5/zxd/Huawei/datasets/COST2100/in_test.pt",
    "decoder_args_json": None,
    "encoder_checkpoint": None,
    "encoder_args_json": None,
}


def load_json(path):
    return json.loads(Path(path).read_text())


def derive_exp_name(cfg):
    norm_name = "no_block_norm" if cfg["no_block_norm"] else "block_norm"
    suffix = f"_h{cfg['hidden_dim']}" if cfg.get("hidden_dim") is not None else ""
    if cfg.get("mapper_type") == "affine_lowrank_residual":
        suffix += f"_rank{cfg['lowrank_rank']}"
    if cfg.get("mapper_type") == "affine_film_residual_mlp":
        suffix += "_film"
    if cfg.get("mapper_type") == "affine_multiscale_residual_mlp":
        suffix += f"_multiscale_bn{cfg['bottleneck_dim']}"
    if cfg.get("mapper_type") == "affine_bottleneck_residual":
        suffix += f"_bn{cfg['bottleneck_dim']}"
    if cfg.get("mapper_type") == "affine_group_gated":
        suffix += (
            f"_groups{cfg['num_groups']}_gh{cfg['group_hidden']}"
            f"_gateh{cfg['gate_hidden']}_ginit{cfg['gate_init']}")
    if cfg.get("mapper_type") == "affine_token_mixer":
        suffix += (
            f"_tokens{cfg['num_tokens']}_th{cfg['token_hidden']}"
            f"_ch{cfg['channel_hidden']}")
    if cfg.get("mapper_type") == "affine_tiny_transformer":
        suffix += (
            f"_tokens{cfg['num_tokens']}_heads{cfg['num_heads']}"
            f"_ffn{cfg['transformer_ffn_dim']}")
    if cfg.get("mapper_type") == "affine_residual_mlp_attention":
        suffix += "_attnd{}_h{}_as{}_{}".format(
            cfg["attention_dim"], cfg["attention_heads"],
            cfg["attention_scale"], cfg["attention_input"])
        if not cfg["attention_use_position"]:
            suffix += "_nopos"
    if cfg.get("mapper_type") == "affine_moe_bottleneck":
        suffix += (
            f"_experts{cfg['num_experts']}_bn{cfg['bottleneck_dim']}"
            f"_gateh{cfg['gate_hidden']}")
    if cfg.get("mapper_type") == "affine_coupling_flow":
        suffix += f"_flowh{cfg['flow_hidden_dim']}"
    if cfg.get("mapper_type") in (
            "affine_whole_residual_mlp",
            "affine_whole_direct_mlp"):
        dims = cfg.get("whole_mlp_dims") or []
        suffix += "_whole" + "x".join(str(dim) for dim in dims)
        suffix += f"_act{cfg.get('whole_mlp_activation', 'gelu')}"
    if cfg.get("learnable_residual_gate"):
        suffix += f"_gate{cfg['gate_max']}"
    if cfg.get("gate_mode", "block") != "block":
        suffix += f"_{cfg['gate_mode']}"
        if cfg.get("gate_mode") == "final_unbounded":
            suffix += f"_fginit{cfg['final_gate_init']}"
        else:
            suffix += f"_fgmax{cfg['final_gate_max']}_fginit{cfg['final_gate_init']}"
    if cfg.get("gate_l1", 0.0):
        suffix += f"_gl1{cfg['gate_l1']}"
    if cfg.get("code_loss_type", "mse") != "mse":
        suffix += f"_{cfg['code_loss_type']}"
    if cfg.get("lambda_feature", 0.0):
        suffix += f"_feat{cfg['lambda_feature']}"
    if cfg.get("lambda_encoder_consistency", 0.0):
        suffix += (
            f"_enc{cfg['lambda_encoder_consistency']}"
            f"_{cfg['encoder_consistency_target']}")
    if cfg.get("lambda_delta_norm", 0.0):
        suffix += f"_dn{cfg['lambda_delta_norm']}"
    if cfg.get("lambda_teacher_code", 0.0):
        suffix += f"_ltc{cfg['lambda_teacher_code']}"
        teacher = cfg.get("teacher_train_code") or ""
        if "reencode0" in teacher:
            suffix += "_tre0"
        elif "reencode_1014to1024_s20" in teacher or "refined_codes_reencode" in teacher:
            suffix += "_tres20"
        elif "s20" in teacher or "gen_s20" in teacher:
            suffix += "_ts20"
    if cfg.get("lambda_fisher", 0.0):
        suffix += (
            f"_fishr{cfg.get('fisher_rank', 0)}"
            f"b{cfg['lambda_fisher']}p{cfg.get('fisher_weight_power', 0.5)}")
    if cfg.get("train_last_blocks", 0):
        suffix += f"_lastb{cfg['train_last_blocks']}"
    if cfg.get("init_mapper_checkpoint"):
        suffix += "_initckpt"
        if cfg.get("init_mapper_use_ema"):
            suffix += "ema"
    if cfg.get("code_noise_std", 0.0):
        suffix += f"_noise{cfg['code_noise_std']}"
    if cfg.get("stage1_epochs", 0):
        suffix += (
            f"_stage1ep{cfg['stage1_epochs']}"
            f"_s1noise{cfg.get('stage1_code_noise_std', 0.0)}"
            f"_s2lr{cfg.get('stage2_lr')}"
            f"_s2afflr{cfg.get('stage2_affine_lr_multiplier', 1.0)}")
    if cfg.get("ema_decay", 0.0):
        suffix += (
            f"_ema{cfg['ema_decay']}"
            f"_emastart{cfg.get('ema_start_epoch', 1)}")
    return (
        f"code{cfg['lambda_code']}_rec{cfg['lambda_recon']}_lr{cfg['lr']}"
        f"_ep{cfg['epochs']}_{norm_name}_ridge{cfg['align_ridge']}"
        f"_ta{int(bool(cfg['train_affine']))}_rs{cfg['residual_scale']}{suffix}"
    )


def normalize_config(raw, config_path):
    cfg = deepcopy(DEFAULTS)
    cfg.update(raw)
    source_arch = cfg.get(
        "source_arch",
        f"{cfg['source_encoder']}_{cfg['source_decoder']}")
    target_arch = cfg.get(
        "target_arch",
        f"{cfg['target_encoder']}_{cfg['target_decoder']}")
    cfg.setdefault(
        "source_exp",
        f"exps/COST2100/in/base/seed{cfg['source_seed']}/{source_arch}")
    cfg.setdefault(
        "target_exp",
        f"exps/COST2100/in/base/seed{cfg['target_seed']}/{target_arch}")
    cfg.setdefault("target_decoder_exp", cfg["target_exp"])
    cfg.setdefault(
        "decoder_checkpoint",
        str(Path(cfg["target_decoder_exp"]) / "checkpoints" / "best_nmse.pth"))
    if not cfg.get("encoder_checkpoint"):
        cfg["encoder_checkpoint"] = cfg["decoder_checkpoint"]
    decoder_args = Path(cfg["target_decoder_exp"]) / "args.json"
    if not cfg.get("decoder_args_json") and decoder_args.exists():
        cfg["decoder_args_json"] = str(decoder_args)
    if not cfg.get("encoder_args_json") and cfg.get("decoder_args_json"):
        cfg["encoder_args_json"] = cfg["decoder_args_json"]
    cfg.setdefault("decoder", cfg["target_decoder"])
    cfg.setdefault("encoder", cfg["target_encoder"])
    cfg.setdefault("exp_seed", f"seed{cfg['source_seed']}")
    cfg.setdefault("exp_arch", cfg["source_encoder"])
    cfg.setdefault("exp_name", derive_exp_name(cfg))
    cfg.setdefault(
        "exp_dir",
        str(Path("adapter") / "exps" / cfg["mapper_type"] /
            cfg["exp_seed"] / cfg["exp_arch"] / cfg["exp_name"]))
    for prefix in ("source", "target"):
        for split in ("train", "val", "test"):
            cfg.setdefault(f"{prefix}_{split}_code", None)
    cfg["_config_path"] = str(config_path)
    return cfg


def to_namespace(cfg):
    payload = {key: value for key, value in cfg.items() if not key.startswith("_")}
    return SimpleNamespace(**payload)


def data_group_key(cfg):
    args = to_namespace(cfg)
    values = [
        split_paths(args, "source", "train"),
        split_paths(args, "source", "val"),
        split_paths(args, "source", "test"),
        split_paths(args, "target", "train"),
        split_paths(args, "target", "val"),
        split_paths(args, "target", "test"),
        cfg["train_csi"],
        cfg["val_csi"],
        cfg["test_csi"],
        cfg["channel"],
        cfg["nt"],
        cfg["nc"],
        cfg["max_train_samples"],
        cfg["max_eval_samples"],
    ]
    return tuple(str(value) for value in values)


def load_shared_data(cfg):
    args = to_namespace(cfg)
    data = {}
    for split in ("train", "val", "test"):
        max_samples = cfg["max_train_samples"] if split == "train" else cfg["max_eval_samples"]
        source = load_code(split_paths(args, "source", split), max_samples)
        target = load_code(split_paths(args, "target", split), max_samples)
        csi = load_csi(
            cfg[f"{split}_csi"],
            channel=cfg["channel"],
            nt=cfg["nt"],
            nc=cfg["nc"],
            max_samples=max_samples)
        for tensor in (source, target, csi):
            tensor.share_memory_()
        data[split] = (source, target, csi)
    return data


def worker_main(cfg, shared_data, torch_num_threads):
    if torch_num_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(torch_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(torch_num_threads)
        torch.set_num_threads(torch_num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    run_training(to_namespace(cfg), preloaded_data=shared_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--start_gap_seconds", type=float, default=2.0)
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=int(os.environ.get("torch_num_threads", 2)),
        help="intra-op CPU threads per training worker; use <=0 to keep PyTorch default",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_paths = sorted(Path(args.config_dir).glob("*.json"))
    if not config_paths:
        raise FileNotFoundError(f"No json configs found in {args.config_dir}")
    configs = [normalize_config(load_json(path), path) for path in config_paths]
    groups = {}
    for cfg in configs:
        groups.setdefault(data_group_key(cfg), []).append(cfg)

    if args.dry_run:
        for cfg in configs:
            print(f"{cfg['_config_path']} -> gpu={cfg['gpu']} exp_dir={cfg['exp_dir']}")
        return

    ctx = mp.get_context("spawn")
    processes = []
    for group_idx, group_configs in enumerate(groups.values(), 1):
        print(
            f"Loading shared data group {group_idx}: "
            f"{len(group_configs)} config(s)",
            flush=True)
        shared_data = load_shared_data(group_configs[0])
        for cfg in group_configs:
            print(
                f"Launching {cfg['_config_path']} "
                f"gpu={cfg['gpu']} threads={args.torch_num_threads} "
                f"exp_dir={cfg['exp_dir']}",
                flush=True)
            proc = ctx.Process(
                target=worker_main,
                args=(cfg, shared_data, args.torch_num_threads))
            proc.start()
            processes.append(proc)
            time.sleep(args.start_gap_seconds)

    failed = []
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failed.append((proc.pid, proc.exitcode))
    if failed:
        raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")


if __name__ == "__main__":
    main()
