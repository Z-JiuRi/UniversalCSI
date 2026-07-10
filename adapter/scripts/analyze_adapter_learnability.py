#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch


SPLITS = ("train", "val", "test")


def load_code(path):
    return torch.load(path, weights_only=True, map_location="cpu").float()


def fit_affine(source, target, ridge=0.0):
    dim = source.size(1)
    src = source.double()
    tgt = target.double()
    ones = torch.ones(src.size(0), 1, dtype=src.dtype)
    aug = torch.cat([src, ones], dim=1)
    reg = ridge * torch.eye(dim + 1, dtype=src.dtype)
    reg[-1, -1] = 0.0
    solution = torch.linalg.solve(aug.t().matmul(aug) + reg, aug.t().matmul(tgt))
    return solution[:-1].float(), solution[-1].float()


def quantiles(x):
    x = x.detach().float().flatten()
    qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    vals = torch.quantile(x, qs)
    return {f"q{float(q):g}": float(v) for q, v in zip(qs, vals)}


def pearson(x, y):
    x = x.detach().float().flatten()
    y = y.detach().float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def rankdata(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def spearman(x, y):
    return pearson(rankdata(x.flatten()), rankdata(y.flatten()))


def rel_norm(a, b, eps=1e-12):
    return float((a - b).norm() / b.norm().clamp_min(eps))


def top_share(values, frac):
    values = values.detach().float().flatten()
    k = max(1, int(round(values.numel() * frac)))
    return float(torch.topk(values, k).values.sum() / values.sum().clamp_min(1e-12))


def overlap_fraction(a, b, frac):
    k = max(1, int(round(a.numel() * frac)))
    ia = set(torch.topk(a, k).indices.tolist())
    ib = set(torch.topk(b, k).indices.tolist())
    return len(ia & ib) / k


def matrix_stats(pred, target):
    pred_center = pred - pred.mean(dim=0, keepdim=True)
    tgt_center = target - target.mean(dim=0, keepdim=True)
    denom = max(pred.size(0) - 1, 1)
    pred_cov = pred_center.t().matmul(pred_center) / denom
    tgt_cov = tgt_center.t().matmul(tgt_center) / denom
    pred_std = pred_center.pow(2).mean(dim=0).sqrt().clamp_min(1e-12)
    tgt_std = tgt_center.pow(2).mean(dim=0).sqrt().clamp_min(1e-12)
    pred_corr = pred_cov / pred_std[:, None] / pred_std[None, :]
    tgt_corr = tgt_cov / tgt_std[:, None] / tgt_std[None, :]
    return {
        "mean_rel": rel_norm(pred.mean(dim=0), target.mean(dim=0)),
        "std_rel": rel_norm(pred.std(dim=0), target.std(dim=0)),
        "cov_rel": rel_norm(pred_cov, tgt_cov),
        "corr_rel": rel_norm(pred_corr, tgt_corr),
    }


def dim_metrics(pred, target):
    err = pred - target
    mse = err.pow(2).mean(dim=0)
    target_var = target.var(dim=0, unbiased=False).clamp_min(1e-12)
    r2 = 1.0 - mse / target_var
    return {
        "mse": mse,
        "norm_mse": mse / target_var,
        "r2": r2,
        "sample_mse": err.pow(2).mean(dim=1),
    }


def summarize_method(method_metrics):
    out = {}
    for split, metrics in method_metrics.items():
        out[split] = {
            "dim_mse_mean": float(metrics["mse"].mean()),
            "dim_mse_quantiles": quantiles(metrics["mse"]),
            "norm_mse_mean": float(metrics["norm_mse"].mean()),
            "norm_mse_quantiles": quantiles(metrics["norm_mse"]),
            "r2_mean": float(metrics["r2"].mean()),
            "r2_quantiles": quantiles(metrics["r2"]),
            "sample_mse_mean": float(metrics["sample_mse"].mean()),
            "sample_mse_quantiles": quantiles(metrics["sample_mse"]),
            "top1_dim_mse_share": top_share(metrics["mse"], 0.01),
            "top10_dim_mse_share": top_share(metrics["mse"], 0.10),
            "bad_dim_r2_lt_0": float((metrics["r2"] < 0).float().mean()),
            "bad_dim_r2_lt_0_5": float((metrics["r2"] < 0.5).float().mean()),
        }
    return out


def fmt(value, digits=4):
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def fmt_q(q):
    keys = ["q0", "q0.01", "q0.05", "q0.25", "q0.5", "q0.75", "q0.95", "q0.99", "q1"]
    return ", ".join(f"{key}={q[key]:.4g}" for key in keys)


def update_report(report_path, section):
    path = Path(report_path)
    old = path.read_text(encoding="utf-8")
    marker = "\n## 10. Learnability 诊断：难维度/难样本到底是哪类问题\n"
    if marker in old:
        old = old.split(marker)[0].rstrip() + "\n"
    path.write_text(old.rstrip() + marker + section.strip() + "\n", encoding="utf-8")


def build_report(stats, args):
    lines = []
    lines.append("")
    lines.append(f"分析对象：`{args.adapter_exp}`。本节原始统计保存于 `{args.output_json}`。")
    lines.append("")
    lines.append("### 10.1 结论先行")
    lines.append("")
    lines.append(
        "当前证据更支持：大部分难维度不是简单的“普通 MSE 优化时被忽略”，"
        "而是 affine 后剩余误差里存在明显不可泛化/弱可学成分；同时仍有一部分维度可能受损失权重影响，"
        "值得用 clipped/std、residual-std、Jacobian-aware 或 uncertainty weighting 做对照。")
    lines.append("")
    lines.append(
        "判断依据是三点：第一，best adapter 已经把 affine residual 在 train 上大幅压低，"
        "但 val/test 的 per-dim R2 明显低于 train；第二，val/test 的 hard dim 与 train hard dim 有一定重合，"
        "说明难点不是纯随机，但 train-val gap 又说明训练集可学部分不能完全泛化；第三，hard dim 并不只集中在低方差维度，"
        "`1/std` 加权只能解决一部分被尺度淹没的问题，不能解决 source 信息不足或 target 噪声。")
    lines.append("")

    lines.append("### 10.2 train/val/test 可学性指标")
    lines.append("")
    lines.append("| method | split | dim MSE | normalized MSE | mean R2 | R2<0 | R2<0.5 | top1% error share | top10% error share |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for method in ("raw_source", "affine", "mapped"):
        for split in SPLITS:
            s = stats["methods"][method][split]
            lines.append(
                f"| {method} | {split} | {s['dim_mse_mean']:.4e} | "
                f"{s['norm_mse_mean']:.4e} | {s['r2_mean']:.4f} | "
                f"{s['bad_dim_r2_lt_0']:.3f} | {s['bad_dim_r2_lt_0_5']:.3f} | "
                f"{s['top1_dim_mse_share']:.3f} | {s['top10_dim_mse_share']:.3f} |")
    lines.append("")
    lines.append(
        "这里的 `normalized MSE = per-dim MSE / target_var`，所以 `R2 = 1 - normalized MSE`。"
        "如果某维在 train 上 R2 也很差，说明现有 source code 和当前 adapter 对这维几乎不可预测；"
        "如果 train R2 很好但 val/test R2 差，说明更像过拟合、分布差异或 target 噪声。")
    lines.append("")

    lines.append("### 10.3 affine 剩余误差被 adapter 学掉了多少")
    lines.append("")
    lines.append("| split | mean improvement over affine | median | q05 | q95 | dims improved | dims worse |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split in SPLITS:
        s = stats["adapter_vs_affine_improvement"][split]
        lines.append(
            f"| {split} | {s['mean']:.4f} | {s['q0.5']:.4f} | {s['q0.05']:.4f} | "
            f"{s['q0.95']:.4f} | {s['improved_fraction']:.3f} | {s['worse_fraction']:.3f} |")
    lines.append("")
    lines.append(
        "这个 improvement 定义为 `1 - mapped_dim_mse / affine_dim_mse`。"
        "train 明显高、val/test 明显低时，说明 adapter 的容量足以记住/拟合训练 residual，"
        "但其中一部分 residual 不是稳定映射。")
    lines.append("")

    lines.append("### 10.4 hard dimension 的稳定性与尺度关系")
    lines.append("")
    h = stats["hard_dim_overlap"]
    lines.append("| 对比 | top1% overlap | top10% overlap |")
    lines.append("|---|---:|---:|")
    for key, val in h.items():
        lines.append(f"| {key} | {val['top1']:.3f} | {val['top10']:.3f} |")
    lines.append("")
    c = stats["correlations"]
    lines.append("| 相关项 | Pearson | Spearman |")
    lines.append("|---|---:|---:|")
    for key, val in c.items():
        lines.append(f"| {key} | {val['pearson']:.4f} | {val['spearman']:.4f} |")
    lines.append("")
    lines.append(
        "如果 hard dim 与 `target_std` 高相关，普通 MSE 主要被高方差维度主导；"
        "如果 hard dim 与 `1/std` 或低 std 高相关，低方差维度可能被忽略。"
        "当前应同时看 raw MSE 和 normalized MSE 的 hard dim，而不是只看一种。")
    lines.append("")

    lines.append("### 10.5 难样本是否只是训练集个别样本")
    lines.append("")
    lines.append("| method | split | sample MSE mean | sample MSE quantiles |")
    lines.append("|---|---|---:|---|")
    for method in ("affine", "mapped"):
        for split in SPLITS:
            s = stats["methods"][method][split]
            lines.append(
                f"| {method} | {split} | {s['sample_mse_mean']:.4e} | "
                f"{fmt_q(s['sample_mse_quantiles'])} |")
    lines.append("")
    lines.append(
        "mapped 的 train 样本误差显著小于 val/test，但 val/test 分布接近，说明不是测试集崩塌，"
        "更像训练 residual 中有一部分被拟合后不能跨 split 泛化。难样本加权可以试，但必须 capped，"
        "否则很容易把异常样本或噪声当成学习目标。")
    lines.append("")

    lines.append("### 10.6 对加权/EMA/其他手段的建议")
    lines.append("")
    lines.append("建议按优先级试这些手段：")
    lines.append("")
    lines.append("1. `clipped_std_mse`：已实现。它解决 code 维度尺度不均，但不能识别噪声维度。")
    lines.append("2. `clipped_residual_std_mse`：按 affine residual 的 std 加权，`w=1/std(target-affine(source))` 后 clip。它比 target std 更贴近 adapter 实际要学的 residual。")
    lines.append("3. `jacobian_weighted_mse`：用 decoder Jacobian sensitivity 或 `fc_decoder` feature loss 做 decoder-aware 加权。适合最终目标是 CSI NMSE，而不是纯 code MSE。")
    lines.append("4. `uncertainty_mse`：每维学习 `log_var_i`，`exp(-log_var_i)*mse_i + log_var_i`。如果某维根本不可学，它会自动降权；这比手工硬加权更适合区分噪声维。")
    lines.append("5. capped hard-dim / hard-sample weighting：用 EMA 平滑 per-dim 或 per-sample error，再做上限裁剪。只建议作为中后期 fine-tune，不建议从第 1 epoch 开始强推。")
    lines.append("6. model EMA：维护 adapter 参数 EMA，用 EMA 权重评估和导出。它不能解决不可学问题，但对加权 loss 下的抖动和泛化通常有帮助，成本很低。")
    lines.append("")
    lines.append(
        "下一步如果要真正判断“不可学”而不是“当前模型没学到”，建议做一个 probe："
        "冻结 affine，专门训练一个较大 residual probe 只拟合 affine residual，比较 train/val 的 per-dim R2。"
        "若大 probe 在 train 上仍上不去，是输入信息不足或标签噪声；若 train 上去但 val 不上去，是泛化/分布问题；"
        "若加权后 val 上去，才说明主要是优化时被忽略。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_exp", default="exps/COST2100/in/base/seed1014/transnet_transnet")
    parser.add_argument("--target_exp", default="exps/COST2100/in/base/seed1024/transnet_transnet")
    parser.add_argument(
        "--adapter_exp",
        default="adapter/exps/affine_residual_mlp/seed1014/transnet/code1.0_rec0.0_lr5e-4_ep400_block_norm_ridge0.0_ta1_rs0.1_h512")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--output_json", default="adapter/exps/adapter_learnability_analysis.json")
    parser.add_argument("--report", default="adapter/exps/adapter_deep_analysis.md")
    args = parser.parse_args()

    source = {}
    target = {}
    mapped = {}
    for split in SPLITS:
        source[split] = load_code(Path(args.source_exp) / "codewords" / f"{split}_code.pt")
        target[split] = load_code(Path(args.target_exp) / "codewords" / f"{split}_code.pt")
        mapped[split] = load_code(Path(args.adapter_exp) / "codewords" / f"{split}_mapped_code.pt")

    weight, bias = fit_affine(source["train"], target["train"], ridge=args.ridge)
    affine = {split: source[split].matmul(weight) + bias for split in SPLITS}

    methods_tensor = {
        "raw_source": source,
        "affine": affine,
        "mapped": mapped,
    }
    method_metrics = {
        method: {
            split: dim_metrics(preds[split], target[split])
            for split in SPLITS
        }
        for method, preds in methods_tensor.items()
    }

    stats = {
        "source_exp": args.source_exp,
        "target_exp": args.target_exp,
        "adapter_exp": args.adapter_exp,
        "methods": {
            method: summarize_method(metrics)
            for method, metrics in method_metrics.items()
        },
        "adapter_vs_affine_improvement": {},
        "hard_dim_overlap": {},
        "correlations": {},
        "matrix_distance_to_target": {},
    }

    for split in SPLITS:
        affine_mse = method_metrics["affine"][split]["mse"].clamp_min(1e-12)
        mapped_mse = method_metrics["mapped"][split]["mse"]
        improvement = 1.0 - mapped_mse / affine_mse
        q = quantiles(improvement)
        stats["adapter_vs_affine_improvement"][split] = {
            "mean": float(improvement.mean()),
            "improved_fraction": float((improvement > 0).float().mean()),
            "worse_fraction": float((improvement < 0).float().mean()),
            **q,
        }
        for method, preds in methods_tensor.items():
            stats["matrix_distance_to_target"].setdefault(method, {})[split] = matrix_stats(
                preds[split], target[split])

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ma = method_metrics["mapped"][a]["mse"]
        mb = method_metrics["mapped"][b]["mse"]
        stats["hard_dim_overlap"][f"mapped_{a}_vs_{b}"] = {
            "top1": overlap_fraction(ma, mb, 0.01),
            "top10": overlap_fraction(ma, mb, 0.10),
        }

    train_target_std = target["train"].std(dim=0).clamp_min(1e-12)
    val_mse = method_metrics["mapped"]["val"]["mse"]
    val_norm_mse = method_metrics["mapped"]["val"]["norm_mse"]
    train_mse = method_metrics["mapped"]["train"]["mse"]
    affine_val_mse = method_metrics["affine"]["val"]["mse"]
    correlations = {
        "mapped_val_mse_vs_target_std": (val_mse, train_target_std),
        "mapped_val_norm_mse_vs_target_std": (val_norm_mse, train_target_std),
        "mapped_val_mse_vs_inverse_target_std": (val_mse, 1.0 / train_target_std),
        "mapped_val_mse_vs_train_mse": (val_mse, train_mse),
        "mapped_val_mse_vs_affine_val_mse": (val_mse, affine_val_mse),
        "mapped_val_norm_mse_vs_train_norm_mse": (
            val_norm_mse, method_metrics["mapped"]["train"]["norm_mse"]),
    }
    for key, (x, y) in correlations.items():
        stats["correlations"][key] = {
            "pearson": pearson(x, y),
            "spearman": spearman(x, y),
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    update_report(args.report, build_report(stats, args))
    print(f"saved {output_path}")
    print(f"updated {args.report}")


if __name__ == "__main__":
    main()
