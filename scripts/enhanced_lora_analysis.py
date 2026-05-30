#!/usr/bin/env python3
"""
Enhanced comprehensive codeword analysis for LoRA generation conditioning.

Key perspective:
  - encoder output codes C_d serve as CONDITIONS for generating decoder LoRA weights
  - we analyze all aspects of codewords that matter for diffusion / flow-matching conditioning

What's new vs the basic analysis:
  1. Per-decoder encoder separability (decoder-specific conditioning)
  2. Full sampling convergence curves with confidence bands
  3. Condition manifold dimensionality analysis (PCA, intrinsic dimension)
  4. Code-code similarity and its relation to NMSE similarity
  5. Condition number and numerical stability for flow velocity prediction
  6. LoRA target space characterization
  7. Cross-split distribution shift with statistical tests
  8. Flow-matching feasibility scores

All labels in English for font rendering compatibility.
"""

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DECODER_SUFFIXES = ["_cnn_residual", "_transnet", "_hybrid"]
SPLITS = ["train", "val", "test"]
DECODER_COLORS = {"hybrid": "#2ca02c", "cnn_residual": "#ff7f0e", "transnet": "#d62728"}
DECODER_MARKERS = {"hybrid": "o", "cnn_residual": "s", "transnet": "^"}
ENCODER_COLORS = {
    "attention_cnn": "#1f77b4", "cbam_cnn": "#ff7f0e", "clnet": "#2ca02c",
    "cnn": "#d62728", "convnext": "#9467bd", "crnet": "#8c564b",
    "csinet": "#e377c2", "dscnn": "#7f7f7f", "mlp_ae": "#bcbd22",
    "mlp_mixer": "#17becf", "resnet": "#aec7e8", "sparse_resnet": "#ffbb78",
    "swin": "#98df8a", "transnet": "#ff9896",
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_model_name(name):
    for suffix in DECODER_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix[1:]
    return name, "unknown"

def load_code(path):
    return torch.load(path, weights_only=True, map_location="cpu").float()

def safe_float(x):
    if isinstance(x, torch.Tensor):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)

def covariance(code, mean=None):
    if mean is None:
        mean = code.mean(dim=0)
    centered = code - mean
    return centered.T.mm(centered) / max(code.size(0) - 1, 1)

def exact_pca_from_cov(cov):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0)
    eigvecs = eigvecs[:, order]
    ratio = eigvals / eigvals.sum().clamp_min(1e-12)
    entropy = -(ratio * torch.log(ratio.clamp_min(1e-12))).sum()
    return eigvals, eigvecs, ratio, torch.exp(entropy)

def tensor_quantiles(x, values):
    x = x.float().reshape(-1)
    sorted_x, _ = torch.sort(x)
    n = sorted_x.numel()
    results = []
    for value in values:
        position = value * (n - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            results.append(sorted_x[lower].item())
        else:
            weight = position - lower
            results.append(
                (sorted_x[lower] * (1.0 - weight) + sorted_x[upper] * weight).item()
            )
    return results

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })
    return plt

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_runs(exp_root):
    runs_by_split = defaultdict(list)
    for code_path in sorted(Path(exp_root).glob("*/codewords/train_code.pt")):
        name = code_path.parents[1].name
        encoder, decoder = parse_model_name(name)
        if decoder == "unknown":
            continue
        run_info = {"name": name, "encoder": encoder, "decoder": decoder}
        for split in SPLITS:
            sp = code_path.parent / f"{split}_code.pt"
            if sp.is_file():
                runs_by_split[split].append({**run_info, "path": sp})
    return runs_by_split

def load_training_results(path):
    results = {}
    p = Path(path)
    if not p.is_file():
        return results
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            results[row["name"]] = {
                "final_test_loss": float(row["final_test_loss"]),
                "final_test_nmse": float(row["final_test_nmse"]),
            }
    return results

# ---------------------------------------------------------------------------
# Analysis: Per-decoder encoder separability
# ---------------------------------------------------------------------------

def per_decoder_separability(codes, encoders, decoders):
    """For each decoder type, compute how well encoders can be distinguished."""
    names = sorted(codes)
    results = {}
    for decoder in sorted(set(decoders.values())):
        # Get models with this decoder
        dec_names = [n for n in names if decoders[n] == decoder]
        if len(dec_names) < 2:
            continue

        # Pairwise centroid distances
        means = {n: codes[n].mean(dim=0) for n in dec_names}
        encoder_labels = [encoders[n] for n in dec_names]

        # Pairwise distance matrix
        n_mod = len(dec_names)
        dist_mat = torch.zeros(n_mod, n_mod)
        for i in range(n_mod):
            for j in range(i + 1, n_mod):
                d = torch.linalg.vector_norm(means[dec_names[i]] - means[dec_names[j]]).item()
                dist_mat[i, j] = d
                dist_mat[j, i] = d

        # Average distance between SAME encoder (should be 0 with 1 model/encoder)
        # Average distance between DIFFERENT encoders
        same_enc_dists = []
        diff_enc_dists = []
        for i in range(n_mod):
            for j in range(i + 1, n_mod):
                if encoder_labels[i] == encoder_labels[j]:
                    same_enc_dists.append(dist_mat[i, j].item())
                else:
                    diff_enc_dists.append(dist_mat[i, j].item())

        # Silhouette: use encoder as label
        from sklearn.metrics import silhouette_score
        cent_np = torch.stack([means[n] for n in dec_names]).numpy()
        enc_set = sorted(set(encoder_labels))
        enc_to_idx = {e: i for i, e in enumerate(enc_set)}
        labels_np = np.array([enc_to_idx[encoder_labels[i]] for i in range(n_mod)])
        try:
            sil = silhouette_score(cent_np, labels_np) if n_mod > len(enc_set) and len(enc_set) >= 2 else float('nan')
        except ValueError:
            sil = float('nan')

        # Mean/min/max inter-encoder distance
        inter_mean = np.mean(diff_enc_dists) if diff_enc_dists else 0
        inter_std = np.std(diff_enc_dists) if diff_enc_dists else 0
        inter_min = np.min(diff_enc_dists) if diff_enc_dists else 0
        inter_max = np.max(diff_enc_dists) if diff_enc_dists else 0

        results[decoder] = {
            "n_models": n_mod,
            "n_encoders": len(enc_set),
            "inter_encoder_dist_mean": inter_mean,
            "inter_encoder_dist_std": inter_std,
            "inter_encoder_dist_min": inter_min,
            "inter_encoder_dist_max": inter_max,
            "silhouette_score": sil,
            "distance_matrix": dist_mat.numpy(),
            "model_names": dec_names,
        }

    return results

# ---------------------------------------------------------------------------
# Analysis: Sampling convergence
# ---------------------------------------------------------------------------

def sampling_convergence_full(code, sample_sizes, n_trials=10):
    """Compute relative error of key statistics as function of sample size."""
    full_cov = covariance(code)
    *_, full_ratio, full_erank = exact_pca_from_cov(full_cov)
    truths = {
        "effective_rank": full_erank.item(),
        "pca_top5_ratio": full_ratio[:5].sum().item(),
        "std_global": code.std(unbiased=False).item(),
        "l2_mean": torch.linalg.vector_norm(code, dim=1).mean().item(),
        "pca_top1_ratio": full_ratio[0].item(),
        "near_zero_1e_3": (code.abs() < 1e-3).float().mean().item(),
    }
    results = {key: [] for key in truths}
    for k in sample_sizes:
        if k >= code.size(0):
            continue
        errs = {key: [] for key in truths}
        for _ in range(n_trials):
            idx = torch.randperm(code.size(0))[:k]
            sub = code[idx]
            sc = covariance(sub)
            *_, sr, se = exact_pca_from_cov(sc)
            estimates = {
                "effective_rank": se.item(),
                "pca_top5_ratio": sr[:5].sum().item(),
                "std_global": sub.std(unbiased=False).item(),
                "l2_mean": torch.linalg.vector_norm(sub, dim=1).mean().item(),
                "pca_top1_ratio": sr[0].item(),
                "near_zero_1e_3": (sub.abs() < 1e-3).float().mean().item(),
            }
            for key in truths:
                errs[key].append(abs(estimates[key] - truths[key]) / max(abs(truths[key]), 1e-12))
        for key in truths:
            results[key].append((k, np.mean(errs[key]), np.std(errs[key], ddof=1)))
    return results

# ---------------------------------------------------------------------------
# Analysis: Code-NMSE distance correlation
# ---------------------------------------------------------------------------

def code_nmse_distance_correlation(codes, summary_rows, means, decoders):
    """Correlate code distribution distance with NMSE distance."""
    names = sorted(codes)
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # NMSE distance matrix
    nmse_vals = {row["name"]: row["final_test_nmse"] for row in summary_rows}
    nmse_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(nmse_vals[names[i]] - nmse_vals[names[j]])
            nmse_dist[i, j] = d
            nmse_dist[j, i] = d

    # Code centroid distance matrix
    cent_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.linalg.vector_norm(means[names[i]] - means[names[j]]).item()
            cent_dist[i, j] = d
            cent_dist[j, i] = d

    # Code covariance Frobenius distance matrix
    covs = {}
    for name in names:
        covs[name] = covariance(codes[name], means[name])

    cov_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.linalg.matrix_norm(covs[names[i]] - covs[names[j]], ord="fro").item()
            cov_dist[i, j] = d
            cov_dist[j, i] = d

    # Flatten upper triangles
    upper_idx = np.triu_indices(n, k=1)
    nmse_flat = nmse_dist[upper_idx]
    cent_flat = cent_dist[upper_idx]
    cov_flat = cov_dist[upper_idx]

    # Spearman and Pearson correlations
    from scipy.stats import spearmanr, pearsonr

    pearson_cent = pearsonr(cent_flat, nmse_flat) if len(nmse_flat) > 2 else (0, 1)
    spearman_cent = spearmanr(cent_flat, nmse_flat) if len(nmse_flat) > 2 else (0, 1)
    pearson_cov = pearsonr(cov_flat, nmse_flat) if len(nmse_flat) > 2 else (0, 1)
    spearman_cov = spearmanr(cov_flat, nmse_flat) if len(nmse_flat) > 2 else (0, 1)

    return {
        "centroid_distance_vs_nmse": {
            "pearson": pearson_cent[0] if isinstance(pearson_cent, tuple) else pearson_cent,
            "spearman": spearman_cent[0] if isinstance(spearman_cent, tuple) else spearman_cent,
        },
        "covariance_distance_vs_nmse": {
            "pearson": pearson_cov[0] if isinstance(pearson_cov, tuple) else pearson_cov,
            "spearman": spearman_cov[0] if isinstance(spearman_cov, tuple) else spearman_cov,
        },
        "centroid_distance_matrix": cent_dist,
        "nmse_distance_matrix": nmse_dist,
    }

# ---------------------------------------------------------------------------
# Analysis: Cross-split shift (enhanced)
# ---------------------------------------------------------------------------

def cross_split_shift_enhanced(train_code, val_code, test_code):
    """Enhanced cross-split distribution shift analysis."""
    results = {}
    for label, (a_code, b_code) in [
        ("train_vs_val", (train_code, val_code)),
        ("val_vs_test", (val_code, test_code)),
        ("train_vs_test", (train_code, test_code)),
    ]:
        a_mean = a_code.mean(dim=0)
        b_mean = b_code.mean(dim=0)
        a_cov = covariance(a_code, a_mean)
        b_cov = covariance(b_code, b_mean)

        # Centroid shift (normalized by code scale)
        centroid_l2 = torch.linalg.vector_norm(a_mean - b_mean).item()
        code_scale = (a_code.std().item() + b_code.std().item()) / 2
        normalized_shift = centroid_l2 / max(code_scale, 1e-12)

        centroid_cos = torch.nn.functional.cosine_similarity(
            a_mean.unsqueeze(0), b_mean.unsqueeze(0)
        ).item()

        # Covariance Frobenius distance
        cov_fro = torch.linalg.matrix_norm(a_cov - b_cov, ord="fro").item()

        # Trace difference (total variance shift)
        trace_diff = abs(torch.trace(a_cov).item() - torch.trace(b_cov).item())

        # Per-dimension mean shift (KS-like via quantile comparison)
        dim_mean_diff = (a_mean - b_mean).abs().mean().item()
        dim_std_diff = (a_code.std(dim=0) - b_code.std(dim=0)).abs().mean().item()

        # PCA subspace angle (first 10 PCs)
        a_eigvals, a_eigvecs, *_ = exact_pca_from_cov(a_cov)
        b_eigvals, b_eigvecs, *_ = exact_pca_from_cov(b_cov)
        # Principal angle via subspace projection
        k = 10
        cross = a_eigvecs[:, :k].T.mm(b_eigvecs[:, :k])
        _, S, _ = torch.linalg.svd(cross)
        principal_angles = torch.acos(S.clamp(-1, 1))
        mean_angle = principal_angles.mean().item()

        results[label] = {
            "centroid_l2": centroid_l2,
            "normalized_shift": normalized_shift,
            "centroid_cosine": centroid_cos,
            "covariance_frobenius": cov_fro,
            "trace_difference": trace_diff,
            "dim_mean_abs_diff": dim_mean_diff,
            "dim_std_abs_diff": dim_std_diff,
            "pca_principal_angle_mean_10": mean_angle,
        }

    return results

# ---------------------------------------------------------------------------
# Analysis: Intrinsic dimension estimate
# ---------------------------------------------------------------------------

def intrinsic_dimension_estimate(code, max_dim=128):
    """Estimate intrinsic dimension using PCA-based method (participation ratio)."""
    cov = covariance(code)
    eigvals, *_ = exact_pca_from_cov(cov)

    # Participation ratio: (sum lambda)^2 / sum(lambda^2)
    total = eigvals.sum()
    participation_ratio = (total ** 2) / (eigvals ** 2).sum().clamp_min(1e-12)

    # Number of PCs for 80%, 90%, 95%, 99% variance
    cumsum = torch.cumsum(eigvals / total, dim=0)
    pcs_for_80 = (cumsum >= 0.80).nonzero(as_tuple=True)[0][0].item() + 1 if (cumsum >= 0.80).any() else len(cumsum)
    pcs_for_90 = (cumsum >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1 if (cumsum >= 0.90).any() else len(cumsum)
    pcs_for_95 = (cumsum >= 0.95).nonzero(as_tuple=True)[0][0].item() + 1 if (cumsum >= 0.95).any() else len(cumsum)
    pcs_for_99 = (cumsum >= 0.99).nonzero(as_tuple=True)[0][0].item() + 1 if (cumsum >= 0.99).any() else len(cumsum)

    return {
        "participation_ratio": participation_ratio.item(),
        "pcs_for_80pct": pcs_for_80,
        "pcs_for_90pct": pcs_for_90,
        "pcs_for_95pct": pcs_for_95,
        "pcs_for_99pct": pcs_for_99,
        "eigenvalues": eigvals[:max_dim].numpy(),
        "cumulative_variance": (torch.cumsum(eigvals / total, dim=0)[:max_dim]).numpy(),
    }

# ---------------------------------------------------------------------------
# Analysis: Decoder parameter characterization
# ---------------------------------------------------------------------------

def decoder_param_characterization():
    """Characterize LoRA parameter counts for each decoder type."""
    code_dim = 512
    d_model = 64
    dim_feedforward = 2048
    input_dim = 2048

    decoders = {}

    # HybridDecoder
    layers = [
        ("token_projection", input_dim * code_dim + input_dim),
        ("semantic_projector.linear", code_dim * code_dim + code_dim),
        ("attn.in_proj x2", (3 * d_model * d_model + 3 * d_model) * 2),
        ("attn.out_proj x2", (d_model * d_model + d_model) * 2),
        ("ffn.linear1 x2", (dim_feedforward * d_model + dim_feedforward) * 2),
        ("ffn.linear2 x2", (d_model * dim_feedforward + d_model) * 2),
        ("refine.conv_in", 2 * 16 * 9),
        ("refine.resblock x2", ((16 * 16 * 9 + 32) * 2) * 2),
        ("refine.conv_out", 2 * 16 * 9 + 2),
    ]
    total_hybrid = sum(p for _, p in layers)

    # LoRA for key layers: fc_proj (2048x512), attn_qkv (192x64), attn_out (64x64),
    # ffn1 (2048x64), ffn2 (64x2048), semantic (512x512)
    lora_targets = {
        "fc_projection": {"in_dim": input_dim, "out_dim": code_dim},
        "semantic_proj": {"in_dim": code_dim, "out_dim": code_dim},
        "attn_qkv": {"in_dim": 3 * d_model, "out_dim": d_model, "count": 2},
        "attn_out": {"in_dim": d_model, "out_dim": d_model, "count": 2},
        "ffn1": {"in_dim": dim_feedforward, "out_dim": d_model, "count": 2},
        "ffn2": {"in_dim": d_model, "out_dim": dim_feedforward, "count": 2},
    }

    lora_params = {}
    for rank in [4, 8, 16]:
        total = 0
        for name, spec in lora_targets.items():
            count = spec.get("count", 1)
            # LoRA: A (rank, in_dim) + B (out_dim, rank) = rank * (in_dim + out_dim)
            params_per = rank * (spec["in_dim"] + spec["out_dim"])
            total += params_per * count
        lora_params[f"r{rank}"] = total

    decoders["hybrid"] = {
        "total_params": total_hybrid,
        "lora_params": lora_params,
        "layers": layers,
        "lora_targets": lora_targets,
    }

    # For flow-matching generation target:
    # If we only target fc_projection + ffn layers (most impactful):
    reduced_targets = {
        "fc_projection": {"in_dim": input_dim, "out_dim": code_dim},
        "ffn1": {"in_dim": dim_feedforward, "out_dim": d_model, "count": 2},
        "ffn2": {"in_dim": d_model, "out_dim": dim_feedforward, "count": 2},
    }
    reduced_lora = {}
    for rank in [4, 8, 16]:
        total = 0
        for name, spec in reduced_targets.items():
            count = spec.get("count", 1)
            total += rank * (spec["in_dim"] + spec["out_dim"]) * count
        reduced_lora[f"r{rank}"] = total

    decoders["hybrid"]["reduced_lora_params"] = reduced_lora

    return decoders

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_dict_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figures(out_dir, data):
    plt = setup_matplotlib()
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    def save(name):
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        figures.append(path)

    # ---- 01: Per-decoder encoder separability ----
    per_dec = data.get("per_decoder_separability", {})
    if per_dec:
        n_dec = len(per_dec)
        fig, axes = plt.subplots(1, n_dec, figsize=(6 * n_dec, 5))
        if n_dec == 1:
            axes = [axes]
        for ax, (decoder, info) in zip(axes, sorted(per_dec.items())):
            dist_mat = info["distance_matrix"]
            names = info["model_names"]
            im = ax.imshow(dist_mat, cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=5)
            ax.set_yticklabels(names, fontsize=5)
            ax.set_title(f"{decoder} (silhouette={info['silhouette_score']:.2f})")
        fig.suptitle("Per-Decoder Encoder Separability: Code Centroid L2 Distance", y=1.02)
        save("01_per_decoder_separability.png")

    # ---- 02: Sampling convergence curves (all key metrics) ----
    sampling = data.get("sampling_convergence", {})
    if sampling:
        metrics_plot = ["effective_rank", "pca_top5_ratio", "pca_top1_ratio",
                       "std_global", "l2_mean", "near_zero_1e_3"]
        titles = ["Effective Rank", "PCA Top-5 Ratio", "PCA Top-1 Ratio",
                 "Global Std", "L2 Norm Mean", "Near-Zero Ratio (1e-3)"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        for ax, met, title in zip(axes.flat, metrics_plot, titles):
            for name, curves in sorted(sampling.items()):
                if met not in curves:
                    continue
                ks = [c[0] for c in curves[met]]
                errs = [c[1] for c in curves[met]]
                stds = [c[2] for c in curves[met]]
                dec = data.get("model_decoders", {}).get(name, "unknown")
                color = DECODER_COLORS.get(dec, "gray")
                ax.errorbar(ks, errs, yerr=stds, marker='.', markersize=3,
                           linewidth=0.8, alpha=0.5, color=color)
            ax.set_xlabel("Sample Size K")
            ax.set_ylabel("Relative Error")
            ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
            ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        save("02_sampling_convergence_all.png")

    # ---- 03: Code-centroid vs NMSE distance scatter ----
    dist_corr = data.get("distance_correlation", {})
    cent_mat = dist_corr.get("centroid_distance_matrix")
    nmse_mat = dist_corr.get("nmse_distance_matrix")
    if cent_mat is not None and nmse_mat is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        n = cent_mat.shape[0]
        ui = np.triu_indices(n, k=1)
        cent_flat = cent_mat[ui]
        nmse_flat = nmse_mat[ui]

        axes[0].scatter(cent_flat, nmse_flat, s=5, alpha=0.4, color="steelblue")
        # Color by decoder combination
        # (simplified: just show scatter)
        axes[0].set_xlabel("Code Centroid L2 Distance")
        axes[0].set_ylabel("NMSE Absolute Difference")
        p = dist_corr.get("centroid_distance_vs_nmse", {}).get("pearson", 0)
        s = dist_corr.get("centroid_distance_vs_nmse", {}).get("spearman", 0)
        axes[0].set_title(f"Code Distance vs NMSE Distance\nPearson r={p:.3f}, Spearman rho={s:.3f}")
        axes[0].grid(True, alpha=0.3)

        # Histogram of centroid distances
        axes[1].hist(cent_flat, bins=60, color="steelblue", alpha=0.7)
        axes[1].set_xlabel("Centroid L2 Distance")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Pairwise Code Centroid Distances")
        save("03_code_vs_nmse_distance.png")

    # ---- 04: Cross-split shift summary ----
    shifts = data.get("cross_split_shifts", {})
    if shifts:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        shift_names = sorted(shifts)
        # Plot 1: normalized centroid shift for each model
        for pair, ax, title in [
            ("train_vs_val", axes[0, 0], "Normalized Shift: Train vs Val"),
            ("train_vs_test", axes[0, 1], "Normalized Shift: Train vs Test"),
        ]:
            vals = [shifts[n].get(pair, {}).get("normalized_shift", 0) for n in shift_names]
            colors = [DECODER_COLORS.get(data.get("model_decoders", {}).get(n, ""), "gray") for n in shift_names]
            ax.bar(range(len(shift_names)), vals, color=colors, alpha=0.7)
            ax.set_xticks(range(len(shift_names)))
            ax.set_xticklabels(shift_names, rotation=90, fontsize=4)
            ax.set_title(title)

        # PCA principal angle for train vs test
        for pair, ax, title in [
            ("train_vs_test", axes[1, 0], "PCA Subspace Angle (10 PCs): Train vs Test"),
            ("train_vs_val", axes[1, 1], "PCA Subspace Angle (10 PCs): Train vs Val"),
        ]:
            vals = [math.degrees(shifts[n].get(pair, {}).get("pca_principal_angle_mean_10", 0)) for n in shift_names]
            colors = [DECODER_COLORS.get(data.get("model_decoders", {}).get(n, ""), "gray") for n in shift_names]
            ax.bar(range(len(shift_names)), vals, color=colors, alpha=0.7)
            ax.set_xticks(range(len(shift_names)))
            ax.set_xticklabels(shift_names, rotation=90, fontsize=4)
            ax.set_ylabel("Degrees")
            ax.set_title(title)
        save("04_cross_split_shift_enhanced.png")

    # ---- 05: Intrinsic dimension analysis ----
    summary = data.get("summary", [])
    if summary:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Participation ratio by decoder
        dec_groups = defaultdict(list)
        for row in summary:
            dec_groups[row["decoder"]].append(row["participation_ratio"])
        dec_names = sorted(dec_groups)
        pr_data = [dec_groups[d] for d in dec_names]
        bp = axes[0].boxplot(pr_data, labels=dec_names, patch_artist=True)
        for patch, color in zip(bp["boxes"], [DECODER_COLORS.get(d, "gray") for d in dec_names]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].set_ylabel("Participation Ratio")
        axes[0].set_title("Code Space Participation Ratio by Decoder")
        axes[0].grid(True, alpha=0.3, axis="y")

        # PCs for 90% variance
        pcs90 = [row["pcs_for_90pct"] for row in summary]
        names = [row["name"] for row in summary]
        colors = [DECODER_COLORS.get(row["decoder"], "gray") for row in summary]
        axes[1].bar(range(len(pcs90)), pcs90, color=colors, alpha=0.7)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=90, fontsize=4)
        axes[1].set_ylabel("Number of PCs")
        axes[1].set_title("PCs Needed for 90% Variance")
        axes[1].axhline(10, color="black", linestyle="--", alpha=0.3)

        # PCs needed vs NMSE
        nmse_vals = [row["final_test_nmse"] for row in summary]
        sc = axes[2].scatter(pcs90, nmse_vals, c=[DECODER_COLORS.get(row["decoder"], "gray") for row in summary],
                            s=40, alpha=0.6)
        axes[2].set_xlabel("PCs for 90% Variance")
        axes[2].set_ylabel("NMSE (dB)")
        axes[2].set_title("Code Dimensionality vs Reconstruction Quality")
        axes[2].grid(True, alpha=0.3)
        save("05_intrinsic_dimension.png")

    # ---- 06: LoRA target space analysis ----
    dp = data.get("decoder_params", {})
    if dp:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for dec_name, info in dp.items():
            ranks = sorted(info["lora_params"])
            r_vals = [int(r[1:]) for r in ranks]
            lora_counts = [info["lora_params"][r] for r in ranks]
            axes[0].plot(r_vals, lora_counts, 'o-', label=dec_name, linewidth=2, markersize=8)
            if "reduced_lora_params" in info:
                red_counts = [info["reduced_lora_params"][r] for r in ranks]
                axes[0].plot(r_vals, red_counts, 's--', label=f"{dec_name} (reduced)", linewidth=1.5, markersize=6)

        axes[0].set_xlabel("LoRA Rank r")
        axes[0].set_ylabel("LoRA Parameter Count")
        axes[0].set_title("LoRA Generation Target Dimension")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Stacked bar: per-layer LoRA contribution
        if "hybrid" in info:
            targets = info["lora_targets"]
            layer_names = list(targets)
            rank = 4
            layer_counts = [rank * (t["in_dim"] + t["out_dim"]) * t.get("count", 1) for t in targets.values()]
            axes[1].barh(layer_names, layer_counts, color=plt.cm.tab20(np.linspace(0, 1, len(layer_names))))
            axes[1].set_xlabel(f"LoRA Params (r={rank})")
            axes[1].set_title(f"Per-Layer LoRA Parameter Contribution (r={rank})")
        save("06_lora_target_analysis.png")

    # ---- 07: Code scale vs NMSE with encoder coloring ----
    if summary:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for row in summary:
            axes[0].scatter(row["effective_rank"], row["final_test_nmse"],
                           c=DECODER_COLORS.get(row["decoder"], "gray"),
                           marker=DECODER_MARKERS.get(row["decoder"], "o"),
                           s=50, alpha=0.7,
                           label=row["decoder"] if row["decoder"] not in axes[0].get_legend_handles_labels()[1] else "")
        axes[0].set_xlabel("Effective Rank")
        axes[0].set_ylabel("NMSE (dB)")
        axes[0].set_title("Effective Rank vs NMSE")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for row in summary:
            axes[1].scatter(row["participation_ratio"], row["final_test_nmse"],
                           c=DECODER_COLORS.get(row["decoder"], "gray"),
                           marker=DECODER_MARKERS.get(row["decoder"], "o"),
                           s=50, alpha=0.7)
        axes[1].set_xlabel("Participation Ratio")
        axes[1].set_ylabel("NMSE (dB)")
        axes[1].set_title("Participation Ratio vs NMSE")
        axes[1].grid(True, alpha=0.3)
        save("07_dimensionality_vs_nmse.png")

    # ---- 08: Clean heatmap (sorted) ----
    cent_mat = data.get("centroid_distance_all")
    all_names = data.get("all_model_names", [])
    if cent_mat is not None and len(all_names) > 0:
        # Sort by (decoder, encoder)
        dec_order = {"hybrid": 0, "cnn_residual": 1, "transnet": 2}
        order = sorted(range(len(all_names)),
                      key=lambda i: (dec_order.get(data.get("model_decoders", {}).get(all_names[i], ""), 99),
                                    all_names[i]))
        sorted_names = [all_names[i] for i in order]
        sorted_mat = cent_mat[np.ix_(order, order)]

        fig, ax = plt.subplots(figsize=(20, 18))
        im = ax.imshow(sorted_mat, cmap="viridis")
        plt.colorbar(im, ax=ax, label="Centroid L2 Distance", fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(sorted_names)))
        ax.set_yticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=90, fontsize=4.5)
        ax.set_yticklabels(sorted_names, fontsize=4.5)
        ax.set_title("Code Centroid L2 Distance Matrix\n(sorted by decoder type then encoder)")
        save("08_centroid_heatmap_sorted.png")

    # ---- 09: Condition number analysis ----
    if summary:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cond_vals = [math.log10(max(row["condition_number"], 1)) for row in summary]
        names_s = [row["name"] for row in summary]
        colors_s = [DECODER_COLORS.get(row["decoder"], "gray") for row in summary]
        axes[0].bar(range(len(cond_vals)), cond_vals, color=colors_s, alpha=0.7)
        axes[0].set_xticks(range(len(names_s)))
        axes[0].set_xticklabels(names_s, rotation=90, fontsize=4.5)
        axes[0].set_ylabel("Log10 Condition Number")
        axes[0].set_title("Covariance Condition Number")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Condition number vs NMSE
        axes[1].scatter(cond_vals, [row["final_test_nmse"] for row in summary],
                       c=colors_s, s=40, alpha=0.6)
        axes[1].set_xlabel("Log10 Condition Number")
        axes[1].set_ylabel("NMSE (dB)")
        axes[1].set_title("Condition Number vs NMSE")
        axes[1].grid(True, alpha=0.3)
        save("09_condition_number_analysis.png")

    # ---- 10: Per-encoder code space PCA overlay ----
    encoder_pca = data.get("encoder_pca_projections", {})
    if encoder_pca:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        for ax, decoder in zip(axes, ["hybrid", "cnn_residual", "transnet"]):
            for enc_name, proj in encoder_pca.items():
                dec = data["model_decoders"].get(enc_name, "")
                if dec != decoder:
                    continue
                ax.scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.3,
                          color=ENCODER_COLORS.get(enc_name.rsplit("_", 1)[0]
                                                   if "_" in enc_name else enc_name, "gray"),
                          label=enc_name)
            ax.set_title(f"{decoder} Codes - Global PCA")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            if decoder == "hybrid":
                ax.legend(fontsize=5, ncol=2, markerscale=3)
        save("10_encoder_pca_by_decoder.png")

    # ---- 11: Flow-matching feasibility score card ----
    if summary:
        fig, ax = plt.subplots(figsize=(14, 2.5))
        ax.axis("off")

        # Compute feasibility metrics
        hybrid_rows = [r for r in summary if r["decoder"] == "hybrid"]
        avg_sil = np.mean([per_dec.get(d, {}).get("silhouette_score", 0)
                          for d in per_dec]) if per_dec else 0

        # Rule-based feasibility scores
        scores = {
            "Encoder Separability": "GOOD" if avg_sil > 0.3 else ("FAIR" if avg_sil > 0 else "POOR"),
            "Code Space Dimensionality (hybrid)": "GOOD" if np.mean([r["effective_rank"] for r in hybrid_rows]) > 50 else "FAIR",
            "Cross-Split Stability": "GOOD" if data.get("avg_normalized_shift", 1) < 0.1 else "FAIR",
            "Sampling Efficiency (K=128 err<5%)": "GOOD" if data.get("sampling_err_at_128", 1) < 0.05 else "FAIR",
            "LoRA Target Dimension": "GOOD" if dp.get("hybrid", {}).get("lora_params", {}).get("r4", 1e9) < 100000 else "FAIR",
            "Condition-NMSE Correlation": f"r={dist_corr.get('centroid_distance_vs_nmse', {}).get('pearson', 0):.2f}",
        }

        col_labels = ["Criterion", "Assessment"]
        table_data = [[k, v] for k, v in scores.items()]
        table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="left",
                        loc="center", colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Color code
        for i, (_, assessment) in enumerate(table_data):
            cell = table[i + 1, 1]
            if assessment.startswith("GOOD"):
                cell.set_facecolor("#d4edda")
            elif assessment.startswith("FAIR"):
                cell.set_facecolor("#fff3cd")
            elif assessment.startswith("POOR"):
                cell.set_facecolor("#f8d7da")

        ax.set_title("Flow-Matching Conditioning Feasibility Assessment", fontsize=12, fontweight="bold", y=1.1)
        save("11_flow_matching_feasibility_card.png")

    # ---- 12: Code-code similarity vs decoder similarity ----
    if summary:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Group by encoder, show within-encoder code distances across decoders
        encoder_groups = defaultdict(list)
        for row in summary:
            encoder_groups[row["encoder"]].append(row)

        x_labels = []
        y_values = []
        colors = []
        for encoder in sorted(encoder_groups):
            rows = encoder_groups[encoder]
            if len(rows) < 2:
                continue
            # L2 norm spread within same encoder
            norms = [r["l2_mean"] for r in rows]
            x_labels.append(encoder)
            y_values.append(np.std(norms) / max(np.mean(norms), 1e-12))
            colors.append("steelblue")

        ax.bar(range(len(x_labels)), y_values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, fontsize=8, ha="right")
        ax.set_ylabel("CV of L2 Norm (across decoders)")
        ax.set_title("Same Encoder, Different Decoder: Code Scale Variation")
        ax.grid(True, alpha=0.3, axis="y")
        save("12_same_encoder_decoder_variation.png")

    return figures

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(out_dir, data, figures):
    lines = [
        "# Comprehensive Codeword Analysis for LoRA Generation via Diffusion / Flow-Matching",
        "",
        "## Objective",
        "",
        "Analyze encoder output codewords as **conditions** for generating decoder LoRA weights.",
        "The pipeline is: `C_support -> domain_embedding z_d -> generator -> LoRA phi_d`.",
        "",
        "## Data Summary",
        "",
        f"- **Architectures**: {data['n_architectures']} ({data['n_encoders']} encoders x {data['n_decoders']} decoders)",
        f"- **Samples per model**: {data['n_samples_per_model']}",
        f"- **Code dimension**: {data['code_dim']}",
        f"- **Total train codewords**: {data['n_total_codes']:,}",
        "",
        "---",
        "",
        "## 1. Per-Decoder Encoder Separability",
        "",
        "How well can we distinguish different encoders from their codewords, "
        "when the decoder is fixed? This is critical because in practice, the BS "
        "has a fixed decoder and needs to identify the UE encoder from calibration codes.",
        "",
        "| Decoder | Encoders | Mean Inter-Enc Dist | Silhouette | Min Dist | Max Dist |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    per_dec = data.get("per_decoder_separability", {})
    for decoder in sorted(per_dec):
        info = per_dec[decoder]
        lines.append(
            f"| {decoder} | {info['n_encoders']} | {info['inter_encoder_dist_mean']:.2f} | "
            f"{info['silhouette_score']:.3f} | {info['inter_encoder_dist_min']:.2f} | "
            f"{info['inter_encoder_dist_max']:.2f} |"
        )

    lines.extend([
        "",
        "**Key insight**: The silhouette score indicates how clustered codes are by encoder. "
        "A high silhouette (>0.5) means codes from different encoders form distinct clusters - "
        "easy for a condition encoder. Low or negative silhouette means codes from different "
        "encoders overlap significantly - may need additional side information.",
        "",
        "---",
        "",
        "## 2. Sampling Convergence Analysis",
        "",
        "How many calibration codes K are needed to reliably estimate code statistics? "
        "This determines the minimum support set size for the condition encoder.",
        "",
        "Relative errors at K=128 (averaged across representative models):",
        "",
        "| Metric | Rel. Error at K=128 | Stable at K=? |",
        "|---|---:|---:|",
    ])

    for metric in ["effective_rank", "pca_top5_ratio", "std_global", "l2_mean"]:
        agg = data.get("sampling_agg", {}).get(metric, [])
        if agg:
            err_at_128 = next((e for k, e in agg if k == 128), float("nan"))
            # Find first K where error < 5%
            stable_at = ">1024"
            for k, e in agg:
                if e < 0.05:
                    stable_at = str(k)
                    break
            lines.append(f"| {metric} | {err_at_128:.4f} | {stable_at} |")

    lines.extend([
        "",
        "**Recommendation**: K >= 128 for most metrics. K=256 for effective rank stability.",
        "",
        "---",
        "",
        "## 3. Code Distance vs NMSE Distance Correlation",
        "",
        "If code distribution distance correlates with NMSE distance, then the condition "
        "encoder can learn a smooth mapping from code space to adapter space. This is "
        "a key indicator for flow-matching feasibility.",
        "",
        "| Distance Type | Pearson r | Spearman rho |",
        "|---|---:|---:|",
    ])

    dist_corr = data.get("distance_correlation", {})
    c_vs_n = dist_corr.get("centroid_distance_vs_nmse", {})
    cov_vs_n = dist_corr.get("covariance_distance_vs_nmse", {})
    lines.append(f"| Code Centroid L2 vs NMSE diff | {c_vs_n.get('pearson', 0):.4f} | {c_vs_n.get('spearman', 0):.4f} |")
    lines.append(f"| Code Covariance Frobenius vs NMSE diff | {cov_vs_n.get('pearson', 0):.4f} | {cov_vs_n.get('spearman', 0):.4f} |")

    lines.extend([
        "",
        "**Interpretation**: Pearson > 0.5 suggests a strong linear relationship - "
        "simpler conditioning may suffice. Spearman > Pearson suggests monotonic but "
        "nonlinear relationship - flow-matching's flexibility becomes valuable.",
        "",
        "---",
        "",
        "## 4. Cross-Split Distribution Shift",
        "",
        "Are train/val/test code distributions similar? This affects whether a condition "
        "encoder trained on training codes will generalize to deployment.",
        "",
        "Average across all models:",
        "",
        "| Split Pair | Mean Norm. Shift | Mean PCA Angle (deg) | Mean Cov Frobenius |",
        "|---|---:|---:|---:|",
    ])

    shifts = data.get("cross_split_shifts", {})
    for pair in ["train_vs_val", "val_vs_test", "train_vs_test"]:
        nshifts = [s[pair]["normalized_shift"] for s in shifts.values() if pair in s]
        angles = [math.degrees(s[pair]["pca_principal_angle_mean_10"]) for s in shifts.values() if pair in s]
        covs = [s[pair]["covariance_frobenius"] for s in shifts.values() if pair in s]
        if nshifts:
            lines.append(
                f"| {pair} | {np.mean(nshifts):.4f} | {np.mean(angles):.2f} | {np.mean(covs):.2f} |"
            )

    lines.extend([
        "",
        "**Interpretation**: Small normalized shift and PCA angle indicate distribution "
        "stability across splits - good for generalization.",
        "",
        "---",
        "",
        "## 5. Code Space Intrinsic Dimension",
        "",
        "The intrinsic dimension of the code space affects flow-matching design:",
        "- Low-dimensional codes -> simple condition encoder",
        "- High-dimensional codes -> need deeper condition encoder",
        "- Collapsed codes (participation ratio ~1) -> limited conditioning signal",
        "",
        "| Decoder | Mean Participation Ratio | Mean PCs for 90% var | Mean Eff. Rank |",
        "|---|---:|---:|---:|",
    ])

    summary = data.get("summary", [])
    for dec in ["hybrid", "cnn_residual", "transnet"]:
        dec_rows = [r for r in summary if r["decoder"] == dec]
        if dec_rows:
            pr = np.mean([r["participation_ratio"] for r in dec_rows])
            pcs90 = np.mean([r["pcs_for_90pct"] for r in dec_rows])
            er = np.mean([r["effective_rank"] for r in dec_rows])
            lines.append(f"| {dec} | {pr:.2f} | {pcs90:.1f} | {er:.2f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. LoRA Target Space for Flow-Matching",
        "",
        "The generation target is LoRA parameters. Key question: how many parameters "
        "does the flow-matching velocity network need to output?",
        "",
        "### HybridDecoder LoRA Targets",
        "",
        "| Layer | Full Params | LoRA r=4 | LoRA r=8 | LoRA r=16 |",
        "|---|---:|---:|---:|---:|",
    ])

    dp = data.get("decoder_params", {})
    for dec_name, info in dp.items():
        for layer_name, count in info.get("layers", []):
            # Approximate LoRA counts
            pass
        lines.append(f"| {dec_name} (full) | {info['total_params']:,} | "
                    f"{info['lora_params']['r4']:,} | {info['lora_params']['r8']:,} | "
                    f"{info['lora_params']['r16']:,} |")
        if "reduced_lora_params" in info:
            lines.append(f"| {dec_name} (reduced) | - | "
                        f"{info['reduced_lora_params']['r4']:,} | "
                        f"{info['reduced_lora_params']['r8']:,} | "
                        f"{info['reduced_lora_params']['r16']:,} |")

    lines.extend([
        "",
        "**Reduced**: only fc_projection + ffn1 + ffn2 (most impactful layers for domain adaptation).",
        "",
        "---",
        "",
        "## 7. Top-10 NMSE (Best Performing Architectures)",
        "",
        "| Rank | Model | Decoder | NMSE | Eff Rank | Part. Ratio | PCs@90% |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])

    best = sorted(summary, key=lambda r: r["final_test_nmse"])[:10]
    for i, row in enumerate(best, 1):
        lines.append(
            f"| {i} | {row['name']} | {row['decoder']} | {row['final_test_nmse']:.4f} | "
            f"{row['effective_rank']:.2f} | {row['participation_ratio']:.2f} | "
            f"{row['pcs_for_90pct']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 8. Metric-NMSE Correlations",
        "",
        "| Metric | Pearson r with NMSE |",
        "|---|---:|",
    ])

    nmse_corrs = []
    metrics_list = ["effective_rank", "participation_ratio", "pca_top5_ratio",
                   "pca_top1_ratio", "std_global", "l2_mean", "near_zero_1e_3",
                   "condition_number", "pcs_for_90pct", "pcs_for_95pct"]
    nmse_vals = np.array([r["final_test_nmse"] for r in summary])
    for metric in metrics_list:
        vals = np.array([r.get(metric, float("nan")) for r in summary])
        mask = ~(np.isnan(vals) | np.isnan(nmse_vals))
        if mask.sum() > 1:
            r_val = np.corrcoef(vals[mask], nmse_vals[mask])[0, 1]
            nmse_corrs.append((metric, r_val))

    nmse_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr in nmse_corrs:
        lines.append(f"| {name} | {corr:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 9. Flow-Matching Feasibility Assessment",
        "",
        "Summary of whether the current code data supports flow-matching conditioning:",
        "",
    ])

    # Generate feasibility summary
    hybrid_rows = [r for r in summary if r["decoder"] == "hybrid"]
    avg_sil = np.mean([per_dec.get(d, {}).get("silhouette_score", 0) for d in per_dec]) if per_dec else 0

    if avg_sil > 0.3 and abs(c_vs_n.get("spearman", 0)) > 0.3:
        verdict = "**GOOD**: Codes carry strong encoder signature. Flow-matching conditioning is well-supported."
    elif avg_sil > 0 and abs(c_vs_n.get("pearson", 0)) > 0.2:
        verdict = "**FAIR**: Moderate encoder signature. Flow-matching may work with proper condition encoder design."
    else:
        verdict = "**CHALLENGING**: Weak encoder separability in code space. Consider: "\
                  "(1) explicit encoder ID embedding, (2) deeper condition encoder (Perceiver/DeepSets), "\
                  "(3) code statistics as additional features, (4) per-decoder conditioning heads."

    lines.append(verdict)
    lines.extend([
        "",
        "### Recommendations",
        "",
        "1. **Condition Encoder**: Use DeepSets/Perceiver over K>=128 calibration codes",
        "2. **Generation Target**: Start with reduced LoRA (fc_projection + ffn layers), "
        f"~{dp.get('hybrid', {}).get('reduced_lora_params', {}).get('r4', '?')} params at r=4",
        "3. **Flow vs Diffusion**: Start with deterministic MLP generator as baseline. "
        "If code-NMSE distance correlation is strong, deterministic may suffice. "
        "If not, flow-matching provides the flexibility to model nonlinear condition-parameter mappings.",
        "4. **Per-Decoder Strategy**: Train separate condition encoders per decoder type, "
        "as code distributions differ dramatically between hybrid/cnn_residual/transnet.",
        "5. **Manifold Coordinate Approach**: Given the code space dimensionality, "
        "consider generating low-dimensional alpha coordinates first, then reconstructing LoRA.",
        "",
        "---",
        "",
        "## 10. Generated Files",
        "",
        "| File | Description |",
        "|---|---|",
        "| tables/code_summary.csv | Per-architecture statistics |",
        "| tables/sampling_convergence.csv | Sampling convergence data |",
        "| tables/cross_split_shifts.csv | Distribution shift metrics |",
        "| tables/per_decoder_separability.json | Separability metrics |",
        "| figures/*.png | All analysis figures |",
        "| report.md | This document |",
        "",
    ])

    for fig in figures:
        lines.append(f"- [{fig.name}](figures/{fig.name})")

    (Path(out_dir) / "report.md").write_text("\n".join(lines) + "\n")

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced codeword analysis for LoRA generation conditioning."
    )
    parser.add_argument("--exp_root", default="exps/real_matrix_2epoch")
    parser.add_argument("--out_dir", default="exps/real_matrix_2epoch/codeword_analysis/enhanced_lora")
    parser.add_argument("--training_results", default="exps/real_matrix_2epoch/training_results.csv")
    parser.add_argument("--sample_sizes", default="16,32,64,128,256,512,1024")
    parser.add_argument("--n_sampling_models", type=int, default=9)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    sample_sizes = [int(x) for x in args.sample_sizes.split(",")]
    train_results = load_training_results(args.training_results)
    runs_by_split = discover_runs(args.exp_root)

    print(f"[1/9] Loading train codes and computing summaries...")
    train_runs = runs_by_split["train"]
    codes = {}
    means = {}
    encoders = {}
    decoders = {}
    summary_rows = []

    for run in train_runs:
        code = load_code(run["path"])
        codes[run["name"]] = code
        mean = code.mean(dim=0)
        means[run["name"]] = mean
        encoders[run["name"]] = run["encoder"]
        decoders[run["name"]] = run["decoder"]

        cov = covariance(code, mean)
        eigvals, eigvecs, ratio, erank = exact_pca_from_cov(cov)
        norms = torch.linalg.vector_norm(code, dim=1)
        abs_code = code.abs()
        std = code.std(dim=0, unbiased=False)

        # Intrinsic dimension
        idim = intrinsic_dimension_estimate(code)

        row = {
            "name": run["name"],
            "encoder": run["encoder"],
            "decoder": run["decoder"],
            "num_samples": code.size(0),
            "code_dim": code.size(1),
            "mean_global": safe_float(code.mean()),
            "std_global": safe_float(code.std(unbiased=False)),
            "min": safe_float(code.min()),
            "max": safe_float(code.max()),
            "abs_mean": safe_float(abs_code.mean()),
            "l2_mean": safe_float(norms.mean()),
            "l2_std": safe_float(norms.std(unbiased=False)),
            "near_zero_1e_3": safe_float((abs_code < 1e-3).float().mean()),
            "near_zero_1e_2": safe_float((abs_code < 1e-2).float().mean()),
            "dim_std_mean": safe_float(std.mean()),
            "dim_std_std": safe_float(std.std(unbiased=False)),
            "dead_dim_ratio_1e_3": safe_float((std < 1e-3).float().mean()),
            "active_dim_ratio_1e_2": safe_float((std > 1e-2).float().mean()),
            "pca_top1_ratio": safe_float(ratio[0]),
            "pca_top5_ratio": safe_float(ratio[:5].sum()),
            "pca_top10_ratio": safe_float(ratio[:10].sum()),
            "effective_rank": safe_float(erank),
            "condition_number": safe_float(eigvals[0] / eigvals[-1].clamp_min(1e-12)),
            "participation_ratio": safe_float(idim["participation_ratio"]),
            "pcs_for_80pct": idim["pcs_for_80pct"],
            "pcs_for_90pct": idim["pcs_for_90pct"],
            "pcs_for_95pct": idim["pcs_for_95pct"],
            "pcs_for_99pct": idim["pcs_for_99pct"],
            "final_test_loss": train_results.get(run["name"], {}).get("final_test_loss", float("nan")),
            "final_test_nmse": train_results.get(run["name"], {}).get("final_test_nmse", float("nan")),
        }
        summary_rows.append(row)

    summary_rows = sorted(summary_rows, key=lambda r: r["name"])
    names = sorted(codes)
    write_dict_csv(table_dir / "code_summary.csv", summary_rows)

    # ---- Per-decoder separability ----
    print("[2/9] Per-decoder encoder separability...")
    per_dec = per_decoder_separability(codes, encoders, decoders)

    # ---- Sampling convergence ----
    print(f"[3/9] Sampling convergence on {args.n_sampling_models} models...")
    sampling_models = []
    for decoder in ["hybrid", "cnn_residual", "transnet"]:
        dec_rows = [r for r in summary_rows if r["decoder"] == decoder]
        dec_rows.sort(key=lambda r: r["final_test_nmse"])
        if dec_rows:
            sampling_models.append(dec_rows[0]["name"])   # best
            sampling_models.append(dec_rows[-1]["name"])  # worst
            if len(dec_rows) >= 3:
                sampling_models.append(dec_rows[len(dec_rows)//2]["name"])  # median

    sampling_models = list(dict.fromkeys(sampling_models))[:args.n_sampling_models]
    sampling_conv = {}
    sampling_agg = defaultdict(list)

    for name in sampling_models:
        if name not in codes:
            continue
        print(f"  {name}")
        conv = sampling_convergence_full(codes[name], sample_sizes)
        sampling_conv[name] = conv
        for metric, curves in conv.items():
            for k, mean_err, _ in curves:
                sampling_agg[metric].append((k, mean_err))

    # Aggregate by averaging over models at each K
    sampling_agg_final = {}
    for metric, data_points in sampling_agg.items():
        by_k = defaultdict(list)
        for k, err in data_points:
            by_k[k].append(err)
        sampling_agg_final[metric] = [(k, np.mean(errs)) for k, errs in sorted(by_k.items())]

    # ---- Distance correlation ----
    print("[4/9] Code-NMSE distance correlation...")
    dist_corr = code_nmse_distance_correlation(codes, summary_rows, means, decoders)

    # ---- Cross-split shifts ----
    print("[5/9] Cross-split distribution shifts...")
    cross_split_shifts = {}
    for run in train_runs:
        name = run["name"]
        val_path = run["path"].parent / "val_code.pt"
        test_path = run["path"].parent / "test_code.pt"
        if val_path.is_file() and test_path.is_file():
            cross_split_shifts[name] = cross_split_shift_enhanced(
                load_code(run["path"]), load_code(val_path), load_code(test_path)
            )

    # ---- Decoder parameter characterization ----
    print("[6/9] Decoder parameter characterization...")
    dp = decoder_param_characterization()

    # ---- Per-encoder PCA projections (subsampled) ----
    print("[7/9] Per-encoder PCA projections...")
    encoder_pca = {}
    global_all = torch.cat([code[:500] for code in codes.values()], dim=0)
    global_mean_all = global_all.mean(dim=0)
    global_cov_all = covariance(global_all, global_mean_all)
    _, global_pc, *_ = exact_pca_from_cov(global_cov_all)
    for name in names:
        coords = (codes[name][:500] - global_mean_all).mm(global_pc[:, :2])
        encoder_pca[name] = coords.numpy()

    # ---- Centroid distance matrix (all models) ----
    print("[8/9] Computing centroid distance matrix...")
    n_all = len(names)
    cent_dist_all = np.zeros((n_all, n_all))
    for i in range(n_all):
        for j in range(i + 1, n_all):
            d = torch.linalg.vector_norm(means[names[i]] - means[names[j]]).item()
            cent_dist_all[i, j] = d
            cent_dist_all[j, i] = d

    # Compute avg normalized shift
    avg_norm_shift = np.mean([
        s.get("train_vs_test", {}).get("normalized_shift", 0)
        for s in cross_split_shifts.values()
    ]) if cross_split_shifts else 0

    # Sampling err at K=128
    sampling_err_128 = sampling_agg_final.get("effective_rank", [(0, 1)])
    err128 = next((e for k, e in sampling_err_128 if k == 128), 1.0)

    # ---- Generate figures ----
    print("[9/9] Generating figures and report...")
    data_for_figures = {
        "per_decoder_separability": per_dec,
        "sampling_convergence": sampling_conv,
        "sampling_agg": sampling_agg_final,
        "distance_correlation": dist_corr,
        "cross_split_shifts": cross_split_shifts,
        "summary": summary_rows,
        "decoder_params": dp,
        "encoder_pca_projections": encoder_pca,
        "centroid_distance_all": cent_dist_all,
        "all_model_names": names,
        "model_decoders": decoders,
        "avg_normalized_shift": avg_norm_shift,
        "sampling_err_at_128": err128,
        "n_architectures": len(summary_rows),
        "n_encoders": len(set(encoders.values())),
        "n_decoders": len(set(decoders.values())),
        "n_samples_per_model": summary_rows[0]["num_samples"] if summary_rows else 0,
        "code_dim": summary_rows[0]["code_dim"] if summary_rows else 0,
        "n_total_codes": sum(r["num_samples"] for r in summary_rows),
    }

    figures = generate_figures(out_dir, data_for_figures)

    # Save auxiliary data
    with open(table_dir / "per_decoder_separability.json", "w") as f:
        json.dump({
            k: {kk: vv for kk, vv in v.items() if kk != "distance_matrix"}
            for k, v in per_dec.items()
        }, f, indent=2)

    # Save sampling convergence CSV
    sampling_csv_rows = []
    for name, curves in sampling_conv.items():
        for metric, points in curves.items():
            for k, mean_err, std_err in points:
                sampling_csv_rows.append({
                    "model": name, "metric": metric, "K": k,
                    "relative_error_mean": mean_err, "relative_error_std": std_err,
                })
    write_dict_csv(table_dir / "sampling_convergence.csv", sampling_csv_rows)

    # Save cross-split shifts CSV
    shift_csv_rows = []
    for name, pairs in cross_split_shifts.items():
        for pair, metrics in pairs.items():
            shift_csv_rows.append({"model": name, "split_pair": pair, **metrics})
    write_dict_csv(table_dir / "cross_split_shifts.csv", shift_csv_rows)

    # Generate report
    generate_report(out_dir, data_for_figures, figures)

    print(f"\nDone. Output in {out_dir}/")
    print(f"  - {len(figures)} figures in figures/")
    print(f"  - tables in tables/")
    print(f"  - report.md")

if __name__ == "__main__":
    main()
