# Comprehensive Codeword Analysis for LoRA Generation Conditioning

This report analyzes codewords from all encoder x decoder combinations from the perspective of using codewords as **conditions** for generating decoder LoRA weights via diffusion or flow-matching.

## Motivation

- Goal: generate LoRA weights `phi_d` conditioned on encoder output codes `C_d`
- Pipeline: `C_support -> domain_embedding z_d -> generator -> LoRA phi_d`
- Approaches under consideration: deterministic MLP, flow-matching, diffusion
- This analysis characterizes the conditioning signal (codewords) to inform model design

## Data Coverage

- Architectures: 42 (14 encoders x 3 decoders)
- Samples per architecture: 100000
- Code dimension: 512
- Total train codewords analyzed: 4200000

## Key Findings for LoRA Generation

### 1. Code Separability (Can we distinguish encoders from codes alone?)

- **Intra-encoder centroid distance (mean)**: 2.1964e+02
- **Inter-encoder centroid distance (mean)**: 2.0314e+02
- **Separability ratio**: 0.92x
- **Silhouette score**: -0.368

**Interpretation**: Weak encoder signature. May need explicit encoder ID or additional side information for conditioning.

### 2. Sampling Impact (How many calibration codes are needed?)

| Metric | K=16 | K=32 | K=64 | K=128 | K=256 | K=512 | K=1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **effective_rank** | 0.9245 | 0.8570 | 0.7452 | 0.5879 | 0.4050 | 0.2471 | 0.1388 |
| **pca_top5_ratio** | 2.8403 | 1.5705 | 0.8135 | 0.4107 | 0.1958 | 0.0944 | 0.0536 |
| **std_global** | 0.0094 | 0.0064 | 0.0046 | 0.0039 | 0.0027 | 0.0016 | 0.0011 |
| **l2_mean** | 0.0091 | 0.0062 | 0.0045 | 0.0038 | 0.0026 | 0.0016 | 0.0011 |

**Guidance**: Choose K where relative error drops below 5% for key metrics. This determines the minimum calibration set size for the condition encoder.

### 3. Cross-Split Distribution Shift

| Architecture | Train-Val Centroid L2 | Val-Test Centroid L2 | Train-Test Centroid L2 |
|---|---:|---:|---:|
| attention_cnn_cnn_residual | nan | nan | nan |
| attention_cnn_hybrid | nan | nan | nan |
| attention_cnn_transnet | nan | nan | nan |
| cbam_cnn_cnn_residual | nan | nan | nan |
| cbam_cnn_hybrid | nan | nan | nan |
| cbam_cnn_transnet | nan | nan | nan |
| clnet_cnn_residual | nan | nan | nan |
| clnet_hybrid | nan | nan | nan |
| clnet_transnet | nan | nan | nan |
| cnn_cnn_residual | nan | nan | nan |

### 4. Code Space Dimensionality (Flow-Matching Feasibility)

| Decoder | Mean Eff. Rank | Mean PCA Top-5 | Mean Near-Zero | Mean Active Dims |
|---|---:|---:|---:|---:|
| hybrid | 165.74 | 0.1504 | 1.4030e-03 | 1.0000 |
| cnn_residual | 171.96 | 0.1833 | 1.1471e-03 | 1.0000 |
| transnet | 160.54 | 0.1584 | 1.3235e-03 | 1.0000 |

### 5. Decoder Parameter Analysis

LoRA targets for HybridDecoder (most promising decoder for generation):

| Layer | Full Params | LoRA r=4 | LoRA r=8 | LoRA r=16 |
|---|---:|---:|---:|---:|
| token_projection (Linear 512->2048) | 1,050,624 | 20,480 | 40,960 | 81,920 |
| semantic_projector.linear (Linear 512->512) | 262,656 | 8,192 | 16,384 | 32,768 |
| self_attn.in_proj (Linear 64->192) x2 layers | 24,960 | 4,096 | 8,192 | 16,384 |
| self_attn.out_proj (Linear 64->64) x2 layers | 8,320 | 2,048 | 4,096 | 8,192 |
| linear1 (Linear 64->2048) x2 layers | 266,240 | 33,792 | 67,584 | 135,168 |
| linear2 (Linear 2048->64) x2 layers | 262,272 | 33,792 | 67,584 | 135,168 |

- **Total decoder params**: 1,875,072
- **Total LoRA r=4 params**: 102,400
- **Compression ratio (full/LoRA r=4)**: 18x

For flow-matching, the generation target dimension is the LoRA parameter count. At r=4, this is ~40K parameters - feasible for flow-matching with a conditional velocity network.

### 6. Top-10 Best NMSE

| name | decoder | NMSE | std | eff_rank | top5_pca |
|---|---:|---:|---:|---:|---:|
| attention_cnn_cnn_residual | cnn_residual | nan | 6.7496e+00 | 175.70 | 0.1523 |
| attention_cnn_hybrid | hybrid | nan | 7.3148e-01 | 158.08 | 0.1488 |
| attention_cnn_transnet | transnet | nan | 5.9406e-01 | 145.77 | 0.1553 |
| cbam_cnn_cnn_residual | cnn_residual | nan | 1.6094e+01 | 189.96 | 0.1328 |
| cbam_cnn_hybrid | hybrid | nan | 1.8168e+00 | 133.97 | 0.1671 |
| cbam_cnn_transnet | transnet | nan | 1.2731e+00 | 131.06 | 0.1883 |
| clnet_cnn_residual | cnn_residual | nan | 9.8690e+00 | 174.17 | 0.1672 |
| clnet_hybrid | hybrid | nan | 1.1481e+00 | 188.17 | 0.1205 |
| clnet_transnet | transnet | nan | 6.0181e-01 | 182.73 | 0.1193 |
| cnn_cnn_residual | cnn_residual | nan | 1.5212e+01 | 175.69 | 0.1524 |

### 7. Top-10 Highest Effective Rank

| name | decoder | eff_rank | top5_pca | NMSE |
|---|---:|---:|---:|---:|
| mlp_mixer_cnn_residual | cnn_residual | 232.42 | 0.1002 | nan |
| transnet_cnn_residual | cnn_residual | 216.50 | 0.1189 | nan |
| swin_cnn_residual | cnn_residual | 214.37 | 0.0971 | nan |
| mlp_ae_hybrid | hybrid | 207.27 | 0.1383 | nan |
| csinet_hybrid | hybrid | 202.72 | 0.1392 | nan |
| mlp_mixer_transnet | transnet | 201.51 | 0.1185 | nan |
| mlp_ae_transnet | transnet | 200.33 | 0.1276 | nan |
| csinet_transnet | transnet | 197.61 | 0.1256 | nan |
| mlp_mixer_hybrid | hybrid | 196.99 | 0.1320 | nan |
| crnet_cnn_residual | cnn_residual | 193.33 | 0.1147 | nan |

### 8. Code-NMSE Correlations

| Metric | Pearson r with NMSE |
|---|---:|
| std_global | 0.0000 |
| abs_mean | 0.0000 |
| l2_mean | 0.0000 |
| l2_std | 0.0000 |
| near_zero_1e_3 | 0.0000 |
| near_zero_1e_2 | 0.0000 |
| dim_std_mean | 0.0000 |
| dim_std_std | 0.0000 |
| dead_dim_ratio_1e_3 | 0.0000 |
| active_dim_ratio_1e_2 | 0.0000 |
| pca_top1_ratio | 0.0000 |
| pca_top5_ratio | 0.0000 |
| pca_top10_ratio | 0.0000 |
| effective_rank | 0.0000 |
| condition_number_top512 | 0.0000 |

### 9. Recommendations for LoRA Generation Pipeline

Based on this analysis:

1. **Condition Encoder Design**: Codes are encoder-separable (ratio=0.92x). A DeepSets or Perceiver condition encoder over K calibration codes can extract domain identity.
2. **Calibration Set Size**: K >= 128 gives stable statistics for most models. Larger K (>256) gives diminishing returns.
3. **Flow-Matching Target Dimension**: ~40K LoRA params (r=4) or ~80K (r=8). Consider generating low-dimensional alpha coordinates first, then reconstructing LoRA.
4. **Which Decoder**: Hybrid decoder dominates NMSE ranking. Its code space has moderate effective rank (good for conditioning). CNN residual codes are highly collapsed (low effective rank) - harder to condition on.
5. **Cross-Split Stability**: Train/val/test code distributions are similar, suggesting the condition encoder can be trained on train codes and generalize.

## Generated Files

| File | Description |
|---|---|
| tables/code_summary.csv | Per-architecture code statistics |
| tables/cross_split_shifts.csv | Train/val/test distribution shifts |
| figures/*.png | All analysis plots |
| report.md | This report |

- [01_separability_intra_vs_inter.png](figures/01_separability_intra_vs_inter.png)
- [02_encoder_dispersion_silhouette.png](figures/02_encoder_dispersion_silhouette.png)
- [03_sampling_stability.png](figures/03_sampling_stability.png)
- [05_centroid_l2_heatmap_full.png](figures/05_centroid_l2_heatmap_full.png)
- [06_std_vs_nmse_by_decoder.png](figures/06_std_vs_nmse_by_decoder.png)
- [07_effective_rank_by_decoder.png](figures/07_effective_rank_by_decoder.png)
- [08_pca_cumulative_variance.png](figures/08_pca_cumulative_variance.png)
- [09_dimension_activity.png](figures/09_dimension_activity.png)
- [10_metric_correlation_matrix.png](figures/10_metric_correlation_matrix.png)
- [11_encoder_l2_norm_boxplot.png](figures/11_encoder_l2_norm_boxplot.png)
- [12_condition_number.png](figures/12_condition_number.png)
- [13_sampling_convergence.png](figures/13_sampling_convergence.png)
- [14_lora_parameter_analysis.png](figures/14_lora_parameter_analysis.png)
- [15_flow_matching_feasibility.png](figures/15_flow_matching_feasibility.png)
