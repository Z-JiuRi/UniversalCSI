# Comprehensive Codeword Analysis for LoRA Generation via Diffusion / Flow-Matching

## Objective

Analyze encoder output codewords as **conditions** for generating decoder LoRA weights.
The pipeline is: `C_support -> domain_embedding z_d -> generator -> LoRA phi_d`.

## Data Summary

- **Architectures**: 42 (14 encoders x 3 decoders)
- **Samples per model**: 100000
- **Code dimension**: 512
- **Total train codewords**: 4,200,000

---

## 1. Per-Decoder Encoder Separability

How well can we distinguish different encoders from their codewords, when the decoder is fixed? This is critical because in practice, the BS has a fixed decoder and needs to identify the UE encoder from calibration codes.

| Decoder | Encoders | Mean Inter-Enc Dist | Silhouette | Min Dist | Max Dist |
|---|---:|---:|---:|---:|---:|
| cnn_residual | 14 | 467.78 | nan | 210.42 | 873.79 |
| hybrid | 14 | 33.75 | nan | 10.43 | 58.21 |
| transnet | 14 | 7.33 | nan | 2.39 | 16.38 |

**Key insight**: The silhouette score indicates how clustered codes are by encoder. A high silhouette (>0.5) means codes from different encoders form distinct clusters - easy for a condition encoder. Low or negative silhouette means codes from different encoders overlap significantly - may need additional side information.

---

## 2. Sampling Convergence Analysis

How many calibration codes K are needed to reliably estimate code statistics? This determines the minimum support set size for the condition encoder.

Relative errors at K=128 (averaged across representative models):

| Metric | Rel. Error at K=128 | Stable at K=? |
|---|---:|---:|
| effective_rank | 0.5773 | >1024 |
| pca_top5_ratio | 0.3792 | 1024 |
| std_global | 0.0077 | 16 |
| l2_mean | 0.0074 | 16 |

**Recommendation**: K >= 128 for most metrics. K=256 for effective rank stability.

---

## 3. Code Distance vs NMSE Distance Correlation

If code distribution distance correlates with NMSE distance, then the condition encoder can learn a smooth mapping from code space to adapter space. This is a key indicator for flow-matching feasibility.

| Distance Type | Pearson r | Spearman rho |
|---|---:|---:|
| Code Centroid L2 vs NMSE diff | nan | nan |
| Code Covariance Frobenius vs NMSE diff | nan | nan |

**Interpretation**: Pearson > 0.5 suggests a strong linear relationship - simpler conditioning may suffice. Spearman > Pearson suggests monotonic but nonlinear relationship - flow-matching's flexibility becomes valuable.

---

## 4. Cross-Split Distribution Shift

Are train/val/test code distributions similar? This affects whether a condition encoder trained on training codes will generalize to deployment.

Average across all models:

| Split Pair | Mean Norm. Shift | Mean PCA Angle (deg) | Mean Cov Frobenius |
|---|---:|---:|---:|

**Interpretation**: Small normalized shift and PCA angle indicate distribution stability across splits - good for generalization.

---

## 5. Code Space Intrinsic Dimension

The intrinsic dimension of the code space affects flow-matching design:
- Low-dimensional codes -> simple condition encoder
- High-dimensional codes -> need deeper condition encoder
- Collapsed codes (participation ratio ~1) -> limited conditioning signal

| Decoder | Mean Participation Ratio | Mean PCs for 90% var | Mean Eff. Rank |
|---|---:|---:|---:|
| hybrid | 93.36 | 169.7 | 165.74 |
| cnn_residual | 91.42 | 189.0 | 171.96 |
| transnet | 89.69 | 168.2 | 160.54 |

---

## 6. LoRA Target Space for Flow-Matching

The generation target is LoRA parameters. Key question: how many parameters does the flow-matching velocity network need to output?

### HybridDecoder LoRA Targets

| Layer | Full Params | LoRA r=4 | LoRA r=8 | LoRA r=16 |
|---|---:|---:|---:|---:|
| hybrid (full) | 1,884,994 | 51,200 | 102,400 | 204,800 |
| hybrid (reduced) | - | 44,032 | 88,064 | 176,128 |

**Reduced**: only fc_projection + ffn1 + ffn2 (most impactful layers for domain adaptation).

---

## 7. Top-10 NMSE (Best Performing Architectures)

| Rank | Model | Decoder | NMSE | Eff Rank | Part. Ratio | PCs@90% |
|---:|---|---:|---:|---:|---:|---:|
| 1 | attention_cnn_cnn_residual | cnn_residual | nan | 175.70 | 82.94 | 185 |
| 2 | attention_cnn_hybrid | hybrid | nan | 158.08 | 91.73 | 157 |
| 3 | attention_cnn_transnet | transnet | nan | 145.77 | 83.59 | 144 |
| 4 | cbam_cnn_cnn_residual | cnn_residual | nan | 189.96 | 110.87 | 195 |
| 5 | cbam_cnn_hybrid | hybrid | nan | 133.97 | 76.57 | 134 |
| 6 | cbam_cnn_transnet | transnet | nan | 131.06 | 69.01 | 142 |
| 7 | clnet_cnn_residual | cnn_residual | nan | 174.17 | 70.97 | 195 |
| 8 | clnet_hybrid | hybrid | nan | 188.17 | 113.60 | 191 |
| 9 | clnet_transnet | transnet | nan | 182.73 | 111.31 | 180 |
| 10 | cnn_cnn_residual | cnn_residual | nan | 175.69 | 93.26 | 187 |

---

## 8. Metric-NMSE Correlations

| Metric | Pearson r with NMSE |
|---|---:|

---

## 9. Flow-Matching Feasibility Assessment

Summary of whether the current code data supports flow-matching conditioning:

**CHALLENGING**: Weak encoder separability in code space. Consider: (1) explicit encoder ID embedding, (2) deeper condition encoder (Perceiver/DeepSets), (3) code statistics as additional features, (4) per-decoder conditioning heads.

### Recommendations

1. **Condition Encoder**: Use DeepSets/Perceiver over K>=128 calibration codes
2. **Generation Target**: Start with reduced LoRA (fc_projection + ffn layers), ~44032 params at r=4
3. **Flow vs Diffusion**: Start with deterministic MLP generator as baseline. If code-NMSE distance correlation is strong, deterministic may suffice. If not, flow-matching provides the flexibility to model nonlinear condition-parameter mappings.
4. **Per-Decoder Strategy**: Train separate condition encoders per decoder type, as code distributions differ dramatically between hybrid/cnn_residual/transnet.
5. **Manifold Coordinate Approach**: Given the code space dimensionality, consider generating low-dimensional alpha coordinates first, then reconstructing LoRA.

---

## 10. Generated Files

| File | Description |
|---|---|
| tables/code_summary.csv | Per-architecture statistics |
| tables/sampling_convergence.csv | Sampling convergence data |
| tables/cross_split_shifts.csv | Distribution shift metrics |
| tables/per_decoder_separability.json | Separability metrics |
| figures/*.png | All analysis figures |
| report.md | This document |

- [01_per_decoder_separability.png](figures/01_per_decoder_separability.png)
- [02_sampling_convergence_all.png](figures/02_sampling_convergence_all.png)
- [03_code_vs_nmse_distance.png](figures/03_code_vs_nmse_distance.png)
- [05_intrinsic_dimension.png](figures/05_intrinsic_dimension.png)
- [06_lora_target_analysis.png](figures/06_lora_target_analysis.png)
- [07_dimensionality_vs_nmse.png](figures/07_dimensionality_vs_nmse.png)
- [08_centroid_heatmap_sorted.png](figures/08_centroid_heatmap_sorted.png)
- [09_condition_number_analysis.png](figures/09_condition_number_analysis.png)
- [10_encoder_pca_by_decoder.png](figures/10_encoder_pca_by_decoder.png)
- [11_flow_matching_feasibility_card.png](figures/11_flow_matching_feasibility_card.png)
- [12_same_encoder_decoder_variation.png](figures/12_same_encoder_decoder_variation.png)
