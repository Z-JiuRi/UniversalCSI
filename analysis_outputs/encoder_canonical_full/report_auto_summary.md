# Encoder Canonical 自动分析摘要

- 已发现实验数：134
- 已完成 final test：127
- 已有码字统计实验数：134

## 当前 NMSE Top 15

| 排名 | NMSE | family | scheme | 实验 |
|---:|---:|---|---|---|
| 1 | -29.889 | encoder | aux_pca_1e-3 | `aux_pca_1e-3/seed3407_transnet_transnet` |
| 2 | -29.726 | encoder | aux_pca_5e-3_code_mean1e-4 | `aux_pca_5e-3_code_mean1e-4/seed42_transnet_transnet` |
| 3 | -29.635 | encoder | aux_pca_5e-3_code_mean1e-4_cov1e-4 | `aux_pca_5e-3_code_mean1e-4_cov1e-4/seed42_transnet_transnet` |
| 4 | -29.620 | encoder | aux_pca_1e-3 | `aux_pca_1e-3/seed2026_transnet_transnet` |
| 5 | -29.579 | encoder | aux_pca_1e-2_code_mean1e-4_cov1e-4 | `aux_pca_1e-2_code_mean1e-4_cov1e-4/seed42_transnet_transnet` |
| 6 | -29.571 | encoder | aux_pca_1e-2_code_mean1e-4_cov1e-4 | `aux_pca_1e-2_code_mean1e-4_cov1e-4/seed3407_transnet_transnet` |
| 7 | -29.546 | encoder | aux_pca_1e-2_code_mean1e-4_cov1e-4 | `aux_pca_1e-2_code_mean1e-4_cov1e-4/seed2026_transnet_transnet` |
| 8 | -29.540 | adapter/aux_pca_1e-2_code_mean1e-4_cov1e-4/gated_lowrank_affine_mlp | aux_pca_1e-2_code_mean1e-4_cov1e-4 | `adapter/aux_pca_1e-2_code_mean1e-4_cov1e-4/gated_lowrank_affine_mlp/rank32_hidden2048_gate0.1_code1e-3_fc1e-2_lr5e-4_ep100/enc_seed3407_transnet_transnet_dec_seed42_transnet_transnet` |
| 9 | -29.539 | adapter/aux_pca_1e-2_code_mean1e-4_cov1e-4/gated_lowrank_affine_mlp | aux_pca_1e-2_code_mean1e-4_cov1e-4 | `adapter/aux_pca_1e-2_code_mean1e-4_cov1e-4/gated_lowrank_affine_mlp/rank32_hidden2048_gate0.1_code1e-3_fc1e-2_lr5e-4_ep400/enc_seed3407_transnet_transnet_dec_seed42_transnet_transnet` |
| 10 | -29.539 | adapter/aux_pca_1e-2_code_mean1e-4_cov1e-4/gated_lowrank_affine_mlp | aux_pca_1e-2_code_mean1e-4_cov1e-4 | `adapter/aux_pca_1e-2_code_mean1e-4_cov1e-4/gated_lowrank_affine_mlp/rank32_hidden2048_gate0.1_code1e-3_fc1e-2_lr5e-4_ep400/enc_seed3407_dec_seed42` |
| 11 | -29.528 | encoder | aux_pca_1e-3 | `aux_pca_1e-3/seed42_transnet_transnet` |
| 12 | -29.408 | encoder | aux_pca_5e-3_code_mean1e-4 | `aux_pca_5e-3_code_mean1e-4/seed3407_transnet_transnet` |
| 13 | -29.371 | encoder | aux_pca_5e-3_code_mean1e-4_cov1e-4 | `aux_pca_5e-3_code_mean1e-4_cov1e-4/seed3407_transnet_transnet` |
| 14 | -29.370 | encoder | aux_pca_1e-2 | `aux_pca_1e-2/seed3407_transnet_transnet` |
| 15 | -29.339 | encoder | aux_pca_1e-2 | `aux_pca_1e-2/seed42_transnet_transnet` |

## 主要图表

- `figures/scheme_nmse_ranking.png`
- `figures/encoder_nmse_heatmap.png`
- `figures/adapter_nmse_top.png`
- `figures/code_effective_rank_vs_nmse.png`
- `figures/code_cov_offdiag_vs_nmse.png`
- `figures/scheme_pairwise_code_cosine.png`

说明：该文件由脚本自动生成，实验未跑完时只代表当前已完成/已有日志的结果。