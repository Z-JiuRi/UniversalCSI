# Codeword Cross-Architecture Summary

This report is generated from the per-split `summary.csv` and `pairwise_distances.csv` files.

## Split: `train`

Models: `42`

### Code Scale Ranking

| rank | model | std | l2_norm_mean | effective_rank_32 | top5_pca |
|---:|---|---:|---:|---:|---:|
| 1 | convnext_cnn_residual | 3.0147e+01 | 6.8099e+02 | 2.7838e+00 | 8.6773e-01 |
| 2 | resnet_cnn_residual | 2.1699e+01 | 4.9100e+02 | 2.9218e+01 | 2.8589e-01 |
| 3 | sparse_resnet_cnn_residual | 2.0830e+01 | 4.7133e+02 | 2.8824e+01 | 3.0196e-01 |
| 4 | cbam_cnn_cnn_residual | 1.6094e+01 | 3.6420e+02 | 2.7833e+01 | 3.3226e-01 |
| 5 | cnn_cnn_residual | 1.5211e+01 | 3.4420e+02 | 2.6948e+01 | 3.4996e-01 |
| 6 | transnet_cnn_residual | 1.2476e+01 | 2.8235e+02 | 2.6464e+01 | 3.4389e-01 |
| 7 | mlp_ae_cnn_residual | 1.2463e+01 | 2.8201e+02 | 1.4021e+01 | 5.5514e-01 |
| 8 | dscnn_cnn_residual | 1.2381e+01 | 2.8017e+02 | 2.7431e+01 | 3.3612e-01 |
| 9 | mlp_mixer_cnn_residual | 1.1964e+01 | 2.7073e+02 | 2.9146e+01 | 2.9905e-01 |
| 10 | crnet_cnn_residual | 1.0312e+01 | 2.3334e+02 | 2.9147e+01 | 2.9020e-01 |

### Highest Effective-Rank Code Spaces

| rank | model | effective_rank_32 | top10_pca | dim_std_mean |
|---:|---|---:|---:|---:|
| 1 | swin_cnn_residual | 2.9753e+01 | 4.6241e-01 | 3.2729e-01 |
| 2 | clnet_transnet | 2.9406e+01 | 4.7141e-01 | 5.9082e-01 |
| 3 | resnet_cnn_residual | 2.9218e+01 | 4.8753e-01 | 7.0615e-01 |
| 4 | crnet_cnn_residual | 2.9147e+01 | 4.8305e-01 | 4.7238e-01 |
| 5 | mlp_mixer_cnn_residual | 2.9146e+01 | 4.8939e-01 | 5.3541e-01 |
| 6 | crnet_transnet | 2.9112e+01 | 4.9073e-01 | 5.7129e-01 |
| 7 | clnet_hybrid | 2.9069e+01 | 4.8361e-01 | 7.4282e-01 |
| 8 | sparse_resnet_cnn_residual | 2.8824e+01 | 5.0072e-01 | 6.8500e-01 |
| 9 | mlp_mixer_transnet | 2.8675e+01 | 4.9968e-01 | 3.2441e-01 |
| 10 | swin_transnet | 2.8527e+01 | 4.9297e-01 | 3.2596e-01 |

### Decoder Effect Within Same Encoder

| encoder | transnet vs cnn_residual | transnet vs hybrid | cnn_residual vs hybrid |
|---|---:|---:|---:|
| attention_cnn | 1.5262e+02 | 1.2891e+01 | 1.5177e+02 |
| cbam_cnn | 3.6438e+02 | 3.2476e+01 | 3.6544e+02 |
| clnet | 2.2329e+02 | 1.9884e+01 | 2.2428e+02 |
| cnn | 3.4388e+02 | 3.3632e+01 | 3.4562e+02 |
| convnext | 6.8057e+02 | 3.7331e+01 | 6.8129e+02 |
| crnet | 2.3286e+02 | 1.2982e+01 | 2.3406e+02 |
| csinet | 2.1518e+02 | 1.9160e+01 | 2.1596e+02 |
| dscnn | 2.7994e+02 | 2.2061e+01 | 2.8071e+02 |
| mlp_ae | 2.8194e+02 | 4.5158e+00 | 2.8195e+02 |
| mlp_mixer | 2.7015e+02 | 1.4211e+01 | 2.6940e+02 |
| resnet | 4.9064e+02 | 3.9980e+01 | 4.9214e+02 |
| sparse_resnet | 4.7092e+02 | 3.9516e+01 | 4.7244e+02 |
| swin | 1.5791e+02 | 9.7839e+00 | 1.5816e+02 |
| transnet | 2.8182e+02 | 2.3242e+01 | 2.8283e+02 |

### Encoder Effect Within Same Decoder

| decoder | mean centroid_l2 | max centroid_l2 | max pair |
|---|---:|---:|---|
| cnn_residual | 4.6770e+02 | 8.7336e+02 | convnext_cnn_residual vs resnet_cnn_residual |
| hybrid | 3.3760e+01 | 5.8228e+01 | resnet_hybrid vs sparse_resnet_hybrid |
| transnet | 7.3311e+00 | 1.6400e+01 | convnext_transnet vs transnet_transnet |

### Closest Architecture Pairs

| rank | left | right | centroid_l2 | centroid_cosine | std_profile_l2 |
|---:|---|---|---:|---:|---:|
| 1 | mlp_ae_transnet | swin_transnet | 2.3884e+00 | 7.2592e-03 | 1.7750e+00 |
| 2 | attention_cnn_transnet | swin_transnet | 2.5198e+00 | -2.2523e-03 | 5.9390e+00 |
| 3 | attention_cnn_transnet | mlp_ae_transnet | 2.9371e+00 | -3.3813e-02 | 7.6133e+00 |
| 4 | clnet_transnet | swin_transnet | 3.0594e+00 | -6.3345e-02 | 6.0441e+00 |
| 5 | attention_cnn_transnet | clnet_transnet | 3.4131e+00 | -1.3084e-02 | 1.0959e+00 |
| 6 | clnet_transnet | mlp_ae_transnet | 3.4233e+00 | -7.7569e-02 | 7.7276e+00 |
| 7 | crnet_transnet | swin_transnet | 3.4281e+00 | 6.8565e-02 | 5.6213e+00 |
| 8 | crnet_transnet | mlp_ae_transnet | 3.7980e+00 | -4.5461e-03 | 7.2976e+00 |
| 9 | csinet_transnet | swin_transnet | 3.7990e+00 | 2.0677e-02 | 1.6115e+01 |
| 10 | attention_cnn_transnet | crnet_transnet | 3.8420e+00 | 1.4559e-02 | 1.1852e+00 |

### Farthest Architecture Pairs

| rank | left | right | centroid_l2 | centroid_cosine | std_profile_l2 |
|---:|---|---|---:|---:|---:|
| 1 | convnext_cnn_residual | resnet_cnn_residual | 8.7336e+02 | -8.8606e-02 | 4.1056e+01 |
| 2 | convnext_cnn_residual | sparse_resnet_cnn_residual | 8.3183e+02 | -1.1316e-02 | 4.1349e+01 |
| 3 | cbam_cnn_cnn_residual | convnext_cnn_residual | 7.7484e+02 | -1.0265e-02 | 4.2768e+01 |
| 4 | cnn_cnn_residual | convnext_cnn_residual | 7.6126e+02 | 3.4233e-03 | 4.2621e+01 |
| 5 | convnext_cnn_residual | dscnn_cnn_residual | 7.4498e+02 | -3.6286e-02 | 4.3441e+01 |
| 6 | convnext_cnn_residual | transnet_cnn_residual | 7.3957e+02 | -1.2195e-02 | 4.1996e+01 |
| 7 | convnext_cnn_residual | mlp_mixer_cnn_residual | 7.3684e+02 | -1.9035e-02 | 4.3189e+01 |
| 8 | convnext_cnn_residual | mlp_ae_cnn_residual | 7.3235e+02 | 1.5509e-02 | 4.5199e+01 |
| 9 | convnext_cnn_residual | csinet_cnn_residual | 7.1782e+02 | -2.1164e-02 | 4.4027e+01 |
| 10 | convnext_cnn_residual | crnet_cnn_residual | 7.1131e+02 | 3.5232e-02 | 4.3534e+01 |

