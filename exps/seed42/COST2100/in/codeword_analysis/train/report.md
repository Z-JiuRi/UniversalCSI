# Codeword Architecture Analysis

Split: `train`
Runs analyzed: `42`

## What Was Measured

- Global code distribution: mean, std, min/max, absolute mean.
- Per-sample code norm: average and spread of L2 norms.
- Sparsity proxy: fraction of values with `abs(code) < 1e-3`.
- Dimension usage: mean per-dimension std and its spread.
- PCA concentration: top-1/top-5/top-10 variance ratios and effective rank.
- Pairwise architecture distance: centroid L2/cosine and std-profile L2/cosine.

## Summary Table

| model | encoder | decoder | samples | dim | mean | std | l2_norm | near_zero | top5_pca | eff_rank32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attention_cnn_cnn_residual | attention_cnn | cnn_residual | 10000 | 512 | 9.3403e-02 | 6.7496e+00 | 1.5273e+02 | 1.8621e-03 | 3.7467e-01 | 2.5145e+01 |
| attention_cnn_hybrid | attention_cnn | hybrid | 10000 | 512 | -7.6676e-02 | 7.3174e-01 | 1.6592e+01 | 1.6998e-03 | 3.1490e-01 | 2.8353e+01 |
| attention_cnn_transnet | attention_cnn | transnet | 10000 | 512 | 1.8042e-03 | 5.9375e-01 | 1.3161e+01 | 1.4057e-03 | 3.1916e-01 | 2.8110e+01 |
| cbam_cnn_cnn_residual | cbam_cnn | cnn_residual | 10000 | 512 | -2.5090e-01 | 1.6094e+01 | 3.6420e+02 | 1.0848e-03 | 3.3226e-01 | 2.7833e+01 |
| cbam_cnn_hybrid | cbam_cnn | hybrid | 10000 | 512 | 1.5328e-01 | 1.8160e+00 | 4.1153e+01 | 7.2109e-04 | 3.2361e-01 | 2.7777e+01 |
| cbam_cnn_transnet | cbam_cnn | transnet | 10000 | 512 | 1.1029e-02 | 1.2718e+00 | 2.8346e+01 | 6.6602e-04 | 3.5198e-01 | 2.6655e+01 |
| clnet_cnn_residual | clnet | cnn_residual | 10000 | 512 | -3.0238e-01 | 9.8693e+00 | 2.2341e+02 | 1.0248e-03 | 4.0308e-01 | 2.3623e+01 |
| clnet_hybrid | clnet | hybrid | 10000 | 512 | 2.4165e-02 | 1.1482e+00 | 2.5875e+01 | 1.1258e-03 | 2.9480e-01 | 2.9069e+01 |
| clnet_transnet | clnet | transnet | 10000 | 512 | 2.3439e-03 | 6.0309e-01 | 1.3348e+01 | 1.4254e-03 | 2.8059e-01 | 2.9406e+01 |
| cnn_cnn_residual | cnn | cnn_residual | 10000 | 512 | -1.3960e-01 | 1.5211e+01 | 3.4420e+02 | 1.0652e-03 | 3.4996e-01 | 2.6948e+01 |
| cnn_hybrid | cnn | hybrid | 10000 | 512 | -2.1783e-01 | 1.8933e+00 | 4.3013e+01 | 6.9238e-04 | 3.3100e-01 | 2.7584e+01 |
| cnn_transnet | cnn | transnet | 10000 | 512 | -1.0824e-02 | 1.0964e+00 | 2.4338e+01 | 7.8320e-04 | 3.2974e-01 | 2.7607e+01 |
| convnext_cnn_residual | convnext | cnn_residual | 10000 | 512 | -2.5780e-01 | 3.0147e+01 | 6.8099e+02 | 4.8320e-04 | 8.6773e-01 | 2.7838e+00 |
| convnext_hybrid | convnext | hybrid | 10000 | 512 | 2.2738e-03 | 2.5452e+00 | 5.7414e+01 | 3.8906e-04 | 3.2953e-01 | 2.8177e+01 |
| convnext_transnet | convnext | transnet | 10000 | 512 | 1.2428e-02 | 1.4076e+00 | 3.1712e+01 | 6.2500e-04 | 3.2717e-01 | 2.8166e+01 |
| crnet_cnn_residual | crnet | cnn_residual | 10000 | 512 | -2.0940e-02 | 1.0312e+01 | 2.3334e+02 | 9.5664e-04 | 2.9020e-01 | 2.9147e+01 |
| crnet_hybrid | crnet | hybrid | 10000 | 512 | -4.8172e-03 | 8.8065e-01 | 1.9788e+01 | 1.2271e-03 | 3.3824e-01 | 2.7447e+01 |
| crnet_transnet | crnet | transnet | 10000 | 512 | -4.9613e-04 | 5.8995e-01 | 1.3080e+01 | 1.4527e-03 | 2.9341e-01 | 2.9112e+01 |
| csinet_cnn_residual | csinet | cnn_residual | 10000 | 512 | 8.6394e-02 | 9.5157e+00 | 2.1529e+02 | 9.7988e-04 | 4.9610e-01 | 1.7714e+01 |
| csinet_hybrid | csinet | hybrid | 10000 | 512 | -4.3342e-02 | 1.1977e+00 | 2.6907e+01 | 1.0066e-03 | 3.5695e-01 | 2.6545e+01 |
| csinet_transnet | csinet | transnet | 10000 | 512 | -5.0216e-03 | 1.0492e+00 | 2.3133e+01 | 8.0371e-04 | 3.1725e-01 | 2.8303e+01 |
| dscnn_cnn_residual | dscnn | cnn_residual | 10000 | 512 | -2.1086e-01 | 1.2381e+01 | 2.8017e+02 | 1.2799e-03 | 3.3612e-01 | 2.7431e+01 |
| dscnn_hybrid | dscnn | hybrid | 10000 | 512 | -1.2997e-01 | 1.4731e+00 | 3.3304e+01 | 7.3750e-04 | 3.3865e-01 | 2.7405e+01 |
| dscnn_transnet | dscnn | transnet | 10000 | 512 | -1.2351e-02 | 1.3140e+00 | 2.9303e+01 | 6.4199e-04 | 3.5389e-01 | 2.7042e+01 |
| mlp_ae_cnn_residual | mlp_ae | cnn_residual | 10000 | 512 | -1.6571e-01 | 1.2463e+01 | 2.8201e+02 | 1.4895e-03 | 5.5514e-01 | 1.4021e+01 |
| mlp_ae_hybrid | mlp_ae | hybrid | 10000 | 512 | -3.3577e-02 | 2.2215e-01 | 5.0666e+00 | 5.8275e-03 | 3.5499e-01 | 2.7221e+01 |
| mlp_ae_transnet | mlp_ae | transnet | 10000 | 512 | -4.6649e-04 | 2.6590e-01 | 5.9030e+00 | 3.2678e-03 | 3.2098e-01 | 2.8219e+01 |
| mlp_mixer_cnn_residual | mlp_mixer | cnn_residual | 10000 | 512 | -1.8324e-02 | 1.1964e+01 | 2.7073e+02 | 1.0684e-03 | 2.9905e-01 | 2.9146e+01 |
| mlp_mixer_hybrid | mlp_mixer | hybrid | 10000 | 512 | 3.3927e-02 | 7.3694e-01 | 1.6642e+01 | 1.9180e-03 | 3.2961e-01 | 2.8114e+01 |
| mlp_mixer_transnet | mlp_mixer | transnet | 10000 | 512 | 3.6964e-03 | 3.6400e-01 | 8.1040e+00 | 2.3219e-03 | 3.0826e-01 | 2.8675e+01 |
| resnet_cnn_residual | resnet | cnn_residual | 10000 | 512 | 1.4130e-01 | 2.1699e+01 | 4.9100e+02 | 8.4121e-04 | 2.8589e-01 | 2.9218e+01 |
| resnet_hybrid | resnet | hybrid | 10000 | 512 | 9.7700e-02 | 2.2917e+00 | 5.1781e+01 | 5.5664e-04 | 3.3098e-01 | 2.7627e+01 |
| resnet_transnet | resnet | transnet | 10000 | 512 | 1.7958e-03 | 1.0210e+00 | 2.2631e+01 | 8.4375e-04 | 3.7898e-01 | 2.5683e+01 |
| sparse_resnet_cnn_residual | sparse_resnet | cnn_residual | 10000 | 512 | 8.6013e-02 | 2.0830e+01 | 4.7133e+02 | 9.4883e-04 | 3.0196e-01 | 2.8824e+01 |
| sparse_resnet_hybrid | sparse_resnet | hybrid | 10000 | 512 | -1.2827e-01 | 2.2350e+00 | 5.0535e+01 | 5.8242e-04 | 3.4197e-01 | 2.7311e+01 |
| sparse_resnet_transnet | sparse_resnet | transnet | 10000 | 512 | -1.4018e-02 | 1.1145e+00 | 2.4763e+01 | 7.5020e-04 | 3.8166e-01 | 2.5606e+01 |
| swin_cnn_residual | swin | cnn_residual | 10000 | 512 | 7.7134e-03 | 6.9880e+00 | 1.5812e+02 | 1.9475e-03 | 2.7695e-01 | 2.9753e+01 |
| swin_hybrid | swin | hybrid | 10000 | 512 | 3.2388e-03 | 5.8971e-01 | 1.3297e+01 | 2.0555e-03 | 3.0683e-01 | 2.8468e+01 |
| swin_transnet | swin | transnet | 10000 | 512 | -7.3695e-04 | 3.3207e-01 | 7.3543e+00 | 2.5094e-03 | 3.1436e-01 | 2.8527e+01 |
| transnet_cnn_residual | transnet | cnn_residual | 10000 | 512 | -2.8421e-01 | 1.2476e+01 | 2.8235e+02 | 1.0184e-03 | 3.4389e-01 | 2.6464e+01 |
| transnet_hybrid | transnet | hybrid | 10000 | 512 | -5.8725e-03 | 1.2083e+00 | 2.7301e+01 | 1.1264e-03 | 3.4363e-01 | 2.7515e+01 |
| transnet_transnet | transnet | transnet | 10000 | 512 | 2.3471e-02 | 7.9861e-01 | 1.7837e+01 | 1.0277e-03 | 4.2418e-01 | 2.3379e+01 |

## Largest Pairwise Centroid Distances

| left | right | centroid_l2 | centroid_cosine | std_profile_l2 |
|---|---|---:|---:|---:|
| convnext_cnn_residual | resnet_cnn_residual | 8.7336e+02 | -8.8606e-02 | 4.1056e+01 |
| convnext_cnn_residual | sparse_resnet_cnn_residual | 8.3183e+02 | -1.1316e-02 | 4.1349e+01 |
| cbam_cnn_cnn_residual | convnext_cnn_residual | 7.7484e+02 | -1.0265e-02 | 4.2768e+01 |
| cnn_cnn_residual | convnext_cnn_residual | 7.6126e+02 | 3.4233e-03 | 4.2621e+01 |
| convnext_cnn_residual | dscnn_cnn_residual | 7.4498e+02 | -3.6286e-02 | 4.3441e+01 |
| convnext_cnn_residual | transnet_cnn_residual | 7.3957e+02 | -1.2195e-02 | 4.1996e+01 |
| convnext_cnn_residual | mlp_mixer_cnn_residual | 7.3684e+02 | -1.9035e-02 | 4.3189e+01 |
| convnext_cnn_residual | mlp_ae_cnn_residual | 7.3235e+02 | 1.5509e-02 | 4.5199e+01 |
| convnext_cnn_residual | csinet_cnn_residual | 7.1782e+02 | -2.1164e-02 | 4.4027e+01 |
| convnext_cnn_residual | crnet_cnn_residual | 7.1131e+02 | 3.5232e-02 | 4.3534e+01 |
| clnet_cnn_residual | convnext_cnn_residual | 7.0227e+02 | 6.4066e-02 | 4.4420e+01 |
| convnext_cnn_residual | swin_cnn_residual | 7.0027e+02 | -1.2232e-02 | 4.5668e+01 |
| attention_cnn_cnn_residual | convnext_cnn_residual | 6.9532e+02 | 1.2625e-02 | 4.6159e+01 |
| resnet_cnn_residual | sparse_resnet_cnn_residual | 6.8859e+02 | -2.4830e-02 | 8.4027e+00 |
| cnn_hybrid | convnext_cnn_residual | 6.8341e+02 | -6.9162e-02 | 3.6462e+01 |
| convnext_cnn_residual | resnet_hybrid | 6.8262e+02 | -2.9395e-02 | 3.5419e+01 |
| convnext_cnn_residual | dscnn_hybrid | 6.8197e+02 | -6.1446e-02 | 3.6831e+01 |
| cbam_cnn_hybrid | convnext_cnn_residual | 6.8172e+02 | -2.1289e-02 | 3.6727e+01 |
| convnext_cnn_residual | convnext_hybrid | 6.8129e+02 | -3.5519e-03 | 3.6887e+01 |
| convnext_cnn_residual | csinet_hybrid | 6.8119e+02 | -3.2887e-02 | 3.9155e+01 |

## Generated Plots

- [centroid_l2_heatmap.png](centroid_l2_heatmap.png)
- [pca_projection.png](pca_projection.png)

## Files

- `summary.csv`: per-model distribution and PCA metrics.
- `pairwise_distances.csv`: pairwise code distribution distances.
- `analysis_config.json`: input settings for this report.
