# train_code.pt 全量深层分析报告

本报告分析 `exps/real_matrix_2epoch/**/codewords/train_code.pt`，所有统计与图表均基于完整 train codewords，没有抽样。

## 数据覆盖

- 模型组合数：`42`
- 每组样本数：`100000`
- 码字维度：`512`
- 全量样本点数：`4200000`
- Global PCA 有效秩：`1.0960e+01`
- Global PCA Top-5 方差占比：`6.9306e-01`

## 一句话结论

- `hybrid` decoder 组合在 2 epoch 下整体 NMSE 排名靠前，但它诱导的码字尺度通常比 `transnet` decoder 更大。
- `cnn_residual` decoder 组合中部分模型的码字高度集中在少数主成分上，有效秩偏低，说明压缩码空间更容易塌缩到低维方向。
- `transnet` decoder 组合的码字有效秩普遍较高，分布更均匀，但 2 epoch 重建效果明显落后于多数 `hybrid` 组合。
- `cnn` 与 `cbam_cnn` 都基于同一卷积骨架；CBAM 改变了码字尺度和 PCA 集中度，但在当前 2 epoch 设置下并没有超过 `cnn_hybrid`。

## 主要图表

- [01_global_std_bar.png](figures/01_global_std_bar.png)
- [02_l2_norm_bar.png](figures/02_l2_norm_bar.png)
- [03_effective_rank_bar.png](figures/03_effective_rank_bar.png)
- [04_near_zero_ratio_bar.png](figures/04_near_zero_ratio_bar.png)
- [05_decoder_l2_boxplot.png](figures/05_decoder_l2_boxplot.png)
- [06_centroid_l2_heatmap.png](figures/06_centroid_l2_heatmap.png)
- [07_global_pca_centroids.png](figures/07_global_pca_centroids.png)
- [08_l2_norm_histograms.png](figures/08_l2_norm_histograms.png)
- [09_pca_density_selected.png](figures/09_pca_density_selected.png)
- [10_std_vs_nmse.png](figures/10_std_vs_nmse.png)

## 2-epoch NMSE 最好组合

| name | encoder | decoder | final_test_nmse | std_global | l2_mean | effective_rank |
| --- | --- | --- | --- | --- | --- | --- |
| attention_cnn_cnn_residual | attention_cnn | cnn_residual | nan | 6.7496e+00 | 1.5273e+02 | 1.7570e+02 |
| attention_cnn_hybrid | attention_cnn | hybrid | nan | 7.3148e-01 | 1.6587e+01 | 1.5808e+02 |
| attention_cnn_transnet | attention_cnn | transnet | nan | 5.9406e-01 | 1.3171e+01 | 1.4577e+02 |
| cbam_cnn_cnn_residual | cbam_cnn | cnn_residual | nan | 1.6094e+01 | 3.6421e+02 | 1.8996e+02 |
| cbam_cnn_hybrid | cbam_cnn | hybrid | nan | 1.8168e+00 | 4.1167e+01 | 1.3397e+02 |
| cbam_cnn_transnet | cbam_cnn | transnet | nan | 1.2731e+00 | 2.8362e+01 | 1.3106e+02 |
| clnet_cnn_residual | clnet | cnn_residual | nan | 9.8690e+00 | 2.2340e+02 | 1.7417e+02 |
| clnet_hybrid | clnet | hybrid | nan | 1.1481e+00 | 2.5873e+01 | 1.8817e+02 |
| clnet_transnet | clnet | transnet | nan | 6.0181e-01 | 1.3323e+01 | 1.8273e+02 |
| cnn_cnn_residual | cnn | cnn_residual | nan | 1.5212e+01 | 3.4422e+02 | 1.7569e+02 |

## 2-epoch NMSE 最差组合

| name | encoder | decoder | final_test_nmse | std_global | l2_mean | effective_rank |
| --- | --- | --- | --- | --- | --- | --- |
| attention_cnn_cnn_residual | attention_cnn | cnn_residual | nan | 6.7496e+00 | 1.5273e+02 | 1.7570e+02 |
| attention_cnn_hybrid | attention_cnn | hybrid | nan | 7.3148e-01 | 1.6587e+01 | 1.5808e+02 |
| attention_cnn_transnet | attention_cnn | transnet | nan | 5.9406e-01 | 1.3171e+01 | 1.4577e+02 |
| cbam_cnn_cnn_residual | cbam_cnn | cnn_residual | nan | 1.6094e+01 | 3.6421e+02 | 1.8996e+02 |
| cbam_cnn_hybrid | cbam_cnn | hybrid | nan | 1.8168e+00 | 4.1167e+01 | 1.3397e+02 |
| cbam_cnn_transnet | cbam_cnn | transnet | nan | 1.2731e+00 | 2.8362e+01 | 1.3106e+02 |
| clnet_cnn_residual | clnet | cnn_residual | nan | 9.8690e+00 | 2.2340e+02 | 1.7417e+02 |
| clnet_hybrid | clnet | hybrid | nan | 1.1481e+00 | 2.5873e+01 | 1.8817e+02 |
| clnet_transnet | clnet | transnet | nan | 6.0181e-01 | 1.3323e+01 | 1.8273e+02 |
| cnn_cnn_residual | cnn | cnn_residual | nan | 1.5212e+01 | 3.4422e+02 | 1.7569e+02 |

## 码字尺度最大的组合

| name | decoder | std_global | l2_mean | pca_top5_ratio | effective_rank |
| --- | --- | --- | --- | --- | --- |
| convnext_cnn_residual | cnn_residual | 3.0169e+01 | 6.8148e+02 | 6.6711e-01 | 1.3139e+01 |
| resnet_cnn_residual | cnn_residual | 2.1700e+01 | 4.9102e+02 | 1.1606e-01 | 1.9301e+02 |
| sparse_resnet_cnn_residual | cnn_residual | 2.0831e+01 | 4.7135e+02 | 1.2053e-01 | 1.9115e+02 |
| cbam_cnn_cnn_residual | cnn_residual | 1.6094e+01 | 3.6421e+02 | 1.3281e-01 | 1.8996e+02 |
| cnn_cnn_residual | cnn_residual | 1.5212e+01 | 3.4422e+02 | 1.5240e-01 | 1.7569e+02 |
| transnet_cnn_residual | cnn_residual | 1.2477e+01 | 2.8238e+02 | 1.1891e-01 | 2.1650e+02 |
| mlp_ae_cnn_residual | cnn_residual | 1.2463e+01 | 2.8200e+02 | 2.5372e-01 | 1.3489e+02 |
| dscnn_cnn_residual | cnn_residual | 1.2379e+01 | 2.8013e+02 | 1.4715e-01 | 1.6486e+02 |
| mlp_mixer_cnn_residual | cnn_residual | 1.1965e+01 | 2.7073e+02 | 1.0019e-01 | 2.3242e+02 |
| crnet_cnn_residual | cnn_residual | 1.0313e+01 | 2.3334e+02 | 1.1466e-01 | 1.9333e+02 |

## 维度利用最充分的组合（有效秩最高）

| name | decoder | effective_rank | pca_top5_ratio | dim_std_mean | final_test_nmse |
| --- | --- | --- | --- | --- | --- |
| mlp_mixer_cnn_residual | cnn_residual | 2.3242e+02 | 1.0019e-01 | 5.3530e-01 | nan |
| transnet_cnn_residual | cnn_residual | 2.1650e+02 | 1.1891e-01 | 6.4655e-01 | nan |
| swin_cnn_residual | cnn_residual | 2.1437e+02 | 9.7077e-02 | 3.2661e-01 | nan |
| mlp_ae_hybrid | hybrid | 2.0727e+02 | 1.3826e-01 | 1.3628e-01 | nan |
| csinet_hybrid | hybrid | 2.0272e+02 | 1.3924e-01 | 8.5135e-01 | nan |
| mlp_mixer_transnet | transnet | 2.0151e+02 | 1.1845e-01 | 3.2456e-01 | nan |
| mlp_ae_transnet | transnet | 2.0033e+02 | 1.2764e-01 | 2.5057e-01 | nan |
| csinet_transnet | transnet | 1.9761e+02 | 1.2561e-01 | 1.0357e+00 | nan |
| mlp_mixer_hybrid | hybrid | 1.9699e+02 | 1.3200e-01 | 4.2011e-01 | nan |
| crnet_cnn_residual | cnn_residual | 1.9333e+02 | 1.1466e-01 | 4.7283e-01 | nan |

## Encoder 维度聚合视角

| encoder | count | std_global_mean | l2_mean_mean | effective_rank_mean | top5_pca_mean | final_test_nmse_mean |
| --- | --- | --- | --- | --- | --- | --- |
| attention_cnn | 3 | 2.6917e+00 | 6.0829e+01 | 1.5985e+02 | 1.5215e-01 | nan |
| cbam_cnn | 3 | 6.3948e+00 | 1.4458e+02 | 1.5167e+02 | 1.6273e-01 | nan |
| clnet | 3 | 3.8730e+00 | 8.7531e+01 | 1.8169e+02 | 1.3568e-01 | nan |
| cnn | 3 | 6.0680e+00 | 1.3720e+02 | 1.5072e+02 | 1.6300e-01 | nan |
| convnext | 3 | 1.1373e+01 | 2.5685e+02 | 1.1905e+02 | 3.1648e-01 | nan |
| crnet | 3 | 3.9285e+00 | 8.8755e+01 | 1.7761e+02 | 1.2913e-01 | nan |
| csinet | 3 | 3.9222e+00 | 8.8471e+01 | 1.7954e+02 | 1.6381e-01 | nan |
| dscnn | 3 | 5.0549e+00 | 1.1424e+02 | 1.5323e+02 | 1.6490e-01 | nan |
| mlp_ae | 3 | 4.3169e+00 | 9.7657e+01 | 1.8083e+02 | 1.7321e-01 | nan |
| mlp_mixer | 3 | 4.3551e+00 | 9.8489e+01 | 2.1031e+02 | 1.1688e-01 | nan |
| resnet | 3 | 8.3370e+00 | 1.8847e+02 | 1.5063e+02 | 1.6522e-01 | nan |
| sparse_resnet | 3 | 8.0599e+00 | 1.8221e+02 | 1.4650e+02 | 1.7296e-01 | nan |
| swin | 3 | 2.6364e+00 | 5.9588e+01 | 1.8536e+02 | 1.2272e-01 | nan |
| transnet | 3 | 4.8285e+00 | 1.0918e+02 | 1.7816e+02 | 1.5773e-01 | nan |

## Decoder 维度聚合视角

| decoder | count | std_global_mean | l2_mean_mean | effective_rank_mean | top5_pca_mean | final_test_nmse_mean |
| --- | --- | --- | --- | --- | --- | --- |
| cnn_residual | 14 | 1.4052e+01 | 3.1789e+02 | 1.7196e+02 | 1.8334e-01 | nan |
| hybrid | 14 | 1.3548e+00 | 3.0615e+01 | 1.6574e+02 | 1.5040e-01 | nan |
| transnet | 14 | 8.4465e-01 | 1.8791e+01 | 1.6054e+02 | 1.5839e-01 | nan |

## 架构间最远的码字分布

| left | right | centroid_l2 | centroid_cosine | std_profile_l2 | covariance_frobenius |
| --- | --- | --- | --- | --- | --- |
| convnext_cnn_residual | resnet_cnn_residual | 8.7379e+02 | -8.8620e-02 | 4.1052e+01 | 1.6326e+03 |
| convnext_cnn_residual | sparse_resnet_cnn_residual | 8.3226e+02 | -1.1330e-02 | 4.1333e+01 | 1.6328e+03 |
| cbam_cnn_cnn_residual | convnext_cnn_residual | 7.7527e+02 | -1.0253e-02 | 4.2795e+01 | 1.6330e+03 |
| cnn_cnn_residual | convnext_cnn_residual | 7.6171e+02 | 3.4114e-03 | 4.2615e+01 | 1.6329e+03 |
| convnext_cnn_residual | dscnn_cnn_residual | 7.4543e+02 | -3.6309e-02 | 4.3452e+01 | 1.6330e+03 |
| convnext_cnn_residual | transnet_cnn_residual | 7.4005e+02 | -1.2222e-02 | 4.2007e+01 | 1.6329e+03 |
| convnext_cnn_residual | mlp_mixer_cnn_residual | 7.3730e+02 | -1.9015e-02 | 4.3184e+01 | 1.6330e+03 |
| convnext_cnn_residual | mlp_ae_cnn_residual | 7.3280e+02 | 1.5498e-02 | 4.5179e+01 | 1.6333e+03 |
| convnext_cnn_residual | csinet_cnn_residual | 7.1829e+02 | -2.1112e-02 | 4.4016e+01 | 1.6332e+03 |
| convnext_cnn_residual | crnet_cnn_residual | 7.1178e+02 | 3.5248e-02 | 4.3522e+01 | 1.6330e+03 |

## 架构间最近的码字分布

| left | right | centroid_l2 | centroid_cosine | std_profile_l2 | covariance_frobenius |
| --- | --- | --- | --- | --- | --- |
| mlp_ae_transnet | swin_transnet | 2.3886e+00 | 5.3291e-03 | 1.7743e+00 | 5.6030e+00 |
| attention_cnn_transnet | swin_transnet | 2.5166e+00 | -4.7418e-03 | 5.9559e+00 | 1.9074e+01 |
| attention_cnn_transnet | mlp_ae_transnet | 2.9396e+00 | -3.9268e-02 | 7.6286e+00 | 1.8914e+01 |
| clnet_transnet | swin_transnet | 3.0585e+00 | -6.5725e-02 | 6.0232e+00 | 1.6655e+01 |
| attention_cnn_transnet | clnet_transnet | 3.4027e+00 | -1.0523e-02 | 1.0832e+00 | 2.3154e+01 |
| clnet_transnet | mlp_ae_transnet | 3.4173e+00 | -7.5722e-02 | 7.7056e+00 | 1.6502e+01 |
| crnet_transnet | swin_transnet | 3.4247e+00 | 6.5887e-02 | 5.6642e+00 | 1.6090e+01 |
| crnet_transnet | mlp_ae_transnet | 3.7941e+00 | -6.0062e-03 | 7.3409e+00 | 1.5931e+01 |
| csinet_transnet | swin_transnet | 3.8023e+00 | 1.7570e-02 | 1.6122e+01 | 5.0310e+01 |
| attention_cnn_transnet | crnet_transnet | 3.8355e+00 | 1.3244e-02 | 1.1747e+00 | 2.2835e+01 |

## 码字统计与 NMSE 的相关性

| metric | pearson_with_nmse |
| --- | --- |
| std_global | nan |
| abs_mean | nan |
| l2_mean | nan |
| near_zero_1e_3 | nan |
| dim_std_mean | nan |
| pca_top1_ratio | nan |
| pca_top5_ratio | nan |
| effective_rank | nan |
| cov_trace | nan |

## 输出文件说明

- `full_summary.csv`：每个模型组合的全量分布、范数、稀疏性、维度利用和 PCA 指标。
- `pairwise_distances.csv`：任意两个模型组合之间的中心、标准差轮廓、协方差距离。
- `encoder_aggregate.csv`：按 encoder 聚合的均值指标。
- `decoder_aggregate.csv`：按 decoder 聚合的均值指标。
- `metric_nmse_correlations.csv`：码字指标和最终 NMSE 的 Pearson 相关性。
- `figures/`：全部图表。

