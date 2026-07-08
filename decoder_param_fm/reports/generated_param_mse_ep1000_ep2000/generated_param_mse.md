# Generated Decoder Parameter MSE

## Setup

- Guide code: `exps/COST2100/in/seed42/transnet_transnet/codewords/train_code.pt`
- Target checkpoint: `exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth`
- Decoder args: `exps/COST2100/in/seed42/transnet_transnet/args.json`
- ODE steps: `16`
- Max guide codes: `0`

## Summary

| Experiment | Param Global MSE | CSI NMSE (dB) | CSI MSE Sum/Sample | Samples | Worst Tensor by MSE | Worst MSE |
|---|---:|---:|---:|---:|---|---:|
| `set_transformer_film_zscore_tok512_h2048_lr1e-4_ep2000_seed42` | 3.24043e-08 | 2.96401 | 1.82942 | 100000 | `decoder.norm.bias` | 0.000627214 |
| `set_transformer_film_zscore_tok512_h2048_lr1e-4_ep1000_seed42` | 7.08266e-08 | 7.01955 | 4.65444 | 100000 | `decoder.norm.bias` | 0.00123341 |

## set_transformer_film_zscore_tok512_h2048_lr1e-4_ep2000_seed42

| Tensor | Shape | Numel | MSE | RMSE | Target RMS | Param NMSE (dB) |
|---|---:|---:|---:|---:|---:|---:|
| `fc_decoder.weight` | `2048x512` | 1048576 | 7.59549e-10 | 2.75599e-05 | 0.0301752 | -60.7875 |
| `fc_decoder.bias` | `2048` | 2048 | 9.42005e-10 | 3.06921e-05 | 0.0295237 | -59.6629 |
| `decoder.layers.0.self_attn.in_proj_weight` | `192x64` | 12288 | 6.7808e-09 | 8.23456e-05 | 0.0765045 | -59.3609 |
| `decoder.layers.0.self_attn.in_proj_bias` | `192` | 192 | 2.07832e-06 | 0.00144164 | 0.0345298 | -27.5868 |
| `decoder.layers.0.self_attn.out_proj.weight` | `64x64` | 4096 | 1.44859e-08 | 0.000120358 | 0.124336 | -60.2824 |
| `decoder.layers.0.self_attn.out_proj.bias` | `64` | 64 | 2.20402e-06 | 0.0014846 | 0.0374053 | -28.0265 |
| `decoder.layers.0.multihead_attn.in_proj_weight` | `192x64` | 12288 | 6.7666e-09 | 8.22594e-05 | 0.0808379 | -59.8486 |
| `decoder.layers.0.multihead_attn.in_proj_bias` | `192` | 192 | 1.38653e-07 | 0.000372362 | 0.00971343 | -28.3281 |
| `decoder.layers.0.multihead_attn.out_proj.weight` | `64x64` | 4096 | 2.05603e-08 | 0.000143388 | 0.124459 | -58.7703 |
| `decoder.layers.0.multihead_attn.out_proj.bias` | `64` | 64 | 1.4638e-07 | 0.000382597 | 0.0080854 | -26.4992 |
| `decoder.layers.0.linear1.weight` | `2048x64` | 131072 | 1.28124e-09 | 3.57945e-05 | 0.0400107 | -60.9672 |
| `decoder.layers.0.linear1.bias` | `2048` | 2048 | 4.68629e-09 | 6.84565e-05 | 0.0737433 | -60.6462 |
| `decoder.layers.0.linear2.weight` | `64x2048` | 131072 | 1.22557e-09 | 3.50082e-05 | 0.039415 | -61.0298 |
| `decoder.layers.0.linear2.bias` | `64` | 64 | 1.16025e-06 | 0.00107715 | 0.0137152 | -22.0985 |
| `decoder.layers.0.norm1.weight` | `64` | 64 | 3.88254e-06 | 0.00197041 | 0.969015 | -53.8355 |
| `decoder.layers.0.norm1.bias` | `64` | 64 | 9.82834e-08 | 0.000313502 | 0.00800029 | -28.1373 |
| `decoder.layers.0.norm2.weight` | `64` | 64 | 2.92427e-06 | 0.00171005 | 0.998719 | -55.3287 |
| `decoder.layers.0.norm2.bias` | `64` | 64 | 6.29239e-08 | 0.000250846 | 0.00709628 | -29.0325 |
| `decoder.layers.0.norm3.weight` | `64` | 64 | 7.86922e-06 | 0.00280521 | 1.02102 | -51.2214 |
| `decoder.layers.0.norm3.bias` | `64` | 64 | 2.33186e-07 | 0.000482894 | 0.0112428 | -27.3404 |
| `decoder.layers.1.self_attn.in_proj_weight` | `192x64` | 12288 | 8.53985e-09 | 9.24113e-05 | 0.0871057 | -59.4864 |
| `decoder.layers.1.self_attn.in_proj_bias` | `192` | 192 | 6.28574e-07 | 0.000792827 | 0.01533 | -25.7273 |
| `decoder.layers.1.self_attn.out_proj.weight` | `64x64` | 4096 | 1.50354e-08 | 0.000122619 | 0.114892 | -59.4346 |
| `decoder.layers.1.self_attn.out_proj.bias` | `64` | 64 | 8.88012e-08 | 0.000297995 | 0.00762181 | -28.157 |
| `decoder.layers.1.multihead_attn.in_proj_weight` | `192x64` | 12288 | 9.1747e-09 | 9.57847e-05 | 0.0990795 | -60.2938 |
| `decoder.layers.1.multihead_attn.in_proj_bias` | `192` | 192 | 4.63519e-07 | 0.000680822 | 0.0139553 | -26.2341 |
| `decoder.layers.1.multihead_attn.out_proj.weight` | `64x64` | 4096 | 1.6649e-08 | 0.000129031 | 0.123299 | -59.6053 |
| `decoder.layers.1.multihead_attn.out_proj.bias` | `64` | 64 | 1.08512e-06 | 0.00104169 | 0.018866 | -25.1588 |
| `decoder.layers.1.linear1.weight` | `2048x64` | 131072 | 1.18685e-09 | 3.44508e-05 | 0.0438793 | -62.1012 |
| `decoder.layers.1.linear1.bias` | `2048` | 2048 | 3.88889e-09 | 6.2361e-05 | 0.0739931 | -61.4856 |
| `decoder.layers.1.linear2.weight` | `64x2048` | 131072 | 1.33473e-09 | 3.6534e-05 | 0.0461373 | -62.0271 |
| `decoder.layers.1.linear2.bias` | `64` | 64 | 1.43755e-06 | 0.00119898 | 0.0248294 | -26.3231 |
| `decoder.layers.1.norm1.weight` | `64` | 64 | 3.0757e-06 | 0.00175377 | 1.04039 | -55.4645 |
| `decoder.layers.1.norm1.bias` | `64` | 64 | 4.57037e-07 | 0.000676045 | 0.0152871 | -27.087 |
| `decoder.layers.1.norm2.weight` | `64` | 64 | 6.13895e-06 | 0.00247769 | 1.0234 | -52.32 |
| `decoder.layers.1.norm2.bias` | `64` | 64 | 1.58661e-06 | 0.00125961 | 0.0239427 | -25.5788 |
| `decoder.layers.1.norm3.weight` | `64` | 64 | 3.5479e-06 | 0.00188359 | 0.986171 | -54.3793 |
| `decoder.layers.1.norm3.bias` | `64` | 64 | 1.54689e-06 | 0.00124374 | 0.0225711 | -25.1765 |
| `decoder.norm.weight` | `64` | 64 | 0.000125691 | 0.0112112 | 0.553997 | -33.8771 |
| `decoder.norm.bias` | `64` | 64 | 0.000627214 | 0.0250442 | 0.392173 | -23.8954 |

## set_transformer_film_zscore_tok512_h2048_lr1e-4_ep1000_seed42

| Tensor | Shape | Numel | MSE | RMSE | Target RMS | Param NMSE (dB) |
|---|---:|---:|---:|---:|---:|---:|
| `fc_decoder.weight` | `2048x512` | 1048576 | 8.72418e-10 | 2.95367e-05 | 0.0301752 | -60.1858 |
| `fc_decoder.bias` | `2048` | 2048 | 1.36677e-09 | 3.69698e-05 | 0.0295237 | -58.0465 |
| `decoder.layers.0.self_attn.in_proj_weight` | `192x64` | 12288 | 6.90956e-09 | 8.31238e-05 | 0.0765045 | -59.2792 |
| `decoder.layers.0.self_attn.in_proj_bias` | `192` | 192 | 2.63936e-06 | 0.00162461 | 0.0345298 | -26.5489 |
| `decoder.layers.0.self_attn.out_proj.weight` | `64x64` | 4096 | 1.54213e-08 | 0.000124183 | 0.124336 | -60.0107 |
| `decoder.layers.0.self_attn.out_proj.bias` | `64` | 64 | 5.32075e-06 | 0.00230667 | 0.0374053 | -24.1989 |
| `decoder.layers.0.multihead_attn.in_proj_weight` | `192x64` | 12288 | 7.7181e-09 | 8.78527e-05 | 0.0808379 | -59.2772 |
| `decoder.layers.0.multihead_attn.in_proj_bias` | `192` | 192 | 1.91411e-07 | 0.000437505 | 0.00971343 | -26.9278 |
| `decoder.layers.0.multihead_attn.out_proj.weight` | `64x64` | 4096 | 1.60112e-08 | 0.000126535 | 0.124459 | -59.8563 |
| `decoder.layers.0.multihead_attn.out_proj.bias` | `64` | 64 | 1.45676e-07 | 0.000381675 | 0.0080854 | -26.5202 |
| `decoder.layers.0.linear1.weight` | `2048x64` | 131072 | 1.49378e-09 | 3.86495e-05 | 0.0400107 | -60.3007 |
| `decoder.layers.0.linear1.bias` | `2048` | 2048 | 4.86569e-09 | 6.97545e-05 | 0.0737433 | -60.483 |
| `decoder.layers.0.linear2.weight` | `64x2048` | 131072 | 1.43967e-09 | 3.7943e-05 | 0.039415 | -60.3306 |
| `decoder.layers.0.linear2.bias` | `64` | 64 | 2.31749e-06 | 0.00152233 | 0.0137152 | -19.0939 |
| `decoder.layers.0.norm1.weight` | `64` | 64 | 2.58935e-05 | 0.00508857 | 0.969015 | -45.5947 |
| `decoder.layers.0.norm1.bias` | `64` | 64 | 2.06034e-07 | 0.00045391 | 0.00800029 | -24.9227 |
| `decoder.layers.0.norm2.weight` | `64` | 64 | 1.35652e-05 | 0.0036831 | 0.998719 | -48.6646 |
| `decoder.layers.0.norm2.bias` | `64` | 64 | 1.09675e-07 | 0.000331171 | 0.00709628 | -26.6196 |
| `decoder.layers.0.norm3.weight` | `64` | 64 | 2.0547e-05 | 0.00453288 | 1.02102 | -47.0532 |
| `decoder.layers.0.norm3.bias` | `64` | 64 | 3.54431e-07 | 0.000595341 | 0.0112428 | -25.5222 |
| `decoder.layers.1.self_attn.in_proj_weight` | `192x64` | 12288 | 7.85417e-09 | 8.86237e-05 | 0.0871057 | -59.8499 |
| `decoder.layers.1.self_attn.in_proj_bias` | `192` | 192 | 7.64853e-07 | 0.000874559 | 0.01533 | -24.875 |
| `decoder.layers.1.self_attn.out_proj.weight` | `64x64` | 4096 | 1.55696e-08 | 0.000124778 | 0.114892 | -59.283 |
| `decoder.layers.1.self_attn.out_proj.bias` | `64` | 64 | 2.56644e-07 | 0.000506601 | 0.00762181 | -23.5478 |
| `decoder.layers.1.multihead_attn.in_proj_weight` | `192x64` | 12288 | 8.75139e-09 | 9.35489e-05 | 0.0990795 | -60.4989 |
| `decoder.layers.1.multihead_attn.in_proj_bias` | `192` | 192 | 6.08323e-07 | 0.000779951 | 0.0139553 | -25.0534 |
| `decoder.layers.1.multihead_attn.out_proj.weight` | `64x64` | 4096 | 1.74191e-08 | 0.000131981 | 0.123299 | -59.4089 |
| `decoder.layers.1.multihead_attn.out_proj.bias` | `64` | 64 | 1.33528e-06 | 0.00115554 | 0.018866 | -24.2579 |
| `decoder.layers.1.linear1.weight` | `2048x64` | 131072 | 1.84247e-09 | 4.29239e-05 | 0.0438793 | -60.1912 |
| `decoder.layers.1.linear1.bias` | `2048` | 2048 | 4.98717e-09 | 7.06199e-05 | 0.0739931 | -60.4053 |
| `decoder.layers.1.linear2.weight` | `64x2048` | 131072 | 2.08155e-09 | 4.5624e-05 | 0.0461373 | -60.0972 |
| `decoder.layers.1.linear2.bias` | `64` | 64 | 2.92255e-06 | 0.00170955 | 0.0248294 | -23.2417 |
| `decoder.layers.1.norm1.weight` | `64` | 64 | 1.10214e-05 | 0.00331985 | 1.04039 | -49.9216 |
| `decoder.layers.1.norm1.bias` | `64` | 64 | 7.60967e-07 | 0.000872334 | 0.0152871 | -24.8728 |
| `decoder.layers.1.norm2.weight` | `64` | 64 | 2.30537e-05 | 0.00480142 | 1.0234 | -46.5735 |
| `decoder.layers.1.norm2.bias` | `64` | 64 | 2.39534e-06 | 0.00154769 | 0.0239427 | -23.7898 |
| `decoder.layers.1.norm3.weight` | `64` | 64 | 9.83816e-06 | 0.00313658 | 0.986171 | -49.9499 |
| `decoder.layers.1.norm3.bias` | `64` | 64 | 4.2959e-06 | 0.00207266 | 0.0225711 | -20.7405 |
| `decoder.norm.weight` | `64` | 64 | 0.000413046 | 0.0203235 | 0.553997 | -28.7102 |
| `decoder.norm.bias` | `64` | 64 | 0.00123341 | 0.03512 | 0.392173 | -20.9585 |
