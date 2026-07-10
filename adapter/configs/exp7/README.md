# exp7 configs

Goal: focused target-encoder consistency sweep after exp6.

Fixed settings:

```text
source = seed1014 transnet_transnet
target encoder/decoder = seed1024 transnet_transnet
mapper = affine_residual_mlp
hidden_dim = 512
num_blocks = 4
train_affine = true
align_ridge = 0
gate_mode = block, learnable_residual_gate = false
code_loss_type = mse
lambda_code = 1
encoder_consistency_target = target
epochs = 400
batch_size = 1024
lr = 5e-4
```

Sweep:

```text
residual_scale = 0.35, 0.4, 0.5
lambda_recon = 700, 900, 1000
lambda_encoder_consistency = 0, 0.75, 1.0, 1.5, 2.0, 3.0
```

Total:

```text
3 * 3 * 6 = 54 configs
```

Directory split:

```text
adapter/configs/exp7/0 -> gpu 0
adapter/configs/exp7/1 -> gpu 1
adapter/configs/exp7/4 -> gpu 4
adapter/configs/exp7/6 -> gpu 6
adapter/configs/exp7/7 -> gpu 7
```

Each `(residual_scale, lambda_recon)` pair includes a `lambda_encoder_consistency=0`
baseline, so target-consistency gains can be measured inside the same sweep.
