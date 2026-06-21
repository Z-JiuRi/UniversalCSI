# Experiment: `seed42/cbam_cnn_transnet`

Full training baseline (seed=42) on the **COST2100** indoor (`in`) dataset.

This is a **full training baseline** (seed=42) covering 14 encoder architectures × 3 decoder types. The model is trained from scratch on CSI angle-delay domain samples with no pretrained components.

## Key configuration

- **Encoder:** `cbam_cnn`
- **Decoder:** `transnet`
- **Compression ratio:** `1/4`
- **Seed:** `42`
- **Epochs:** `400`
- **Learning rate:** `0.0002`
- **Scheduler:** `cosine`
- **Batch size:** `200`

## Dataset

- **Train:** `/storage/hujiacong/zxd/datasets/cost2100/in_train.pt`
- **Validation:** `/storage/hujiacong/zxd/datasets/cost2100/in_val.pt`
- **Test:** `/storage/hujiacong/zxd/datasets/cost2100/in_test.pt`

## Model architecture

CSI input → Encoder → (optional CodeAdapter) → Decoder → reconstructed CSI

- Input shape: `(B, 2, 32, 32)`
- Code dimension: `512`

## Outputs

- `checkpoints/best_nmse.pth` — best model weights by validation NMSE
- `checkpoints/last.pth` — final epoch weights (present for frozen_decoder experiments)
- `run.log` — training and validation log
- `args.json` — full hyperparameter configuration
- `tensorboard/` — TensorBoard event files