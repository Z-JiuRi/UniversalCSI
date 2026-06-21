# Experiment: `seed2026/transnet_hybrid`

Full training baseline (seed=2026) on the **COST2100** indoor (`in`) dataset.

This is a **full training baseline** with a different random seed. The model is trained from scratch with no pretrained components. It serves as the pretrained encoder/decoder checkpoint for downstream fine-tuning experiments.

## Key configuration

- **Encoder:** `transnet`
- **Decoder:** `hybrid`
- **Compression ratio:** `1/4`
- **Seed:** `2026`
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