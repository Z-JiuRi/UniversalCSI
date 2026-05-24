# UniversalCSI

UniversalCSI is a TransNet-based CSI feedback experiment project. It keeps the
training, logging, checkpointing, and `.pt` dataloader flow from `TransNet`, but
changes the model into:

```text
CSI input
  -> selectable encoder
  -> optional code adapter
  -> selectable decoder
  -> reconstructed CSI
```

Supported encoders:

```text
csinet    PyTorch rewrite of the CsiNet encoder
crnet     CRNet-style multi-resolution CNN encoder
clnet     CLNet-style attention/lightweight CNN encoder
transnet  Original TransNet Transformer encoder
```

Supported decoders:

```text
transnet      TransNet-style Transformer decoder baseline
cnn_residual  Linear expansion + lightweight CNN residual refinement
hybrid        Transformer decoder + lightweight CNN residual refinement
```

The default `transnet` decoder is copied from TransNet's decoder side:

```text
code: (B, 2048 / cr)
  -> fc_decoder
  -> TransformerDecoder
  -> output: (B, 2, nt, nc)
```

## Files

Main additions:

```text
models/UniversalCSI.py   selectable encoders + selectable decoders
models/__init__.py       exports universal_csi
utils/parser.py          adds --encoder, --decoder and --code_adapter
utils/init.py            builds UniversalCSI instead of plain TransNet
```

Most other files are inherited from `TransNet`.

## Quick Usage

Example:

```bash
python main.py \
  --exp_name crnet_encoder_trans_decoder \
  --encoder crnet \
  --train_path /path/to/in_train.pt \
  --val_path /path/to/in_val.pt \
  --test_path /path/to/in_test.pt \
  --epochs 400 \
  --batch_size 200 \
  --workers 0 \
  --cr 4 \
  --decoder hybrid \
  --nt 32 \
  --nc 32 \
  --d_model 64 \
  --scheduler const \
  --gpu 0
```

Other encoder choices:

```bash
--encoder csinet
--encoder clnet
--encoder transnet
```

Other decoder choices:

```bash
--decoder transnet
--decoder cnn_residual
--decoder hybrid
```

Useful optional switches:

```bash
--code_adapter
```

## Shape Contract

All encoders follow the same interface:

```text
input:  (B, 2, 32, 32)
code:   (B, 2048 / cr)
```

All decoders follow:

```text
code:   (B, 2048 / cr)
output: (B, 2, 32, 32)
```

For non-default `nt`/`nc`, the input dimension is:

```text
input_dim = channel * nt * nc
code_dim = input_dim / cr
```

`input_dim` must be divisible by `cr`. Transformer-based decoders
(`transnet`, `hybrid`) also require divisibility by `d_model`.

## Notes

- This project currently uses the TransNet `.pt` dataloader.
- CsiNet here is a PyTorch encoder rewrite, not a TensorFlow/Keras weight import.
- LoRA flags are still present from TransNet but are not wired for UniversalCSI.
- Use `--freeze_components encoder` if you want to train only the shared decoder.
