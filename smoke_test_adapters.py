#!/usr/bin/env python3
"""Smoke test: random (1000, 2, 32, 32) CSI data, 5 epochs per adapter config.

We bypass `utils.parser`'s module-level parse_args by pre-populating
sys.argv before the first import.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Build synthetic argv so that utils.parser.parse_args() succeeds without
# real file paths.  We override per-test via manual attribute assignment
# after the module is loaded.
# ---------------------------------------------------------------------------
_original_argv = sys.argv[:]
sys.argv = [
    "smoke_test",
    "--train_path", "/dev/null",
    "--val_path", "/dev/null",
    "--test_path", "/dev/null",
    "--batch_size", "32",
    "--workers", "0",
    "--encoder", "transnet",
    "--decoder", "hybrid",
    "--cr", "4",
    "--epochs", "5",
    "--cpu",
]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Now safe to import — parser has already parsed dummy args
from utils.parser import args as parsed_args
from utils import init_model, init_device
from models import universal_csi, multi_seed_adapter_csi

# Restore argv
sys.argv = _original_argv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
torch.manual_seed(42)
device = torch.device("cpu")

N = 1000
X = torch.randn(N, 2, 32, 32)
Y = X.clone()
ds = TensorDataset(X, Y)
dl = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def train_one_epoch(model, opt, crit, loader):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        preds = model(x)
        loss = 0.0
        if isinstance(preds, dict):
            for k, p in preds.items():
                loss += crit(p, y)
            loss /= len(preds)
        else:
            loss = crit(preds, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def all_params(model):
    return sum(p.numel() for p in model.parameters())


def make_optimizer(model, lr=1e-3, wd=1e-3):
    decay, no_decay = [], []
    for nm, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or nm.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr)


# ---------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------
configs = [
    ("no_adapter", dict()),
    ("encoder_only", dict(adapter_positions=["encoder"])),
    ("semantic_projector_only", dict(adapter_positions=["semantic_projector"])),
    ("token_projection_only", dict(adapter_positions=["token_projection"])),
    ("token_mixer_only", dict(adapter_positions=["token_mixer"])),
    ("all_four", dict(adapter_positions=["encoder", "semantic_projector",
                                          "token_projection", "token_mixer"])),
    ("encoder+semantic_projector", dict(adapter_positions=["encoder",
                                             "semantic_projector"])),
    ("all_with_custom_hidden", dict(adapter_positions=["encoder", "semantic_projector",
                                         "token_projection", "token_mixer"],
                                    adapter_hidden_dim=1024)),
]

print("=" * 72)
print("SMOKE TEST: adapter positions on HybridDecoder (frozen encoder+decoder)")
print(f"Device: {device}  |  Data: {X.shape}")
print("=" * 72)

errors = []

for name, kwargs in configs:
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  args: {kwargs}")
    sys.stdout.flush()

    try:
        model = universal_csi(
            encoder_name="transnet",
            decoder_name="hybrid",
            reduction=4,
            d_model=64,
            channel=2, nt=32, nc=32,
            dim_feedforward=256,
            hidden=16, num_blocks=2,
            **kwargs,
        )
        model.to(device)

        # Freeze encoder + decoder (but not internal adapters)
        for n, p in model.named_parameters():
            if (n.startswith("encoder.") or
                (n.startswith("decoder.") and
                 not n.startswith("decoder.sp_adapter.") and
                 not n.startswith("decoder.tp_adapter.") and
                 not n.startswith("decoder.tm_adapter."))):
                p.requires_grad = False

        trainable = trainable_params(model)
        total = all_params(model)
        print(f"  Params: {total:,} total / {trainable:,} trainable")

        if trainable == 0:
            if name == "no_adapter":
                print(f"  ✅ EXPECTED (all frozen, no adapter)")
                continue
            print(f"  ⚠ No trainable params — adapter may not be created")
            errors.append((name, "zero trainable params"))
            continue

        opt = make_optimizer(model)
        crit = nn.MSELoss()

        for ep in range(1, 6):
            l = train_one_epoch(model, opt, crit, dl)
            print(f"  Epoch {ep:2d}  loss = {l:.6f}")

        print(f"  ✅ PASSED")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        errors.append((name, str(e)))

print("\n" + "=" * 72)
if errors:
    print(f"FAILURES ({len(errors)}/{len(configs)}):")
    for n, e in errors:
        print(f"  - {n}: {e}")
    sys.exit(1)
else:
    print(f"ALL {len(configs)} CONFIGS PASSED ✅")
    sys.exit(0)
