"""Load each .pt from cost2100, take 1/10 random samples, save to cost2100_10000/."""

import torch
import os

SRC = "/storage/hujiacong/zxd/datasets/cost2100"
DST = "/storage/hujiacong/zxd/datasets/cost2100_10000"

SPLITS = ["in_train", "in_val", "in_test"]

for name in SPLITS:
    src_path = os.path.join(SRC, f"{name}.pt")
    dst_path = os.path.join(DST, f"{name}.pt")

    print(f"Loading {src_path} ...")
    data = torch.load(src_path, weights_only=True, map_location="cpu")
    data = data.to(torch.float32)

    n = data.size(0)
    n_sub = n // 10
    print(f"  original: {n} samples, subsample: {n_sub}")

    # deterministic random shuffle
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    sub = data[idx[:n_sub]].clone()
    torch.save(sub, dst_path)
    print(f"  saved {dst_path}  shape={tuple(sub.shape)}  size={sub.numel() * sub.element_size() / 1e6:.1f}MB")

print("Done.")
