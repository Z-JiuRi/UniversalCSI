import torch
from torch.utils.data import Dataset


class CodewordPairDataset(Dataset):
    def __init__(self, source_path, target_path, split="train",
                 val_ratio=0.1, max_samples=0):
        source = torch.load(source_path, weights_only=True,
                            map_location="cpu").float()
        target = torch.load(target_path, weights_only=True,
                            map_location="cpu").float()
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source and target codewords must be 2D tensors")
        if source.shape != target.shape:
            raise ValueError(
                f"source and target shape mismatch: {source.shape} vs "
                f"{target.shape}")
        if max_samples and source.size(0) > max_samples:
            source = source[:max_samples].contiguous()
            target = target[:max_samples].contiguous()
        n = source.size(0)
        if val_ratio <= 0:
            sl = slice(0, n)
            self.source = source[sl].contiguous()
            self.target = target[sl].contiguous()
            self.indices = torch.arange(n, dtype=torch.long)[sl].contiguous()
            return
        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n_val, n - 1)) if n > 1 else 0
        if split == "train":
            sl = slice(0, n - n_val)
        elif split in ("val", "test"):
            sl = slice(n - n_val, n)
        elif split == "all":
            sl = slice(0, n)
        else:
            raise ValueError(f"Unknown split: {split}")
        self.source = source[sl].contiguous()
        self.target = target[sl].contiguous()
        self.indices = torch.arange(n, dtype=torch.long)[sl].contiguous()

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx], self.indices[idx]
