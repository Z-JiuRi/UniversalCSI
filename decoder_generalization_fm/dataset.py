import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentEntry:
    split: str
    exp_dir: Path
    args_json: Path
    code_path: Path
    checkpoint_path: Path
    encoder: str
    decoder: str
    seed: int


def parse_data_txt(path):
    path = Path(path)
    entries = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno} should be 'split,exp_dir'")
        split, exp_dir = parts
        if split not in ("train", "test"):
            raise ValueError(f"{path}:{lineno} split must be train or test")
        exp_dir = Path(exp_dir)
        args_json = exp_dir / "args.json"
        code_path = exp_dir / "codewords" / "train_code.pt"
        checkpoint_path = exp_dir / "checkpoints" / "best_nmse.pth"
        for required in (args_json, code_path, checkpoint_path):
            if not required.exists():
                raise FileNotFoundError(f"{path}:{lineno} missing {required}")
        cfg = json.loads(args_json.read_text(encoding="utf-8"))
        decoder = cfg.get("decoder")
        if decoder != "transnet":
            raise ValueError(
                f"{path}:{lineno} expected decoder=transnet, got {decoder}")
        entries.append(ExperimentEntry(
            split=split,
            exp_dir=exp_dir,
            args_json=args_json,
            code_path=code_path,
            checkpoint_path=checkpoint_path,
            encoder=cfg.get("encoder", ""),
            decoder=decoder,
            seed=int(cfg.get("seed", -1)),
        ))
    if not entries:
        raise ValueError(f"{path} does not contain any experiments")
    if not any(item.split == "train" for item in entries):
        raise ValueError(f"{path} does not contain train experiments")
    if not any(item.split == "test" for item in entries):
        raise ValueError(f"{path} does not contain test experiments")
    return entries


def split_entries(entries):
    train = [item for item in entries if item.split == "train"]
    test = [item for item in entries if item.split == "test"]
    return train, test

