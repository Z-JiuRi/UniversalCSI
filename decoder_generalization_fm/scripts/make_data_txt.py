import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate data.txt from pure transnet-decoder experiments.")
    parser.add_argument("--root", default="exps/COST2100/in")
    parser.add_argument("--output", default="decoder_generalization_fm/data/data.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []
    for args_json in sorted(root.rglob("args.json")):
        rel = args_json.parent.relative_to(root)
        parts = rel.parts
        if any(part in {
                "adapter", "encoder_canonical", "unfreeze_decoder",
                "unfreeze_fc_decoder", "teacher_code_adapter"} for part in parts):
            continue
        cfg = json.loads(args_json.read_text(encoding="utf-8"))
        if cfg.get("decoder") != "transnet":
            continue
        exp_dir = args_json.parent
        if not (exp_dir / "codewords" / "train_code.pt").exists():
            continue
        if not (exp_dir / "checkpoints" / "best_nmse.pth").exists():
            continue
        seed = cfg.get("seed")
        encoder = cfg.get("encoder")
        split = "train"
        if seed == 3407:
            split = "test"
        if seed == 42 and encoder not in {
                "transnet", "csinet", "crnet", "clnet", "cnn", "resnet"}:
            split = "test"
        rows.append((split, str(exp_dir)))
    text = [
        "# format: split,experiment_dir",
        "# split must be train or test.",
        "",
    ]
    for split in ("train", "test"):
        for item_split, exp_dir in rows:
            if item_split == split:
                text.append(f"{item_split},{exp_dir}")
        text.append("")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")
    print(f"saved={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
