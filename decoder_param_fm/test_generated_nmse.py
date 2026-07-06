import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decoder_param_fm.param_utils import (  # noqa: E402
    build_decoder_from_args,
    load_codes,
    load_csi,
    load_generated_decoder_state,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NMSE for a generated decoder with paired codes.")
    parser.add_argument("--decoder_state", type=str, required=True)
    parser.add_argument("--decoder_args_json", type=str, required=True)
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--csi_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


@torch.no_grad()
def evaluate(decoder, codes, csi, batch_size, device):
    decoder.eval()
    error_sum = torch.zeros((), dtype=torch.float64, device=device)
    power_sum = torch.zeros((), dtype=torch.float64, device=device)
    num = codes.size(0)
    for start in range(0, num, batch_size):
        end = min(start + batch_size, num)
        code_batch = codes[start:end].to(device, non_blocking=True)
        csi_batch = csi[start:end].to(device, non_blocking=True)
        pred = decoder(code_batch)
        diff = pred - csi_batch
        error_sum += diff.double().pow(2).sum()
        power_sum += csi_batch.double().pow(2).sum()
    nmse = 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))
    mse = error_sum / max(num, 1)
    return {
        "num_samples": int(num),
        "mse_sum_per_sample": float(mse.detach().cpu()),
        "nmse_db": float(nmse.detach().cpu()),
    }


def main():
    args = parse_args()
    decoder_cfg = json.loads(
        Path(args.decoder_args_json).read_text(encoding="utf-8"))
    csi_path = args.csi_path or decoder_cfg["train_path"]
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu
        else "cpu")

    codes = load_codes(args.code_path, max_samples=args.max_samples)
    csi = load_csi(
        csi_path,
        decoder_cfg.get("channel", 2),
        decoder_cfg.get("nt", 32),
        decoder_cfg.get("nc", 32),
        max_samples=args.max_samples,
    )
    if codes.size(0) != csi.size(0):
        raise ValueError(
            f"code/csi sample mismatch: {codes.size(0)} vs {csi.size(0)}")

    decoder_state = load_generated_decoder_state(args.decoder_state)
    decoder, _ = build_decoder_from_args(args.decoder_args_json)
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"decoder load mismatch: {missing}, {unexpected}")
    decoder = decoder.to(device)

    result = evaluate(decoder, codes, csi, args.batch_size, device)
    result.update({
        "decoder_state": str(args.decoder_state),
        "decoder_args_json": str(args.decoder_args_json),
        "code_path": str(args.code_path),
        "csi_path": str(csi_path),
    })
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
