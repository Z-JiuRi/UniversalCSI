"""
Generate test codes by encoding test CSI using the pre-trained encoder.
Usage: conda run -n torch python decoder_param_fm/gen_test_codes.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Need parent for imports
PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from models import universal_csi
from utils.init import _load_clean_state_dict


def main():
    target_exp = Path("exps/COST2100/in/seed42/transnet_transnet")
    test_csi_path = "/storage/hujiacong/zxd/datasets/cost2100/in_test.pt"
    output_code_path = target_exp / "codewords" / "test_code.pt"

    args_path = target_exp / "args.json"
    checkpoint_path = target_exp / "checkpoints" / "best_nmse.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = json.loads(args_path.read_text(encoding="utf-8"))

    channel = args.get("channel", 2)
    nt = args.get("nt", 32)
    nc = args.get("nc", 32)
    cr = args.get("cr", 4)
    d_model = args.get("d_model", 64)
    dim_feedforward = args.get("dim_feedforward", 2048)

    print(f"Building model encoder={args['encoder']}, decoder={args['decoder']}, cr={cr}")
    model = universal_csi(
        encoder_name=args["encoder"],
        decoder_name=args["decoder"],
        reduction=cr,
        d_model=d_model,
        channel=channel,
        nt=nt,
        nc=nc,
        dim_feedforward=dim_feedforward,
        hidden=args.get("hidden", 16),
        num_blocks=args.get("num_blocks", 2),
    )

    # Load pretrained weights
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = _load_clean_state_dict(str(checkpoint_path))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("Model loaded successfully")

    print(f"Loading test CSI from {test_csi_path}")
    test_csi = torch.load(test_csi_path, weights_only=True, map_location="cpu").float()
    if test_csi.ndim == 2:
        test_csi = test_csi.view(-1, channel, nt, nc)
    print(f"Test CSI shape: {test_csi.shape}")

    # Encode in batches
    batch_size = 1024
    all_codes = []
    with torch.no_grad():
        for start in range(0, test_csi.size(0), batch_size):
            end = min(start + batch_size, test_csi.size(0))
            batch = test_csi[start:end].to(device)
            codes = model.encode(batch)
            all_codes.append(codes.cpu())
            if (start // batch_size) % 5 == 0:
                print(f"  encoded {end}/{test_csi.size(0)}")

    test_codes = torch.cat(all_codes, dim=0)
    print(f"Test codes shape: {test_codes.shape}")

    output_code_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_codes, str(output_code_path))
    print(f"Saved test codes to {output_code_path}")


if __name__ == "__main__":
    main()
