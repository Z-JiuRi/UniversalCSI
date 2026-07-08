"""Noise injection on codewords to map code MSE → decoder NMSE.

Directly builds TransNet encoder/decoder by importing specific classes.
"""
import torch, math, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = "cuda:0"
torch.set_grad_enabled(False)

from models.encoders.transnet import TransNetEncoder
from models.decoders.transnet import TransNetDecoder

# Build models (default: 2 layers, nhead=2)
enc = TransNetEncoder(reduction=4, d_model=64, channel=2, nt=32, nc=32,
                       dim_feedforward=2048)
dec = TransNetDecoder(reduction=4, d_model=64, channel=2, nt=32, nc=32,
                       dim_feedforward=2048)

# Load checkpoint and strip the top-level "decoder." / "encoder." prefix
ckpt = torch.load(
    "exps/COST2100/in/seed42/transnet_transnet/checkpoints/best_nmse.pth",
    weights_only=True, map_location="cpu"
)
sd = ckpt["state_dict"]

enc_sd = {}
dec_sd = {}
for k, v in sd.items():
    if k.startswith("encoder.") and "total_" not in k:
        enc_sd[k[len("encoder."):]] = v
    elif k.startswith("decoder.") and "total_" not in k:
        dec_sd[k[len("decoder."):]] = v

enc.load_state_dict(enc_sd, strict=False)
dec.load_state_dict(dec_sd, strict=True)
enc.eval().to(device)
dec.eval().to(device)

dec_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
enc_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
print(f"Encoder params: {enc_params:,}  Decoder params: {dec_params:,}")

# Load subset of CSI for speed
csi = torch.load(
    "/storage/hujiacong/zxd/datasets/cost2100/in_train.pt",
    weights_only=True, map_location="cpu"
).to(torch.float32)
n = 20000
csi = csi[:n]
print(f"CSI: {csi.shape}")

# Baseline: encode → decode
with torch.no_grad():
    codes = enc(csi.to(device)).cpu()
    recon = dec(codes.to(device)).cpu()

error = (csi - recon).pow(2).sum(dim=(1, 2, 3))
power = csi.pow(2).sum(dim=(1, 2, 3))
baseline_nmse = 10 * math.log10(error.sum().item() / power.sum().item())
baseline_mse = error.mean().item()
print(f"Baseline NMSE: {baseline_nmse:.4f} dB  MSE: {baseline_mse:.2e}")

# ---- Noise sweep ----
results = []
noise_rel_range = [0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3,
                   2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5]

code_std = codes.std().item()
var_per_elem = code_std ** 2

print(f"\n{'rel_sigma':>9} {'code_MSE':>14} {'code_RMSE':>10} {'code_SNR':>8} {'dec_MSE':>14} {'NMSE':>9} {'gap':>7}")
print("-" * 75)

for rel in noise_rel_range:
    sigma = rel * code_std
    noise = torch.randn_like(codes) * sigma
    codes_noisy = codes + noise
    code_mse = noise.pow(2).mean().item()

    with torch.no_grad():
        recon = dec(codes_noisy.to(device)).cpu()
    error = (csi - recon).pow(2).sum(dim=(1, 2, 3))
    nmse = 10 * math.log10((error.sum() / power.sum()).item())
    mse = error.mean().item()
    gap = nmse - baseline_nmse
    snr = -10 * math.log10(code_mse / var_per_elem) if code_mse > 1e-20 else float("inf")

    results.append({
        "sigma_rel": rel,
        "code_mse": code_mse,
        "code_rmse": math.sqrt(code_mse),
        "code_snr_db": round(snr, 2),
        "decoder_mse": mse,
        "nmse_db": round(nmse, 4),
        "nmse_gap": round(gap, 4),
    })
    print(f"{rel:9.6f} {code_mse:14.6e} {math.sqrt(code_mse):10.6f} {snr:8.1f} {mse:14.6e} {nmse:9.2f} {gap:7.3f}")

# ---- Find thresholds ----
max_code_mse_1db = None
max_code_rmse_1db = None
for r in results:
    if abs(r["nmse_gap"]) <= 1.0:
        max_code_mse_1db = r["code_mse"]
        max_code_rmse_1db = r["code_rmse"]
    else:
        break

print(f"\n{'='*60}")
print(f"BASELINE: NMSE = {baseline_nmse:.2f} dB")
print(f"1dB TARGET: NMSE <= {baseline_nmse + 1:.2f} dB")
print(f"  Max allowable code MSE  = {max_code_mse_1db:.6e}")
print(f"  Max allowable code RMSE = {max_code_rmse_1db:.6f}")
print(f"  Min required code SNR   = {-10 * math.log10(max_code_mse_1db / var_per_elem):.1f} dB")

# Compare with existing Staged MLP+LoRA
mapper_code_mse = 4.659476e-03  # from affine_mlp_h1024 mapper
mapper_code_rmse = math.sqrt(mapper_code_mse)
mapper_snr = -10 * math.log10(mapper_code_mse / var_per_elem)
print(f"\nEXISTING STAGED MLP(h=1024) MAPPER:")
print(f"  Code MSE  = {mapper_code_mse:.6e}")
print(f"  Code RMSE = {mapper_code_rmse:.6f}")
print(f"  Code SNR  = {mapper_snr:.1f} dB")
print(f"  Decoder NMSE after LoRA = -26.655 dB")
print(f"  Gap to native           = {28.13 - 26.655:.3f} dB")
print(f"  Code MSE exceeds 1dB threshold by {mapper_code_mse / max_code_mse_1db:.1f}x")

# --------------------------------------------------------------
# Also test: what if we just improve the mapper?
# Find what code MSE we need to hit 1dB gap
target_nmse = baseline_nmse + 1.0  # -27.13 dB
print(f"\n{'='*60}")
print(f"TO HIT 1dB GAP (NMSE = {target_nmse:.2f} dB):")
print(f"  Need code MSE <= {max_code_mse_1db:.6e}")
print(f"  Current mapper code MSE = {mapper_code_mse:.6e}")
print(f"  Need to reduce code MSE by {mapper_code_mse / max_code_mse_1db:.1f}x")

os.makedirs("analysis", exist_ok=True)
with open("analysis/code_noise_sweep.json", "w") as f:
    json.dump({
        "baseline_nmse": baseline_nmse,
        "baseline_mse": baseline_mse,
        "code_dim": 512,
        "code_std": code_std,
        "var_per_elem": var_per_elem,
        "decoder_params": dec_params,
        "1db_threshold": {
            "max_code_mse": max_code_mse_1db,
            "max_code_rmse": max_code_rmse_1db,
            "min_code_snr": -10 * math.log10(max_code_mse_1db / var_per_elem),
        },
        "existing_mapper": {
            "code_mse": mapper_code_mse,
            "code_rmse": mapper_code_rmse,
            "code_snr": mapper_snr,
            "decoder_nmse_after_lora": -26.655,
            "gap_to_native": 28.13 - 26.655,
            "code_mse_ratio_to_1db": mapper_code_mse / max_code_mse_1db if max_code_mse_1db else None,
        },
        "results": results,
    }, f, indent=2)
print(f"\nSaved analysis/code_noise_sweep.json")
