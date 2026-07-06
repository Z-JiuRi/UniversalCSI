import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import universal_csi  # noqa: E402


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_state_dict(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, weights_only=True, map_location=torch.device("cpu"))
    state = checkpoint.get("state_dict", checkpoint)
    for key in list(state.keys()):
        if key.endswith("total_ops") or key.endswith("total_params"):
            del state[key]
    return state


def load_param_meta(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_generated_decoder_state(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if isinstance(checkpoint, dict) and "decoder_state_dict" in checkpoint:
        state = checkpoint["decoder_state_dict"]
    else:
        state = checkpoint.get("state_dict", checkpoint)
        state = {
            key[len("decoder."):] if key.startswith("decoder.") else key: value
            for key, value in state.items()
        }
    return {key: value.detach().cpu().float() for key, value in state.items()}


def extract_decoder_state(checkpoint_path):
    state = clean_state_dict(checkpoint_path)
    decoder_state = {
        key[len("decoder."):]: value.detach().cpu().float()
        for key, value in state.items()
        if key.startswith("decoder.")
    }
    if decoder_state:
        return decoder_state
    return {key: value.detach().cpu().float() for key, value in state.items()}


def build_decoder_from_args(args_json, seed=None):
    cfg = load_json(args_json)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    model = universal_csi(
        encoder_name="transnet",
        decoder_name=cfg.get("decoder", "transnet"),
        reduction=cfg.get("cr", 4),
        d_model=cfg.get("d_model", 64),
        channel=cfg.get("channel", 2),
        nt=cfg.get("nt", 32),
        nc=cfg.get("nc", 32),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        hidden=cfg.get("hidden", 16),
        num_blocks=cfg.get("num_blocks", 2),
    )
    return model.decoder, cfg


def make_base_and_target(args_json, target_checkpoint, seed):
    base_decoder, cfg = build_decoder_from_args(args_json, seed=seed)
    base_state = {
        key: value.detach().cpu().float()
        for key, value in base_decoder.state_dict().items()
    }
    target_state = extract_decoder_state(target_checkpoint)
    missing = sorted(set(base_state) - set(target_state))
    unexpected = sorted(set(target_state) - set(base_state))
    if missing or unexpected:
        raise ValueError(
            f"decoder state mismatch: missing={missing}, unexpected={unexpected}")
    for key in base_state:
        if tuple(base_state[key].shape) != tuple(target_state[key].shape):
            raise ValueError(
                f"shape mismatch for {key}: {tuple(base_state[key].shape)} vs "
                f"{tuple(target_state[key].shape)}")
    return base_state, target_state, cfg


def infer_layer_id(name):
    if name.startswith("fc_decoder."):
        return 0
    if name.startswith("decoder.layers."):
        parts = name.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            return 1 + int(parts[2])
    if name.startswith("decoder.norm."):
        return 1000
    return 1001


def build_param_meta(state, token_size):
    meta = {
        "token_size": int(token_size),
        "tensors": [],
        "tokens": [],
        "num_tensors": len(state),
        "max_layer_id": 0,
        "max_token_offset": 0,
    }
    global_token = 0
    for tensor_id, (name, value) in enumerate(state.items()):
        flat_numel = value.numel()
        num_tokens = int(math.ceil(flat_numel / token_size))
        layer_id = infer_layer_id(name)
        token_start = global_token
        for token_offset in range(num_tokens):
            start = token_offset * token_size
            end = min(start + token_size, flat_numel)
            valid = end - start
            meta["tokens"].append({
                "global_token_id": global_token,
                "tensor_name": name,
                "tensor_id": tensor_id,
                "layer_id": layer_id,
                "token_offset": token_offset,
                "valid_elements": valid,
            })
            global_token += 1
        meta["tensors"].append({
            "name": name,
            "tensor_id": tensor_id,
            "shape": list(value.shape),
            "numel": flat_numel,
            "token_start": token_start,
            "token_end": global_token,
            "layer_id": layer_id,
        })
        meta["max_layer_id"] = max(meta["max_layer_id"], layer_id)
        meta["max_token_offset"] = max(meta["max_token_offset"], num_tokens - 1)
    meta["num_tokens"] = global_token
    return meta


def meta_tensors_from_meta(meta, device=None):
    tensor_ids = torch.tensor(
        [token["tensor_id"] for token in meta["tokens"]], dtype=torch.long)
    layer_ids = torch.tensor(
        [token["layer_id"] for token in meta["tokens"]], dtype=torch.long)
    token_offsets = torch.tensor(
        [token["token_offset"] for token in meta["tokens"]], dtype=torch.long)
    if device is not None:
        tensor_ids = tensor_ids.to(device)
        layer_ids = layer_ids.to(device)
        token_offsets = token_offsets.to(device)
    return {
        "tensor_ids": tensor_ids,
        "layer_ids": layer_ids,
        "token_offsets": token_offsets,
    }


def mask_from_meta(meta, device=None):
    token_size = meta["token_size"]
    masks = []
    for token in meta["tokens"]:
        mask = torch.zeros(token_size, dtype=torch.float32)
        mask[:token["valid_elements"]] = 1.0
        masks.append(mask)
    masks = torch.stack(masks)
    if device is not None:
        masks = masks.to(device)
    return masks


def compute_norm_stats(base_state, target_state, method="rms", eps=1e-8):
    stats = {}
    for name, base in base_state.items():
        target = target_state[name]
        delta = target - base
        item = {"method": method, "shape": list(base.shape)}
        if method == "rms":
            scale = delta.pow(2).mean().sqrt().clamp_min(eps)
            item["scale"] = scale
        elif method == "zscore":
            mean = delta.mean()
            std = delta.std(unbiased=False).clamp_min(eps)
            item["mean"] = mean
            item["std"] = std
        else:
            raise ValueError(f"Unknown norm method: {method}")
        stats[name] = item
    return stats


def normalize_target_delta(base_state, target_state, norm_stats):
    norm_state = {}
    for name, base in base_state.items():
        target = target_state[name].to(base.device)
        delta = target - base
        stat = norm_stats[name]
        if stat["method"] == "rms":
            norm_state[name] = delta / stat["scale"].to(delta.device)
        elif stat["method"] == "zscore":
            mean = stat["mean"].to(delta.device)
            std = stat["std"].to(delta.device)
            norm_state[name] = (delta - mean) / std
        else:
            raise ValueError(stat["method"])
    return norm_state


def denormalize_state(base_state, norm_state, norm_stats):
    out = {}
    for name, base in base_state.items():
        stat = norm_stats[name]
        value = norm_state[name]
        base = base.to(value.device)
        if stat["method"] == "rms":
            out[name] = base + value * stat["scale"].to(value.device)
        elif stat["method"] == "zscore":
            std = stat["std"].to(value.device)
            mean = stat["mean"].to(value.device)
            out[name] = base + value * std + mean
        else:
            raise ValueError(stat["method"])
    return out


def state_to_tokens(norm_state, meta, device=None):
    token_size = meta["token_size"]
    tokens = []
    masks = []
    tensor_ids = []
    layer_ids = []
    token_offsets = []
    for token in meta["tokens"]:
        name = token["tensor_name"]
        start = token["token_offset"] * token_size
        valid = token["valid_elements"]
        flat = norm_state[name].flatten()
        out = torch.zeros(token_size, dtype=flat.dtype)
        mask = torch.zeros(token_size, dtype=torch.float32)
        out[:valid] = flat[start:start + valid]
        mask[:valid] = 1.0
        tokens.append(out)
        masks.append(mask)
        tensor_ids.append(token["tensor_id"])
        layer_ids.append(token["layer_id"])
        token_offsets.append(token["token_offset"])
    tokens = torch.stack(tokens)
    masks = torch.stack(masks)
    tensor_ids = torch.tensor(tensor_ids, dtype=torch.long)
    layer_ids = torch.tensor(layer_ids, dtype=torch.long)
    token_offsets = torch.tensor(token_offsets, dtype=torch.long)
    if device is not None:
        tokens = tokens.to(device)
        masks = masks.to(device)
        tensor_ids = tensor_ids.to(device)
        layer_ids = layer_ids.to(device)
        token_offsets = token_offsets.to(device)
    return tokens, masks, {
        "tensor_ids": tensor_ids,
        "layer_ids": layer_ids,
        "token_offsets": token_offsets,
    }


def tokens_to_norm_state(tokens, meta):
    state = {}
    token_size = meta["token_size"]
    for tensor in meta["tensors"]:
        name = tensor["name"]
        flat = torch.empty(
            tensor["numel"], dtype=tokens.dtype, device=tokens.device)
        for idx in range(tensor["token_start"], tensor["token_end"]):
            token = meta["tokens"][idx]
            start = token["token_offset"] * token_size
            valid = token["valid_elements"]
            flat[start:start + valid] = tokens[idx, :valid]
        state[name] = flat.view(*tensor["shape"])
    return state


def save_artifacts(path, base_state, target_state, meta, norm_stats, cfg):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(base_state, path / "theta_base.pt")
    torch.save(target_state, path / "theta_star.pt")
    torch.save(norm_stats, path / "norm_stats.pt")
    (path / "param_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    (path / "decoder_args_resolved.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def masked_mse(pred, target, mask):
    mask = mask.to(pred.dtype)
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def nmse_from_sums(error_sum, power_sum):
    return 10.0 * torch.log10(error_sum / power_sum.clamp_min(1e-12))


def load_csi(path, channel, nt, nc, max_samples=0):
    data = torch.load(path, weights_only=True, map_location="cpu").float()
    if data.ndim == 2:
        data = data.view(-1, channel, nt, nc)
    if data.ndim != 4 or tuple(data.shape[1:]) != (channel, nt, nc):
        raise ValueError(
            f"{path} should have shape (N,{channel},{nt},{nc}), "
            f"got {tuple(data.shape)}")
    if max_samples and data.size(0) > max_samples:
        data = data[:max_samples].contiguous()
    return data


def load_codes(path, max_samples=0):
    codes = torch.load(path, weights_only=True, map_location="cpu").float()
    if codes.ndim != 2:
        raise ValueError(f"code tensor must be 2D, got {tuple(codes.shape)}")
    if max_samples and codes.size(0) > max_samples:
        codes = codes[:max_samples].contiguous()
    return codes


def clone_decoder_with_state(args_json, state, device):
    decoder, _ = build_decoder_from_args(args_json)
    missing, unexpected = decoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"decoder load mismatch: {missing}, {unexpected}")
    return decoder.to(device).eval()
