import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_main_models_package():
    package_name = "decoder_generalization_main_models"
    if package_name in sys.modules:
        return sys.modules[package_name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "models" / "__init__.py",
        submodule_search_locations=[str(ROOT / "models")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_state_dict(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, weights_only=True, map_location=torch.device("cpu"))
    state = checkpoint.get("state_dict", checkpoint)
    state = dict(state)
    for key in list(state.keys()):
        if key.endswith("total_ops") or key.endswith("total_params"):
            del state[key]
    return state


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
    main_models = load_main_models_package()
    model = main_models.universal_csi(
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


def validate_compatible_states(reference, state, source):
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    if missing or unexpected:
        raise ValueError(
            f"decoder state mismatch for {source}: "
            f"missing={missing}, unexpected={unexpected}")
    for key, ref_value in reference.items():
        if tuple(ref_value.shape) != tuple(state[key].shape):
            raise ValueError(
                f"shape mismatch for {key} in {source}: "
                f"{tuple(ref_value.shape)} vs {tuple(state[key].shape)}")


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
        numel = value.numel()
        num_tokens = int(math.ceil(numel / token_size))
        layer_id = infer_layer_id(name)
        token_start = global_token
        for token_offset in range(num_tokens):
            start = token_offset * token_size
            end = min(start + token_size, numel)
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
            "numel": numel,
            "token_start": token_start,
            "token_end": global_token,
            "layer_id": layer_id,
        })
        meta["max_layer_id"] = max(meta["max_layer_id"], layer_id)
        meta["max_token_offset"] = max(meta["max_token_offset"], num_tokens - 1)
    meta["num_tokens"] = global_token
    return meta


def meta_tensors_from_meta(meta, device=None):
    out = {
        "tensor_ids": torch.tensor(
            [token["tensor_id"] for token in meta["tokens"]], dtype=torch.long),
        "layer_ids": torch.tensor(
            [token["layer_id"] for token in meta["tokens"]], dtype=torch.long),
        "token_offsets": torch.tensor(
            [token["token_offset"] for token in meta["tokens"]], dtype=torch.long),
    }
    if device is not None:
        out = {key: value.to(device) for key, value in out.items()}
    return out


def state_to_tokens(state, meta, device=None):
    token_size = meta["token_size"]
    tokens = []
    masks = []
    for token in meta["tokens"]:
        name = token["tensor_name"]
        start = token["token_offset"] * token_size
        valid = token["valid_elements"]
        flat = state[name].flatten()
        out = torch.zeros(token_size, dtype=flat.dtype)
        mask = torch.zeros(token_size, dtype=torch.float32)
        out[:valid] = flat[start:start + valid]
        mask[:valid] = 1.0
        tokens.append(out)
        masks.append(mask)
    tokens = torch.stack(tokens)
    masks = torch.stack(masks)
    if device is not None:
        tokens = tokens.to(device)
        masks = masks.to(device)
    return tokens, masks, meta_tensors_from_meta(meta, device=device)


def tokens_to_state(tokens, meta):
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


def compute_tensor_zscore_stats(states, eps=1e-8):
    if not states:
        raise ValueError("states must not be empty")
    stats = {}
    for name in states[0]:
        values = [state[name].double() for state in states]
        total = sum(value.sum() for value in values)
        denom = sum(value.numel() for value in values)
        mean = total / max(denom, 1)
        var = sum((value - mean).pow(2).sum() for value in values) / max(denom, 1)
        std = var.sqrt().clamp_min(eps)
        stats[name] = {
            "method": "zscore",
            "shape": list(states[0][name].shape),
            "mean": mean.float(),
            "std": std.float(),
        }
    return stats


def load_or_compute_stats(cache_path, train_states):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True, map_location="cpu"), True
    stats = compute_tensor_zscore_stats(train_states)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, cache_path)
    return stats, False


def normalize_state(state, stats):
    out = {}
    for name, value in state.items():
        stat = stats[name]
        mean = stat["mean"].to(value.device)
        std = stat["std"].to(value.device)
        out[name] = (value - mean) / std
    return out


def denormalize_state(norm_state, stats):
    out = {}
    for name, value in norm_state.items():
        stat = stats[name]
        mean = stat["mean"].to(value.device)
        std = stat["std"].to(value.device)
        out[name] = value * std + mean
    return out


def masked_mse(pred, target, mask):
    mask = mask.to(pred.dtype)
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def load_codes(path, max_samples=0):
    codes = torch.load(path, weights_only=True, map_location="cpu").float()
    if codes.ndim != 2:
        raise ValueError(f"code tensor must be 2D, got {tuple(codes.shape)}")
    if max_samples and codes.size(0) > max_samples:
        codes = codes[:max_samples].contiguous()
    return codes


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


def clone_decoder_with_state(args_json, state, device):
    decoder, _ = build_decoder_from_args(args_json)
    missing, unexpected = decoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"decoder load mismatch: {missing}, {unexpected}")
    return decoder.to(device).eval()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True),
                          encoding="utf-8")
