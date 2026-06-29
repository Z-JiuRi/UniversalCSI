import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


SONGTI_PATH = "/home/hujiacong/zxd/.envs/SongTi.ttf"
TNR_PATH = "/home/hujiacong/zxd/.envs/TimesNewRoman.ttf"


def setup_matplotlib_fonts():
    """Use the same font files as test_fonts.py when available."""
    font_paths = [SONGTI_PATH, TNR_PATH]
    for path in font_paths:
        if Path(path).exists():
            fm.fontManager.addfont(path)
    if Path(SONGTI_PATH).exists():
        songti = fm.FontProperties(fname=SONGTI_PATH)
        plt.rcParams["font.family"] = songti.get_name()
    if Path(TNR_PATH).exists():
        tnr = fm.FontProperties(fname=TNR_PATH)
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = tnr.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def write_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def is_float(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def safe_float(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def parse_seed_arch(name):
    match = re.match(r"seed(\d+)_(.+)_([^_]+)$", name)
    if not match:
        return None, None, None
    return int(match.group(1)), match.group(2), match.group(3)


def infer_scheme(exp_dir, root):
    exp_dir = Path(exp_dir)
    root = Path(root)
    rel = exp_dir.relative_to(root)
    parts = rel.parts
    if not parts:
        return ""
    if parts[0] != "adapter":
        return parts[0]
    if len(parts) >= 2:
        return parts[1]
    return "adapter"


def infer_family(exp_dir, root):
    exp_dir = Path(exp_dir)
    root = Path(root)
    rel = exp_dir.relative_to(root)
    parts = rel.parts
    if not parts:
        return ""
    if parts[0] != "adapter":
        return "encoder"
    if len(parts) >= 3:
        return f"adapter/{parts[1]}/{parts[2]}"
    return "adapter"


def discover_experiments(root):
    root = Path(root)
    for args_path in sorted(root.rglob("args.json")):
        exp_dir = args_path.parent
        run_log = exp_dir / "run.log"
        if run_log.exists():
            yield exp_dir


def finite_min(values):
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    return min(vals) if vals else np.nan


def finite_mean(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else np.nan
