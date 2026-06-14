import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "extract": {"fps": 2, "dedup": True, "dedup_threshold": 3.0},
    "sam3": {"model": "model/sam3.pt", "conf": 0.25, "half": True},
    "qc": {"conf_threshold": 0.40, "topk_percent": None},
    "train": {"yolo_model": "yolo11s", "epochs": 100, "imgsz": 640, "batch": 16},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path="config.yaml") -> dict:
    """Load config.yaml deep-merged over DEFAULTS. Missing file -> defaults."""
    path = Path(path)
    user = {}
    if path.exists():
        user = yaml.safe_load(path.read_text()) or {}
    return _deep_merge(DEFAULTS, user)
