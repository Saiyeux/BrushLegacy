"""config_loader.py — Load config.yaml from the repo root."""
from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else _CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def robot_ip(path: str | None = None) -> str:
    return load_config(path)["robot"]["ip"]
