"""YAML config loading. All thresholds, gains and power numbers live in configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config.

    Accepts a bare name (``"camera"`` -> ``configs/camera.yaml``) or a path.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{path.name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level")
    return data
