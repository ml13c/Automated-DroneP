"""YAML config loading. All thresholds, gains and power numbers live in configs/."""

from __future__ import annotations

import sys
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


def platform_dataset_yaml(dataset_yaml: str | Path) -> Path:
    r"""Copy of an Ultralytics dataset.yaml whose `path` points at this checkout on this OS.

    build_dataset.py stores an absolute path from the OS that built it (e.g. C:\...), which
    WSL can't read; writes `dataset_<sys.platform>.yaml` next to it and returns that path.
    """
    src = Path(dataset_yaml)
    if not src.is_absolute():
        src = REPO_ROOT / src
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    doc["path"] = str(src.parent.resolve())
    dst = src.with_name(f"dataset_{sys.platform}.yaml")
    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dst
