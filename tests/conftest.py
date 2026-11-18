"""Markers for tests that need things a fresh clone (or CI) doesn't have. They skip, never fail:

@pytest.mark.weights("models/detector.onnx")  # file (relative to the repo) must exist
@pytest.mark.data("data/detector")            # dataset folder must exist
@pytest.mark.webcam                           # only with DRONEVISION_WEBCAM=1
@pytest.mark.gpu                              # only if ONNX Runtime has the CUDA provider
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dronevision.config import REPO_ROOT


def _cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def pytest_collection_modifyitems(config, items):
    gpu = None
    for item in items:
        for kind in ("weights", "data"):
            m = item.get_closest_marker(kind)
            if m is not None:
                missing = [p for p in m.args if not (REPO_ROOT / Path(p)).exists()]
                if missing:
                    item.add_marker(pytest.mark.skip(reason=f"{kind} not present: {missing[0]}"))
        if item.get_closest_marker("webcam") and os.environ.get("DRONEVISION_WEBCAM") != "1":
            item.add_marker(pytest.mark.skip(reason="webcam test (set DRONEVISION_WEBCAM=1)"))
        if item.get_closest_marker("gpu"):
            gpu = _cuda_available() if gpu is None else gpu
            if not gpu:
                item.add_marker(pytest.mark.skip(reason="no CUDA provider in ONNX Runtime"))
