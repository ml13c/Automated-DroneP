"""Finger-count classifier: square hand crop -> "0".."5" / "none".

`preprocess` is shared by training (val/test), ONNX export checks and live inference, so the
network always sees identically prepared crops. The crop itself comes from the FULL-resolution
frame around the detector's hand box (the "ROI zoom" that the IMX500 will do on-sensor).
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronevision.datasets import square_crop_px

# Normalisation of the ImageNet-pretrained backbone (a property of the weights, not a knob).
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(crop_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR crop (any size) -> normalised float32 tensor (1, 3, size, size), RGB."""
    img = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1)[None])


def hand_crop(frame_bgr: np.ndarray, xyxy: Sequence[float], pad: float) -> np.ndarray:
    """Square crop around a pixel box, enlarged by ``pad`` per side (as in the dataset)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    yolo = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
    cx0, cy0, cx1, cy1 = square_crop_px(yolo, w, h, pad)
    return frame_bgr[cy0:cy1, cx0:cx1]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class FingerPrediction:
    label: str
    prob: float
    probs: dict[str, float]


class OnnxFingerClassifier:
    def __init__(
        self, model_path: str | Path, providers: Sequence[str] = ("CPUExecutionProvider",)
    ):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), so, providers=list(providers))
        self.provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name, self.size = inp.name, int(inp.shape[-1])
        meta = self.session.get_modelmeta().custom_metadata_map
        if "classes" not in meta:
            raise ValueError(f"{model_path} has no 'classes' metadata")
        self.classes: list[str] = list(ast.literal_eval(meta["classes"]))

    @classmethod
    def from_config(cls, cfg: dict, repo_root: Path) -> OnnxFingerClassifier:
        inf = cfg["inference"]
        return cls(repo_root / inf["onnx"], inf.get("providers", ["CPUExecutionProvider"]))

    def __call__(self, crop_bgr: np.ndarray) -> FingerPrediction:
        logits = self.session.run(None, {self.input_name: preprocess(crop_bgr, self.size)})[0][0]
        p = softmax(logits)
        k = int(p.argmax())
        return FingerPrediction(
            self.classes[k], float(p[k]), dict(zip(self.classes, p.tolist(), strict=True))
        )
