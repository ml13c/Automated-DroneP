"""ONNX Runtime YOLO detector (hand, ball): the inference path that mirrors the embedded one.

Pre-processing matches Ultralytics (letterbox to a square, pad value 114, RGB, /255, NCHW);
post-processing decodes the raw head output [1, 4 + n_classes, anchors] (no NMS baked in),
applies per-class confidence thresholds and class-aware NMS. Thresholds come from
configs/detector.yaml, never from code.
"""

from __future__ import annotations

import ast
import contextlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int
    name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def size(self) -> tuple[float, float]:
        return (self.x2 - self.x1, self.y2 - self.y1)


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to ``size`` x ``size`` (centered, value 114)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    out[top : top + nh, left : left + nw] = resized
    return out, Letterbox(scale, left, top)


def to_tensor(letterboxed_bgr: np.ndarray) -> np.ndarray:
    rgb = letterboxed_bgr[:, :, ::-1]
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS on xyxy boxes; returns kept indices, highest score first."""
    order = np.argsort(-scores)
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(
    raw: np.ndarray,
    names: Sequence[str],
    conf: Mapping[str, float],
    iou: float,
    max_det: int,
    lb: Letterbox,
    image_size: tuple[int, int],
) -> list[Detection]:
    """Raw YOLO head output [1, 4 + C, A] -> detections in original-image pixels."""
    pred = raw[0].T  # (A, 4 + C)
    boxes_cxcywh, scores = pred[:, :4], pred[:, 4:]
    cls = scores.argmax(1)
    best = scores[np.arange(len(cls)), cls]
    thr = np.array([conf[n] for n in names], dtype=np.float32)[cls]
    m = best >= thr
    if not m.any():
        return []
    b, s, c = boxes_cxcywh[m], best[m], cls[m]
    xyxy = np.stack(
        [
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        ],
        axis=1,
    )
    # class-aware NMS: offset boxes per class so different classes never suppress each other
    offset = c[:, None].astype(np.float32) * 4096.0
    keep = nms(xyxy + offset, s, iou)[:max_det]

    w, h = image_size
    xyxy = xyxy[keep]
    xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - lb.pad_x) / lb.scale, 0, w)
    xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - lb.pad_y) / lb.scale, 0, h)
    return [
        Detection(*map(float, box), float(score), int(k), names[int(k)])
        for box, score, k in zip(xyxy, s[keep], c[keep], strict=True)
    ]


class OnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf: Mapping[str, float],
        iou: float,
        max_det: int,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        intra_op_threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            # On failure ORT falls back to CPU; the provider actually used is in .provider
            with contextlib.suppress(Exception):
                ort.preload_dlls()
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), so, providers=list(providers))
        self.provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[-1])
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.names = [names[i] for i in sorted(names)]
        missing = [n for n in self.names if n not in conf]
        if missing:
            raise ValueError(f"no confidence threshold configured for classes {missing}")
        self.conf, self.iou, self.max_det = dict(conf), iou, max_det
        self.last_ms: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg: Mapping, repo_root: Path) -> OnnxDetector:
        inf = cfg["inference"]
        return cls(
            repo_root / inf["onnx"],
            conf=inf["conf"],
            iou=inf["iou"],
            max_det=inf["max_det"],
            providers=inf.get("providers", ["CPUExecutionProvider"]),
            intra_op_threads=inf.get("intra_op_threads", 0),
        )

    def __call__(self, image_bgr: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        lb_img, lb = letterbox(image_bgr, self.imgsz)
        x = to_tensor(lb_img)
        t1 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: x})[0]
        t2 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        dets = decode(raw, self.names, self.conf, self.iou, self.max_det, lb, (w, h))
        t3 = time.perf_counter()
        self.last_ms = {"pre": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3}
        return dets
