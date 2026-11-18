"""Pluggable user markers: something on the user's wrist that says "this hand is the user".

Every implementation turns a BGR frame into a list of `Marker`s. The user lock only uses a
marker's position (and size), so swapping the AprilTag on cardboard for the glove's wrist pod
(colour patch or LED) is a new `MarkerDetector` subclass plus one entry in `_REGISTRY`;
nothing downstream changes.

Implemented: `AprilTagMarkerDetector` (tag36h11 printed and mounted flat on cardboard).
Planned:     a colour-patch / LED detector for the glove's wrist pod.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Marker:
    id: int  # tag ID (AprilTag) or 0 for single-marker types
    center: tuple[float, float]  # pixels
    corners: np.ndarray  # (N, 2) outline in pixels (4 for a tag)
    size_px: float  # mean side length, a rough distance cue
    score: float  # detector-specific confidence (AprilTag: decision margin)


class MarkerDetector(abc.ABC):
    """Interface: frame (BGR) -> markers that identify the user."""

    @abc.abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Marker]: ...


class AprilTagMarkerDetector(MarkerDetector):
    def __init__(
        self,
        family: str = "tag36h11",
        ids: Sequence[int] | None = None,
        quad_decimate: float = 1.0,
        min_decision_margin: float = 30.0,
        threads: int = 2,
    ) -> None:
        from pupil_apriltags import Detector

        self._det = Detector(families=family, nthreads=threads, quad_decimate=quad_decimate)
        self.ids = set(ids) if ids else None
        self.min_margin = min_decision_margin

    def detect(self, frame_bgr: np.ndarray) -> list[Marker]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        out = []
        for r in self._det.detect(np.ascontiguousarray(gray)):
            if self.ids is not None and r.tag_id not in self.ids:
                continue
            if r.decision_margin < self.min_margin:
                continue
            c = np.asarray(r.corners, np.float32)
            side = float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
            out.append(
                Marker(
                    int(r.tag_id),
                    (float(r.center[0]), float(r.center[1])),
                    c,
                    side,
                    float(r.decision_margin),
                )
            )
        return out


_REGISTRY: dict[str, type[MarkerDetector]] = {"apriltag": AprilTagMarkerDetector}
_PLANNED = {"color_patch": "glove wrist pod colour patch / LED (not implemented yet)"}


def make_marker_detector(cfg: Mapping) -> MarkerDetector:
    """Build the detector named by cfg["type"] with its own sub-section as kwargs."""
    kind = cfg["type"]
    if kind in _PLANNED:
        raise NotImplementedError(f"marker type {kind!r} is planned: {_PLANNED[kind]}")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown marker type {kind!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**dict(cfg.get(kind, {})))
