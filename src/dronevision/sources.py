"""Pluggable frame sources.

Every pipeline stage consumes :class:`Frame` objects from a :class:`FrameSource`, so a
Picamera2 / IMX500 source can be added later by subclassing ``FrameSource`` without touching
anything downstream.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
}


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR, HxWx3, uint8
    index: int  # 0-based frame counter for this source
    timestamp: float  # time.perf_counter() seconds at capture


class FrameSource(abc.ABC):
    """Base class: ``read()`` returns the next Frame, or None when the stream ends."""

    name: str = "source"

    @abc.abstractmethod
    def read(self) -> Frame | None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[int, int]:
        """(width, height) actually delivered by the device/file."""

    def __iter__(self) -> Iterator[Frame]:
        while (frame := self.read()) is not None:
            yield frame

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _OpenCVSource(FrameSource):
    def __init__(self, cap: cv2.VideoCapture, name: str) -> None:
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open {name}")
        self._cap = cap
        self._index = 0
        self.name = name

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = Frame(image=image, index=self._index, timestamp=time.perf_counter())
        self._index += 1
        return frame

    def close(self) -> None:
        self._cap.release()

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def reported_fps(self) -> float:
        """FPS the driver/file *claims*; measure real FPS with profiler.FpsMeter."""
        return float(self._cap.get(cv2.CAP_PROP_FPS))


class WebcamSource(_OpenCVSource):
    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        backend: str = "auto",
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; choose from {sorted(_BACKENDS)}")
        cap = cv2.VideoCapture(index, _BACKENDS[backend])
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        super().__init__(cap, f"webcam {index}")


class VideoFileSource(_OpenCVSource):
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")
        super().__init__(cv2.VideoCapture(str(path)), f"file {path.name}")


def parse_source_spec(spec: str | int) -> int | Path:
    """``"0"`` / ``0`` -> webcam index 0; anything else -> a file path."""
    if isinstance(spec, int):
        return spec
    spec = str(spec).strip()
    if spec.isdigit():
        return int(spec)
    return Path(spec)


def open_source(
    spec: str | int,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    backend: str = "auto",
) -> FrameSource:
    """Open a webcam (by index) or a video file (by path)."""
    target = parse_source_spec(spec)
    if isinstance(target, int):
        return WebcamSource(target, width=width, height=height, fps=fps, backend=backend)
    return VideoFileSource(target)
