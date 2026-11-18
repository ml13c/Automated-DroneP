"""On-screen overlay. M0: a stats panel (FPS, source). Later: boxes, mode, velocity arrow."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_stats(
    image: np.ndarray,
    stats: Mapping[str, str],
    origin: tuple[int, int] = (10, 10),
    scale: float = 0.6,
) -> np.ndarray:
    """Draw ``key: value`` lines on a translucent dark panel, in place. Returns ``image``."""
    if not stats:
        return image
    lines = [f"{k}: {v}" for k, v in stats.items()]
    thickness = 1
    sizes = [cv2.getTextSize(line, _FONT, scale, thickness)[0] for line in lines]
    line_h = max(h for _, h in sizes) + 8
    pad = 6
    x0, y0 = origin
    x1 = min(image.shape[1] - 1, x0 + max(w for w, _ in sizes) + 2 * pad)
    y1 = min(image.shape[0] - 1, y0 + line_h * len(lines) + pad)

    roi = image[y0:y1, x0:x1]
    if roi.size:
        roi[:] = (roi * 0.4).astype(image.dtype)

    for i, line in enumerate(lines):
        y = y0 + pad + line_h * (i + 1) - 6
        cv2.putText(
            image, line, (x0 + pad, y), _FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
    return image


_CLASS_COLORS = {"hand": (80, 220, 80), "ball": (0, 160, 255)}  # BGR


def draw_detections(image: np.ndarray, detections, thickness: int = 2) -> np.ndarray:
    """Boxes + `name score` labels for dronevision.detector.Detection objects, in place."""
    for d in detections:
        color = _CLASS_COLORS.get(d.name, (255, 80, 80))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(image, p1, p2, color, thickness)
        label = f"{d.name} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
        y = max(th + 4, p1[1])
        cv2.rectangle(image, (p1[0], y - th - 4), (p1[0] + tw + 4, y), color, -1)
        cv2.putText(image, label, (p1[0] + 2, y - 3), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return image
