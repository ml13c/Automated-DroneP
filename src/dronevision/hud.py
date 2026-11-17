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
