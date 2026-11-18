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


_USER = (60, 230, 60)  # BGR
_USER_COAST = (0, 215, 255)
_IGNORED = (150, 150, 150)
_MARKER = (255, 0, 255)


def _label(image: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    y = max(th + 4, y)
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(image, text, (x + 2, y - 3), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tracks(image: np.ndarray, tracks, status, now: float) -> np.ndarray:
    """Hand tracks: the user's (locked) hand bold green, coasting yellow; others grey 'ignored'."""
    locked = status is not None and status.state.value == "locked"
    for tr in tracks:
        x1, y1, x2, y2 = (int(v) for v in tr.box)
        if locked and tr.id == status.track_id:
            color = _USER if tr.updated_now else _USER_COAST
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            text = f"USER #{tr.id}" + ("" if tr.updated_now else f" coast {tr.coast_s(now):.1f}s")
            if len(tr.history) > 1:
                pts = np.array(tr.history, np.int32).reshape(-1, 1, 2)
                cv2.polylines(image, [pts], False, color, 2, cv2.LINE_AA)
        else:
            color = _IGNORED
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            candidate = status is not None and tr.id == status.track_id
            text = f"#{tr.id} " + ("acquiring" if candidate else "ignored")
        _label(image, text, x1, y1, color)
    return image


def draw_markers(image: np.ndarray, markers) -> np.ndarray:
    for m in markers:
        cv2.polylines(image, [m.corners.astype(np.int32).reshape(-1, 1, 2)], True, _MARKER, 2)
        _label(image, f"tag {m.id}", int(m.center[0]), int(m.center[1]), _MARKER)
    return image


_BALL = (0, 160, 255)
_BALL_COAST = (0, 230, 255)


def draw_ball(image: np.ndarray, state, roi=None) -> np.ndarray:
    """Locked ball (circle + trail; lighter while coasting) and the next-frame ROI box."""
    if state is None or state.state != "tracking" or state.center is None:
        return image
    color = _BALL if state.visible else _BALL_COAST
    if len(state.trail) > 1:
        pts = np.array([(x, y) for _, x, y in state.trail], np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], False, color, 2, cv2.LINE_AA)
    c = (int(state.center[0]), int(state.center[1]))
    cv2.circle(image, c, max(int(state.diam / 2) + 3, 6), color, 2, cv2.LINE_AA)
    text = "BALL" + ("" if state.visible else f" coast {state.coast_s:.1f}s")
    _label(image, text, c[0] + 8, c[1] - 8, color)
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    return image
