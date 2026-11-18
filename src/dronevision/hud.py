"""On-screen overlay: stats panel, detections/tracks/markers/ball, the mode panel
(mode, hold-progress bar, glove link) and the velocity-command widget."""

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


# ---- mode panel + velocity command ---- --------------------------------------------------
_MODE_COLORS = {  # BGR
    "hover": (255, 200, 80),
    "follow_hand": (60, 230, 60),
    "track_ball": (0, 160, 255),
    "land": (60, 60, 255),
}
_OK = (60, 230, 60)
_UNLOCKED = (0, 220, 255)  # BGR yellow
_BAD = (60, 60, 255)


def _panel(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    roi = image[y0:y1, x0:x1]
    if roi.size:
        roi[:] = (roi * 0.35).astype(image.dtype)


def _text(image, text, org, scale=0.5, color=(255, 255, 255), thick=1) -> None:
    cv2.putText(image, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def draw_mode_panel(
    image: np.ndarray,
    status,
    commands: Mapping[str, str | None],
    glove_connected: bool | None = None,
    glove_lost_s: float = 0.0,
    width: int = 275,
) -> np.ndarray:
    """Top-right panel: current mode (+ why), lock state, the vision hold-progress bar, glove
    link. `commands` maps a count to what it does (a mode name or "unlock")."""
    w = image.shape[1]
    x0, y0 = w - width - 10, 10
    _panel(image, x0, y0, x0 + width, y0 + (116 if glove_connected is not None else 94))
    color = _MODE_COLORS.get(status.mode, (255, 255, 255))
    _text(image, status.mode.replace("_", " ").upper(), (x0 + 8, y0 + 24), 0.7, color, 2)
    locked = getattr(status, "locked", True)
    lock_text = "LOCKED" if locked else f"UNLOCKED {getattr(status, 'unlock_left_s', 0.0):.1f}s"
    (lw, _), _ = cv2.getTextSize(lock_text, _FONT, 0.45, 1)
    _text(
        image,
        lock_text,
        (x0 + width - lw - 8, y0 + 22),
        0.45,
        (170, 170, 170) if locked else _UNLOCKED,
        1,
    )
    _text(image, status.reason[:34], (x0 + 8, y0 + 44), 0.42, (210, 210, 210))

    # hold bar: which count is being held and how far along
    bx0, by0, bx1, by1 = x0 + 8, y0 + 54, x0 + width - 8, y0 + 68
    cv2.rectangle(image, (bx0, by0), (bx1, by1), (120, 120, 120), 1)
    if status.pending is not None:
        target = commands.get(status.pending) or "?"
        fill = int(bx0 + (bx1 - bx0) * max(0.0, min(status.hold_progress, 1.0)))
        bar = _UNLOCKED if target == "unlock" else _MODE_COLORS.get(target, color)
        cv2.rectangle(image, (bx0, by0), (fill, by1), bar, -1)
        label = f"hold {status.pending} -> {target.replace('_', ' ')}"
    elif locked:
        unlock = next((k for k, v in commands.items() if v == "unlock"), "1")
        label = f"locked: hold {unlock} to unlock"
    else:
        label = "unlocked: hold a count to pick a mode"
    _text(image, label, (bx0, by1 + 16), 0.42, (230, 230, 230))

    if glove_connected is not None:
        y = y0 + 108
        if glove_connected and glove_lost_s == 0:
            _text(image, "GLOVE LINK OK", (x0 + 8, y), 0.5, _OK, 1)
        else:
            _text(image, f"GLOVE LINK LOST {glove_lost_s:.1f} s", (x0 + 8, y), 0.5, _BAD, 2)
    return image


def draw_command(
    image: np.ndarray,
    cmd,
    limits: Mapping[str, float],
    radius: int = 55,
) -> np.ndarray:
    """Bottom-right widget: top-down drone with the horizontal velocity arrow (forward = up),
    a yaw-rate arc, a vertical-speed bar, and the numbers. Arrow/arc/bar scale to the limits."""
    h, w = image.shape[:2]
    cx, cy = w - radius - 70, h - radius - 44
    px = w - 2 * radius - 140  # panel left edge (room for the numbers)
    _panel(image, px, cy - radius - 24, w - 8, cy + radius + 36)
    _text(image, "drone command", (px + 8, cy - radius - 8), 0.45, (230, 230, 230))
    cv2.circle(image, (cx, cy), radius, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - 6), (cx, cy + 6), (200, 200, 200), 1)
    cv2.line(image, (cx - 6, cy), (cx + 6, cy), (200, 200, 200), 1)

    vx = cmd.vx / limits["vx"] if limits["vx"] else 0.0
    vy = cmd.vy / limits["vy"] if limits["vy"] else 0.0
    end = (int(cx + vy * radius), int(cy - vx * radius))
    if abs(vx) + abs(vy) > 1e-3:
        cv2.arrowedLine(image, (cx, cy), end, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.25)

    yaw = cmd.yaw_rate / limits["yaw_rate"] if limits["yaw_rate"] else 0.0
    if abs(yaw) > 1e-3:  # arc from "forward", clockwise = right
        sweep = 90 * yaw
        cv2.ellipse(
            image,
            (cx, cy),
            (radius + 6, radius + 6),
            -90,
            0,
            sweep,
            (255, 120, 255),
            3,
            cv2.LINE_AA,
        )

    # vertical speed bar on the right
    bx, top, bot = cx + radius + 22, cy - radius, cy + radius
    cv2.line(image, (bx, top), (bx, bot), (140, 140, 140), 1)
    vz = cmd.vz / limits["vz"] if limits["vz"] else 0.0
    if abs(vz) > 1e-3:
        cv2.line(image, (bx, cy), (bx, int(cy - vz * radius)), (255, 255, 0), 5)
    _text(image, "up", (bx - 8, top - 4), 0.35, (200, 200, 200))

    _text(
        image,
        f"vx {cmd.vx:+.2f} vy {cmd.vy:+.2f} vz {cmd.vz:+.2f} m/s",
        (px + 8, cy + radius + 16),
        0.38,
    )
    _text(image, f"yaw {cmd.yaw_rate:+.0f} deg/s", (px + 8, cy + radius + 30), 0.38)
    return image


def draw_target_line(image: np.ndarray, target_xy: tuple[float, float] | None) -> np.ndarray:
    """Thin line from the image centre to what the controller is centring (the error)."""
    if target_xy is None:
        return image
    h, w = image.shape[:2]
    c = (w // 2, h // 2)
    cv2.drawMarker(image, c, (230, 230, 230), cv2.MARKER_CROSS, 16, 1)
    cv2.line(image, c, (int(target_xy[0]), int(target_xy[1])), (230, 230, 230), 1, cv2.LINE_AA)
    return image
