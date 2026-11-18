"""Hand tracker: Kalman box tracks + ByteTrack-style two-stage association (numpy only).

Why own code instead of Ultralytics' tracker: it runs on the Pi without torch/ultralytics, and
time-based coasting (seconds, not frames) keeps behaviour identical at 15 or 30 FPS.

- Each track: constant-velocity Kalman filter on (cx, cy, w, h); dt from frame timestamps.
- Stage 1: confident detections (>= high_thr) matched to predicted boxes by IoU.
- Stage 2: weak detections (low_thr..high_thr), typically blurred hands, can only extend
  tracks left over from stage 1. They never start a track.
- Tracks coast (prediction only) through missed frames for up to max_coast_s. While coasting,
  the predicted velocity decays (a hidden hand has probably slowed) and the box used for
  matching grows with coast time, so a hand that reappears near the prediction keeps its ID.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(iou: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """Pairs (row, col) by descending IoU, each row/col used once, IoU >= thr."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    iou = iou.copy()
    while True:
        r, c = np.unravel_index(int(np.argmax(iou)), iou.shape)
        if iou[r, c] < thr:
            return pairs
        pairs.append((int(r), int(c)))
        iou[r, :] = -1
        iou[:, c] = -1


def _xyxy_to_z(b: Box) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], np.float64)


@dataclass
class Track:
    id: int
    x: np.ndarray  # state: cx, cy, w, h, vx, vy, vw, vh
    P: np.ndarray
    last_update: float
    last_predict: float
    score: float
    hits: int = 1
    confirmed: bool = False
    updated_now: bool = True  # matched to a detection in the latest update() call
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def box(self) -> Box:
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def coast_s(self, now: float) -> float:
        return now - self.last_update


class HandTracker:
    def __init__(
        self,
        high_thr: float,
        low_thr: float,
        match_iou: float,
        min_hits: int,
        max_coast_s: float,
        std_pos: float,
        std_vel: float,
        coast_vel_halflife_s: float,
        coast_box_growth: float,
        coast_grace_s: float,
    ) -> None:
        self.high_thr, self.low_thr, self.match_iou = high_thr, low_thr, match_iou
        self.min_hits, self.max_coast_s = min_hits, max_coast_s
        self.std_pos, self.std_vel = std_pos, std_vel
        self.coast_vel_halflife_s, self.coast_box_growth = coast_vel_halflife_s, coast_box_growth
        self.coast_grace_s = coast_grace_s
        self.tracks: list[Track] = []
        self._next_id = 1

    @classmethod
    def from_config(cls, cfg: Mapping) -> HandTracker:
        return cls(
            **{
                k: cfg[k]
                for k in (
                    "high_thr",
                    "low_thr",
                    "match_iou",
                    "min_hits",
                    "max_coast_s",
                    "std_pos",
                    "std_vel",
                    "coast_vel_halflife_s",
                    "coast_box_growth",
                    "coast_grace_s",
                )
            }
        )

    # ---- Kalman ---------------------------------------------------------------------------
    def _new_track(self, box: Box, score: float, t: float) -> Track:
        z = _xyxy_to_z(box)
        h = max(z[3], 1.0)
        x = np.concatenate([z, np.zeros(4)])
        std = np.array([2 * self.std_pos * h] * 4 + [10 * self.std_vel * h] * 4)
        tr = Track(self._next_id, x, np.diag(std**2), t, t, score)
        tr.confirmed = self.min_hits <= 1
        self._next_id += 1
        return tr

    def _vel_level(self, coast: float) -> float:
        """Velocity multiplier after `coast` s without detections: 1 within the grace period,
        then halving every coast_vel_halflife_s."""
        if self.coast_vel_halflife_s <= 0 or coast <= self.coast_grace_s:
            return 1.0
        return 0.5 ** ((coast - self.coast_grace_s) / self.coast_vel_halflife_s)

    def _effective_dt(self, a: float, b: float) -> float:
        """Integral of _vel_level over coast times [a, b]: how far (in seconds at the original
        speed) the box moves between two predictions."""
        g, hl = self.coast_grace_s, self.coast_vel_halflife_s
        if hl <= 0:
            return b - a
        full = max(0.0, min(b, g) - a)
        lo = max(a, g)
        if b <= lo:
            return full
        tau = hl / np.log(2)
        return full + tau * (self._vel_level(lo) - self._vel_level(b))

    def _predict(self, tr: Track, t: float) -> None:
        dt = max(t - tr.last_predict, 0.0)
        if dt == 0:
            return
        a, b = tr.last_predict - tr.last_update, t - tr.last_update  # coast at start / end
        tr.last_predict = t
        level_a = self._vel_level(a)
        eff = self._effective_dt(a, b) / level_a  # stored velocity is already at level_a
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * eff * 30.0  # velocities are per 1/30 s: keeps noise scales sane
        h = max(tr.x[3], 1.0)
        q = np.array([self.std_pos * h] * 4 + [self.std_vel * h] * 4) * max(dt * 30.0, 1e-3)
        tr.x = F @ tr.x
        tr.x[4:] *= self._vel_level(b) / level_a
        tr.x[2:4] = np.maximum(tr.x[2:4], 1.0)
        tr.P = F @ tr.P @ F.T + np.diag(q**2)

    def _update(self, tr: Track, box: Box, score: float, t: float) -> None:
        z = _xyxy_to_z(box)
        H = np.hstack([np.eye(4), np.zeros((4, 4))])
        h = max(tr.x[3], 1.0)
        R = np.diag((np.array([self.std_pos * h] * 4)) ** 2)
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - H @ tr.x)
        tr.P = (np.eye(8) - K @ H) @ tr.P
        tr.last_update, tr.score, tr.updated_now = t, score, True
        tr.hits += 1
        if tr.hits >= self.min_hits:
            tr.confirmed = True

    def _match_box(self, tr: Track, t: float) -> Box:
        """Predicted box, enlarged for matching in proportion to how long it has coasted."""
        cx, cy, w, h = tr.x[:4]
        k = 1.0 + self.coast_box_growth * max(tr.coast_s(t), 0.0)
        return (cx - k * w / 2, cy - k * h / 2, cx + k * w / 2, cy + k * h / 2)

    # ---- main entry -----------------------------------------------------------------------
    def update(self, boxes: Sequence[Box], scores: Sequence[float], t: float) -> list[Track]:
        """Feed this frame's hand detections (any confidence >= low_thr); returns live tracks."""
        for tr in self.tracks:
            self._predict(tr, t)
            tr.updated_now = False
        boxes_a = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores_a = np.asarray(scores, np.float64).reshape(-1)
        high = np.flatnonzero(scores_a >= self.high_thr)
        low = np.flatnonzero((scores_a >= self.low_thr) & (scores_a < self.high_thr))

        track_boxes = np.array([self._match_box(tr, t) for tr in self.tracks], np.float64).reshape(
            -1, 4
        )
        # stage 1: confident detections
        m1 = greedy_match(iou_matrix(track_boxes, boxes_a[high]), self.match_iou)
        for ti, di in m1:
            self._update(self.tracks[ti], tuple(boxes_a[high[di]]), scores_a[high[di]], t)
        left_tracks = [i for i in range(len(self.tracks)) if i not in {ti for ti, _ in m1}]
        # stage 2: weak detections may only extend remaining tracks
        m2 = greedy_match(iou_matrix(track_boxes[left_tracks], boxes_a[low]), self.match_iou)
        for ti, di in m2:
            tr = self.tracks[left_tracks[ti]]
            self._update(tr, tuple(boxes_a[low[di]]), scores_a[low[di]], t)
        # unmatched confident detections start tentative tracks
        used_high = {di for _, di in m1}
        for k, di in enumerate(high):
            if k not in used_high:
                self.tracks.append(self._new_track(tuple(boxes_a[di]), scores_a[di], t))
        # drop: tentative tracks missed once, any track coasting too long
        self.tracks = [
            tr
            for tr in self.tracks
            if (tr.confirmed or tr.updated_now) and tr.coast_s(t) <= self.max_coast_s
        ]
        for tr in self.tracks:
            if tr.updated_now:
                tr.history.append(((tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2))
                del tr.history[:-60]
        return self.tracks

    def confirmed(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.confirmed]

    def get(self, track_id: int) -> Track | None:
        return next((tr for tr in self.tracks if tr.id == track_id), None)
