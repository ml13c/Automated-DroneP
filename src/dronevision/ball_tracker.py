"""Tracker for THE ball in play (soccer use case): one target, lock on, stay on it.

Priors (soccer): exactly one ball is in play and it is moving; ball designs vary;
per-frame recall can be well below 100 %. So instead of trusting every detection:

SEARCHING  candidate tracklets are built from (low-confidence) ball detections. A tracklet is
           acquired once it has been seen `acquire_hits` times AND has moved `acquire_motion`
           ball diameters (in play). Tracklets that sit still for `static_s` become "static
           spots" (lamps, round signs, a spare ball on the sideline) and are ignored there.
TRACKING   alpha-beta filter (steady-state Kalman) on the ball centre. Each frame, only the
           detection nearest the prediction inside the gate is used; everything else is
           ignored, which removes most false positives. Through misses it coasts (velocity
           decays) and the gate widens; after `max_coast_s` it goes back to SEARCHING.

Distances are in ball diameters so the same settings work near and far. Times in seconds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

BallDet = tuple[float, float, float, float]  # cx, cy, diameter (px), score


@dataclass
class _Candidate:
    pos: tuple[float, float]
    diam: float
    first_t: float
    last_t: float
    start: tuple[float, float]
    hits: int = 1
    max_disp: float = 0.0  # in diameters, from the first sighting
    history: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)


@dataclass
class BallState:
    state: str  # "searching" | "tracking"
    center: tuple[float, float] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s
    diam: float = 0.0
    visible: bool = False  # matched a detection this frame
    coast_s: float = 0.0
    trail: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _theil_sen_velocity(hist: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """Median of pairwise velocities (px/s): one stray detection can't skew the initial
    velocity the way an end-to-end difference can."""
    vx, vy = [], []
    for i in range(len(hist)):
        for j in range(i + 1, len(hist)):
            dt = hist[j][0] - hist[i][0]
            if dt > 0:
                vx.append((hist[j][1] - hist[i][1]) / dt)
                vy.append((hist[j][2] - hist[i][2]) / dt)
    if not vx:
        return (0.0, 0.0)
    vx.sort()
    vy.sort()
    return (vx[len(vx) // 2], vy[len(vy) // 2])


class BallTracker:
    def __init__(self, cfg: Mapping) -> None:
        self.c = dict(cfg)
        self.reset()
        self.static_spots: list[tuple[tuple[float, float], float, float]] = []  # pos, diam, t

    def reset(self) -> None:
        self.state = "searching"
        self.candidates: list[_Candidate] = []
        self.pos: tuple[float, float] | None = None
        self.vel = (0.0, 0.0)
        self.diam = 0.0
        self.last_t = self.last_update = 0.0
        self.trail: list[tuple[float, float, float]] = []

    # ---- helpers --------------------------------------------------------------------------
    def _in_static_spot(self, d: BallDet, t: float) -> bool:
        return any(
            _dist((d[0], d[1]), p) <= self.c["candidate_gate"] * max(diam, d[2])
            for p, diam, _ in self.static_spots
        )

    def predicted(self, t: float) -> tuple[float, float] | None:
        if self.pos is None:
            return None
        dt = max(t - self.last_t, 0.0)
        return (self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt)

    # ---- searching ------------------------------------------------------------------------
    def _update_candidates(self, dets: Sequence[BallDet], t: float) -> None:
        """Tracklets for unclaimed detections; still ones become static spots. Runs while
        searching AND while tracking (so a spare ball seen after acquisition is learned too)."""
        c = self.c
        self.static_spots = [s for s in self.static_spots if t - s[2] <= c["static_memory_s"]]
        self.candidates = [k for k in self.candidates if t - k.last_t <= c["candidate_max_age_s"]]
        dets = [d for d in dets if not self._in_static_spot(d, t)]
        used: set[int] = set()
        for k in sorted(self.candidates, key=lambda k: -k.hits):
            best = None
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = _dist((d[0], d[1]), k.pos) / max(k.diam, d[2], 1e-6)
                if dist <= c["candidate_gate"] and (best is None or dist < best[0]):
                    best = (dist, i)
            if best is not None:
                d = dets[best[1]]
                used.add(best[1])
                k.pos, k.diam, k.last_t, k.hits = (d[0], d[1]), d[2], t, k.hits + 1
                k.history.append((t, d[0], d[1]))
                k.max_disp = max(k.max_disp, _dist(k.pos, k.start) / max(k.diam, 1e-6))
        for i, d in enumerate(dets):
            if i not in used:
                k = _Candidate((d[0], d[1]), d[2], t, t, (d[0], d[1]))
                k.history.append((t, d[0], d[1]))
                self.candidates.append(k)

        keep = []
        for k in self.candidates:
            if t - k.first_t >= c["static_s"] and k.max_disp < c["static_motion"]:
                self.static_spots.append((k.pos, k.diam, t))
            else:
                keep.append(k)
        self.candidates = keep

    def _search(self, dets: Sequence[BallDet], t: float) -> None:
        c = self.c
        self._update_candidates(dets, t)
        ready = [
            k
            for k in self.candidates
            if k.hits >= c["acquire_hits"] and k.max_disp >= c["acquire_motion"]
        ]
        if ready:
            k = max(ready, key=lambda k: (k.hits, k.max_disp))
            self.state, self.pos, self.diam = "tracking", k.pos, k.diam
            self.vel = _theil_sen_velocity(k.history)
            self.last_t = self.last_update = t
            self.candidates = []
            self.trail = [(t, *k.pos)]

    # ---- tracking -------------------------------------------------------------------------
    def _track(self, dets: Sequence[BallDet], t: float) -> bool:
        c = self.c
        dt = max(t - self.last_t, 1e-6)
        pred = self.predicted(t)
        coast = t - self.last_update
        gate = max(c["gate_diam"] * self.diam, c["gate_min_px"]) * (1 + c["gate_growth"] * coast)
        dets = [d for d in dets if not self._in_static_spot(d, t)]
        best = None
        for d in dets:
            dist = _dist((d[0], d[1]), pred)
            if dist <= gate and (best is None or dist < best[0]):
                best = (dist, d)
        if best is None:
            self.pos = pred
            if coast > dt and c["coast_vel_halflife_s"] > 0:  # missed more than this frame
                f = 0.5 ** (dt / c["coast_vel_halflife_s"])
                self.vel = (self.vel[0] * f, self.vel[1] * f)
            self.last_t = t
            if t - self.last_update > c["max_coast_s"]:
                self.reset()
            return False
        d = best[1]
        a, b = c["smoothing"], c["vel_smoothing"]
        rx, ry = d[0] - pred[0], d[1] - pred[1]
        meas_dt = max(t - self.last_update, 1e-6)  # residual built up since the last detection
        self.pos = (pred[0] + a * rx, pred[1] + a * ry)
        self.vel = (self.vel[0] + b * rx / meas_dt, self.vel[1] + b * ry / meas_dt)
        self.diam = (1 - a) * self.diam + a * d[2]
        self.last_t = self.last_update = t
        self.trail.append((t, *self.pos))
        self.trail = [p for p in self.trail if t - p[0] <= c["trail_s"]]
        return True

    # ---- main entry -----------------------------------------------------------------------
    def update(self, dets: Sequence[BallDet], t: float) -> BallState:
        """dets: this frame's ball detections (cx, cy, diameter, score)."""
        dets = [d for d in dets if d[3] >= self.c["low_conf"]]
        if self.state == "searching":
            self._search(dets, t)
            if self.state == "searching":
                return BallState("searching")
            return self._status(t, visible=True)
        before = list(dets)
        visible = self._track(dets, t)
        # learn static spots from the detections the ball did not claim
        claimed = self.pos if visible else None
        rest = [
            d
            for d in before
            if claimed is None or _dist((d[0], d[1]), claimed) > max(d[2], self.diam)
        ]
        if self.state == "tracking":
            self._update_candidates(rest, t)
        if self.state == "searching":
            return BallState("searching")
        return self._status(t, visible)

    def _status(self, t: float, visible: bool) -> BallState:
        return BallState(
            "tracking",
            self.pos,
            self.vel,
            self.diam,
            visible,
            t - self.last_update,
            list(self.trail),
        )


def ball_roi(
    state: BallState, frame_w: int, frame_h: int, cfg: Mapping
) -> tuple[int, int, int, int] | None:
    """Square crop (x, y, w, h) around the ball's predicted position for the next frame, or
    None while searching (use the full frame). Clipped/shifted to stay inside the frame."""
    if state.state != "tracking" or state.center is None:
        return None
    lead = cfg["lead_s"]
    cx = state.center[0] + state.velocity[0] * lead
    cy = state.center[1] + state.velocity[1] * lead
    side = max(cfg["size_diam"] * state.diam, cfg["min_px"])
    side *= 1 + cfg["coast_growth"] * state.coast_s
    side = int(min(side, frame_w, frame_h))
    x = int(min(max(cx - side / 2, 0), frame_w - side))
    y = int(min(max(cy - side / 2, 0), frame_h - side))
    return (x, y, side, side)
