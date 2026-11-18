"""Single-user lock: decide which tracked hand belongs to the user.

    SEARCHING --marker near a confirmed hand track--> ACQUIRING
    ACQUIRING --same track nearest the marker for `acquire_frames` sightings--> LOCKED
    LOCKED    --holds the track ID; marker no longer needed--> stays LOCKED while the tracker
                keeps the track alive (it coasts through blur/dropouts up to max_coast_s)
    LOCKED    --tracker drops the track--> SEARCHING (needs the marker again)

The marker is only for acquisition, so a small wrist marker that is readable at close range is
enough: the user steps back after locking and the hand track carries the lock. While LOCKED
the caller can skip marker detection entirely (`needs_marker`).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dronevision.markers import Marker
from dronevision.tracker import Box, Track


class LockState(enum.Enum):
    SEARCHING = "searching"
    ACQUIRING = "acquiring"
    LOCKED = "locked"


@dataclass(frozen=True)
class LockStatus:
    state: LockState
    track_id: int | None  # locked (or candidate while acquiring) track
    visible: bool  # locked track matched a detection this frame (False = coasting)
    coast_s: float  # how long the locked track has gone without a detection
    progress: float  # acquiring progress 0..1


def marker_hand_distance(marker_xy: tuple[float, float], box: Box) -> float:
    """Distance from the marker centre to the hand box, in units of hand-box size (0 inside)."""
    x, y = marker_xy
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    size = max(x2 - x1, y2 - y1, 1e-6)
    return float((dx * dx + dy * dy) ** 0.5 / size)


class UserLock:
    def __init__(self, acquire_frames: int, max_marker_dist: float) -> None:
        self.acquire_frames, self.max_marker_dist = acquire_frames, max_marker_dist
        self.state = LockState.SEARCHING
        self.track_id: int | None = None
        self._streak = 0

    @classmethod
    def from_config(cls, cfg: Mapping) -> UserLock:
        return cls(cfg["acquire_frames"], cfg["max_marker_dist"])

    @property
    def needs_marker(self) -> bool:
        return self.state is not LockState.LOCKED

    def reset(self) -> None:
        self.state, self.track_id, self._streak = LockState.SEARCHING, None, 0

    def _candidate(self, tracks: Sequence[Track], markers: Sequence[Marker]) -> int | None:
        best: tuple[float, int] | None = None
        for m in markers:
            for tr in tracks:
                if not tr.confirmed:
                    continue
                d = marker_hand_distance(m.center, tr.box)
                if d <= self.max_marker_dist and (best is None or d < best[0]):
                    best = (d, tr.id)
        return best[1] if best else None

    def update(
        self, tracks: Sequence[Track], markers: Sequence[Marker] | None, t: float
    ) -> LockStatus:
        """tracks: the tracker's live tracks; markers: this frame's markers (None = not run)."""
        by_id = {tr.id: tr for tr in tracks}

        if self.state is LockState.LOCKED:
            tr = by_id.get(self.track_id)
            if tr is not None:
                return LockStatus(LockState.LOCKED, tr.id, tr.updated_now, tr.coast_s(t), 1.0)
            self.reset()  # tracker dropped the track: need the marker again

        cand = self._candidate(tracks, markers or [])
        if cand is None:
            self.reset()
        elif cand == self.track_id:
            self._streak += 1
        else:
            self.track_id, self._streak = cand, 1
        if cand is not None:
            self.state = LockState.ACQUIRING
            if self._streak >= self.acquire_frames:
                self.state = LockState.LOCKED
                tr = by_id[cand]
                return LockStatus(LockState.LOCKED, cand, tr.updated_now, tr.coast_s(t), 1.0)
        return LockStatus(
            self.state, self.track_id, False, 0.0, min(self._streak / self.acquire_frames, 1.0)
        )
