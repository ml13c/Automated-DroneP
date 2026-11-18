"""Mode state machine: finger counts (vision) and glove commands -> flight mode.

Rules (configs/modes.yaml, PLAN.md M5):
- Vision: a count must be held `hold_s` with probability >= `min_prob`. Brief gaps (< max_gap_s)
  of missing/uncertain readings don't reset the hold; a different count does.
- Glove: acts after `glove.hold_s` (0 = immediately). The emergency command "land" always acts
  immediately.
- Land always wins: once in `land` (with land_latched) nothing changes until `reset()`.
- User lost: > hover_after_s -> hover, > land_after_s -> land, only in `applies_in_modes`
  (not in track_ball, where the user being out of frame is normal).

Pure logic with explicit timestamps, so every transition is unit-tested without a camera.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

LAND = "land"
HOVER = "hover"


@dataclass(frozen=True)
class ModeStatus:
    mode: str
    pending: str | None  # command currently being held (vision)
    hold_progress: float  # 0..1 for the pending command
    reason: str  # why the mode last changed


class ModeMachine:
    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg
        self.commands = {str(k): v for k, v in cfg["commands"].items()}
        if cfg["glove"]["hold_s"] > 0:
            raise NotImplementedError("glove.hold_s > 0 (glove hold-to-confirm) is not implemented")
        self.reset()

    def reset(self) -> None:
        self.mode = self.cfg["initial_mode"]
        self.reason = "start"
        self._pending: str | None = None
        self._since = 0.0
        self._last_ok = 0.0

    def _set(self, mode: str, reason: str) -> None:
        if mode != self.mode:
            self.mode, self.reason = mode, reason

    def _locked_land(self) -> bool:
        return self.mode == LAND and self.cfg["land_latched"]

    def update(
        self,
        t: float,
        finger: tuple[str, float] | None = None,
        glove: str | None = None,
        user_lost_s: float = 0.0,
    ) -> ModeStatus:
        """finger: (label, prob) of the USER's hand this frame, or None if not read.
        glove: command from the glove ("1".."5" or "land"), or None.
        user_lost_s: seconds since the locked user hand was last seen (0 = visible)."""
        if self._locked_land():
            return ModeStatus(self.mode, None, 0.0, self.reason)

        # 1. emergency / glove commands
        if glove == LAND:
            self._set(LAND, "glove: emergency land")
            return ModeStatus(self.mode, None, 0.0, self.reason)
        if glove is not None and self.cfg["glove"]["hold_s"] <= 0:
            mode = self.commands.get(str(glove))
            if mode:
                self._set(mode, f"glove: {glove}")
                self._pending = None

        # 2. user-lost failsafes
        lost = self.cfg["user_lost"]
        applies = self.mode in lost["applies_in_modes"]
        if applies and user_lost_s > lost["land_after_s"]:
            self._set(LAND, f"user lost > {lost['land_after_s']:g} s")
            return ModeStatus(self.mode, None, 0.0, self.reason)
        if applies and user_lost_s > lost["hover_after_s"] and self.mode != HOVER:
            self._set(HOVER, f"user lost > {lost['hover_after_s']:g} s")
            self._pending = None

        # 3. vision hold-to-confirm
        v = self.cfg["vision"]
        label = None
        if finger is not None and finger[1] >= v["min_prob"] and self.commands.get(finger[0]):
            label = finger[0]
        if label is None:
            if self._pending is not None and t - self._last_ok > v["max_gap_s"]:
                self._pending = None
        elif label != self._pending:
            self._pending, self._since, self._last_ok = label, t, t
        else:
            self._last_ok = t

        progress = 0.0
        if self._pending is not None:
            progress = min((t - self._since) / v["hold_s"], 1.0) if v["hold_s"] > 0 else 1.0
            if progress >= 1.0:
                mode = self.commands[self._pending]
                self._set(mode, f"vision: {self._pending} held {v['hold_s']:g} s")
                if self._locked_land():
                    return ModeStatus(self.mode, None, 0.0, self.reason)
        return ModeStatus(self.mode, self._pending, progress, self.reason)
