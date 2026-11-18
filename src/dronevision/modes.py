"""Mode state machine: finger counts (vision) and glove commands -> flight mode.

Rules (configs/modes.yaml):
- Vision: a count must be held `hold_s` with probability >= `min_prob`. Brief gaps (< max_gap_s)
  of missing/uncertain readings don't reset the hold; a different count does.
- Glove: acts after `glove.hold_s` (0 = immediately). The emergency command "land" always acts
  immediately.
- Land always wins: once in `land` (with land_latched) nothing changes until `reset()`.
- User lost: > hover_after_s -> hover, > land_after_s -> land, only in `applies_in_modes`
  (not in track_ball, where the user being out of frame is normal).
- Glove link lost (no heartbeat): > hover_after_s -> hover, > land_after_s -> land, in ALL
  modes. While the link is down, vision can still command land but nothing else.
- Ball lost: in track_ball, the ball tracker searching for > hover_after_s -> hover (counted
  from entering the mode at the earliest).

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

    def reset(self, t: float = 0.0) -> None:
        self.mode = self.cfg["initial_mode"]
        self.reason = "start"
        self._mode_since = t
        self._pending: str | None = None
        self._since = 0.0
        self._last_ok = 0.0

    def _set(self, mode: str, reason: str, t: float) -> None:
        if mode != self.mode:
            self.mode, self.reason, self._mode_since = mode, reason, t

    def _locked_land(self) -> bool:
        return self.mode == LAND and self.cfg["land_latched"]

    def update(
        self,
        t: float,
        finger: tuple[str, float] | None = None,
        glove: str | None = None,
        user_lost_s: float = 0.0,
        glove_lost_s: float = 0.0,
        ball_lost_s: float = 0.0,
    ) -> ModeStatus:
        """finger: (label, prob) of the USER's hand this frame, or None if not read.
        glove: command from the glove ("1".."5" or "land"), or None.
        user_lost_s: seconds since the locked user hand was last seen (0 = visible).
        glove_lost_s: seconds since the last glove heartbeat (0 = link up).
        ball_lost_s: seconds since the ball tracker last had the ball (0 = tracking)."""
        if self._locked_land():
            return ModeStatus(self.mode, None, 0.0, self.reason)

        # 1. emergency land, glove-link land
        if glove == LAND:
            self._set(LAND, "glove: emergency land", t)
            return ModeStatus(self.mode, None, 0.0, self.reason)
        link = self.cfg["glove_link"]
        if glove_lost_s > link["land_after_s"]:
            self._set(LAND, f"glove link lost > {link['land_after_s']:g} s", t)
            return ModeStatus(self.mode, None, 0.0, self.reason)
        link_down = glove_lost_s > link["hover_after_s"]

        # 2. glove commands (only arrive while the link is up)
        if glove is not None and not link_down and self.cfg["glove"]["hold_s"] <= 0:
            mode = self.commands.get(str(glove))
            if mode:
                self._set(mode, f"glove: {glove}", t)
                self._pending = None

        # 3. failsafes -> hover
        lost = self.cfg["user_lost"]
        applies = self.mode in lost["applies_in_modes"]
        if applies and user_lost_s > lost["land_after_s"]:
            self._set(LAND, f"user lost > {lost['land_after_s']:g} s", t)
            return ModeStatus(self.mode, None, 0.0, self.reason)
        if link_down and self.mode != HOVER:
            self._set(HOVER, f"glove link lost > {link['hover_after_s']:g} s", t)
            self._pending = None
        if applies and user_lost_s > lost["hover_after_s"] and self.mode != HOVER:
            self._set(HOVER, f"user lost > {lost['hover_after_s']:g} s", t)
            self._pending = None
        bl = self.cfg["ball_lost"]
        if (
            self.mode in bl["applies_in_modes"]
            and min(ball_lost_s, t - self._mode_since) > bl["hover_after_s"]
        ):
            self._set(HOVER, f"ball lost > {bl['hover_after_s']:g} s", t)
            self._pending = None

        # 4. vision hold-to-confirm (only land while the glove link is down)
        v = self.cfg["vision"]
        label = None
        wanted = self.commands.get(finger[0]) if finger is not None else None
        if wanted and finger[1] >= v["min_prob"] and (not link_down or wanted == LAND):
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
                self._set(mode, f"vision: {self._pending} held {v['hold_s']:g} s", t)
                if self._locked_land():
                    return ModeStatus(self.mode, None, 0.0, self.reason)
        return ModeStatus(self.mode, self._pending, progress, self.reason)
