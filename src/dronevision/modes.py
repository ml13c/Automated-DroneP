"""Mode state machine: finger counts (vision) and glove commands -> flight mode.

Rules (configs/modes.yaml):
- Vision: a count must be held `hold_s` with probability >= `min_prob`. Brief gaps (< max_gap_s)
  of missing/uncertain readings don't reset the hold; a different count does.
- Glove: acts immediately (a deliberate press). The emergency command "land" always acts.
- Mode lock: every mode locks when it is entered. While locked, only the unlock command
  ("1") and the always-allowed modes (land) are accepted. Unlock -> the next command picks a
  mode and locks again; with no command within `unlock_timeout_s` it locks again unchanged.
  Leaving `land` works the same way: unlock, then pick a mode.
- Failsafes force a mode (and lock it), but never override `land`:
  - User lost: > hover_after_s -> hover, > land_after_s -> land, only in `applies_in_modes`
    (not in track_ball, where the user being out of frame is normal).
  - Glove link lost (no heartbeat): > hover_after_s -> hover, > land_after_s -> land, in ALL
    modes. While the link is down, vision can still command land but nothing else.
  - Ball lost: in track_ball, the ball tracker searching for > hover_after_s -> hover
    (counted from entering the mode at the earliest).

Pure logic with explicit timestamps, so every transition is unit-tested without a camera.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

LAND = "land"
HOVER = "hover"
UNLOCK = "unlock"


@dataclass(frozen=True)
class ModeStatus:
    mode: str
    pending: str | None  # count currently being held (vision)
    hold_progress: float  # 0..1 for the pending count
    reason: str  # why the mode (or lock) last changed
    locked: bool = True
    unlock_left_s: float = 0.0  # while unlocked: time left to pick a mode


class ModeMachine:
    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg
        self.commands = {str(k): v for k, v in cfg["commands"].items()}
        lock = cfg["lock"]
        self.unlock_cmd = str(lock["unlock_command"])
        if self.unlock_cmd in self.commands:
            raise ValueError(f"unlock command {self.unlock_cmd!r} is also mapped to a mode")
        self.always_allowed = set(lock["always_allowed"])
        if cfg["glove"]["hold_s"] > 0:
            raise NotImplementedError("glove.hold_s > 0 (glove hold-to-confirm) is not implemented")
        self.reset()

    def reset(self, t: float = 0.0) -> None:
        self.mode = self.cfg["initial_mode"]
        self.reason = "start"
        self._mode_since = t
        self.locked = bool(self.cfg["lock"]["start_locked"])
        self._unlocked_at = t
        self._pending: str | None = None
        self._since = 0.0
        self._last_ok = 0.0

    # ---- helpers ------------------------------------------------------------------------
    def action(self, command: str | None) -> str | None:
        """What a command does: a mode name, UNLOCK, or None (unmapped)."""
        if command is None:
            return None
        command = str(command)
        if command == self.unlock_cmd:
            return UNLOCK
        return self.commands.get(command)

    def _allowed(self, action: str | None, link_down: bool) -> bool:
        if action is None:
            return False
        if link_down:  # forced hover: only land gets through
            return action == LAND
        return action == UNLOCK or not self.locked or action in self.always_allowed

    def _set(self, mode: str, reason: str, t: float) -> None:
        """Enter `mode` (if different) and lock it."""
        if mode != self.mode:
            self.mode, self.reason, self._mode_since = mode, reason, t
        elif not self.locked:
            self.reason = reason
        self.locked = True
        self._pending = None

    def _apply(self, action: str, reason: str, t: float) -> None:
        if action == UNLOCK:
            self.locked, self._unlocked_at, self.reason = False, t, f"{reason}: unlocked"
            self._pending = None
        else:
            self._set(action, reason, t)

    def _status(self, t: float, progress: float = 0.0) -> ModeStatus:
        left = 0.0
        if not self.locked:
            left = max(self.cfg["lock"]["unlock_timeout_s"] - (t - self._unlocked_at), 0.0)
        return ModeStatus(
            self.mode,
            self._pending,
            progress if self._pending is not None else 0.0,
            self.reason,
            self.locked,
            left,
        )

    # ---- main entry ---------------------------------------------------------------------
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
        link = self.cfg["glove_link"]
        link_down = glove_lost_s > link["hover_after_s"]

        # 1. emergency land (glove) and glove-link land: always
        if glove == LAND:
            self._set(LAND, "glove: emergency land", t)
            return self._status(t)
        if glove_lost_s > link["land_after_s"]:
            self._set(LAND, f"glove link lost > {link['land_after_s']:g} s", t)
            return self._status(t)

        # 2. glove commands act immediately (they only arrive while the link is up)
        act = self.action(glove)
        if self._allowed(act, link_down):
            self._apply(act, f"glove: {glove}", t)

        # 3. failsafes (never override land)
        if self.mode != LAND:
            lost = self.cfg["user_lost"]
            applies = self.mode in lost["applies_in_modes"]
            if applies and user_lost_s > lost["land_after_s"]:
                self._set(LAND, f"user lost > {lost['land_after_s']:g} s", t)
                return self._status(t)
            if link_down and (self.mode != HOVER or not self.locked):
                self._set(HOVER, f"glove link lost > {link['hover_after_s']:g} s", t)
            if applies and user_lost_s > lost["hover_after_s"] and self.mode != HOVER:
                self._set(HOVER, f"user lost > {lost['hover_after_s']:g} s", t)
            bl = self.cfg["ball_lost"]
            if (
                self.mode in bl["applies_in_modes"]
                and min(ball_lost_s, t - self._mode_since) > bl["hover_after_s"]
            ):
                self._set(HOVER, f"ball lost > {bl['hover_after_s']:g} s", t)

        # 4. unlock timeout: nothing picked in time -> lock again, mode unchanged
        if (
            not self.locked
            and t - self._unlocked_at > self.cfg["lock"]["unlock_timeout_s"]
            and self._pending in (None, self.unlock_cmd)
        ):
            self.locked, self.reason, self._pending = True, "no command: locked again", None

        # 5. vision hold-to-confirm
        v = self.cfg["vision"]
        label = None
        if finger is not None and finger[1] >= v["min_prob"]:
            act = self.action(finger[0])
            # an unlock while already unlocked is a no-op, not a new hold
            if self._allowed(act, link_down) and not (act == UNLOCK and not self.locked):
                label = str(finger[0])
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
                held = self._pending
                self._apply(self.action(held), f"vision: {held} held {v['hold_s']:g} s", t)
        return self._status(t, progress)
