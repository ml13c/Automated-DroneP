"""Keyboard stand-in for the touch-contact glove: keys inject the same commands the glove's
radio will send ("1".."5", "land"). Key map in configs/modes.yaml -> keyboard_glove.

The real glove sends a periodic heartbeat; the mode machine treats a missing heartbeat as
"glove link lost". Here the link toggle key (default `g`) stops/restarts the simulated
heartbeat, and no commands get through while it is off.
"""

from __future__ import annotations

from collections.abc import Mapping


class KeyboardGlove:
    def __init__(self, keys: Mapping[str, str], toggle_key: str | None = None) -> None:
        self._map = {ord(k): str(v) for k, v in keys.items()}
        self._toggle = ord(toggle_key) if toggle_key else None
        self.connected = True
        self._last_beat: float | None = None

    @classmethod
    def from_config(cls, cfg: Mapping) -> KeyboardGlove:
        kg = cfg["keyboard_glove"]
        return cls(kg["keys"], kg.get("link_toggle_key"))

    def command(self, key_code: int) -> str | None:
        """cv2.waitKey() code (already & 0xFF) -> glove command, or None (pure key lookup)."""
        return self._map.get(key_code)

    def poll(self, key_code: int, t: float) -> str | None:
        """Call once per frame with the key (or -1/255) and time: handles the link toggle,
        sends the heartbeat while connected, and returns a command only if the link is up."""
        if self.connected:
            self._last_beat = t  # the link was up until (at least) now
        if self._toggle is not None and key_code == self._toggle:
            self.connected = not self.connected
            if self.connected:
                self._last_beat = t
            return None
        return self.command(key_code) if self.connected else None

    def link_lost_s(self, t: float) -> float:
        """Seconds since the last heartbeat (0 while connected or before the first poll)."""
        if self._last_beat is None:
            return 0.0
        return max(t - self._last_beat, 0.0)
