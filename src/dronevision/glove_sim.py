"""Keyboard stand-in for the touch-contact glove: keys inject the same commands the glove's
radio will send ("1".."5", "land"). Key map in configs/modes.yaml -> keyboard_glove.keys."""

from __future__ import annotations

from collections.abc import Mapping


class KeyboardGlove:
    def __init__(self, keys: Mapping[str, str]) -> None:
        self._map = {ord(k): str(v) for k, v in keys.items()}

    @classmethod
    def from_config(cls, cfg: Mapping) -> KeyboardGlove:
        return cls(cfg["keyboard_glove"]["keys"])

    def command(self, key_code: int) -> str | None:
        """cv2.waitKey() code (already & 0xFF) -> glove command, or None."""
        return self._map.get(key_code)
