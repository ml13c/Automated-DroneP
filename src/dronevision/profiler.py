"""Timing and usage measurement.

A frame-rate meter; per-stage timing and CPU/RAM/GPU sampling will live here too.
"""

from __future__ import annotations

from collections import deque


class FpsMeter:
    """Measured FPS over the last ``window`` frames: (frames - 1) / elapsed seconds.

    Timestamps come from the caller (normally ``Frame.timestamp``), which keeps this
    deterministic in tests and lets replayed logs reuse the same code.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, timestamp: float) -> float | None:
        """Record a frame time; return current FPS, or None until 2 frames are seen."""
        self._times.append(timestamp)
        return self.fps

    @property
    def fps(self) -> float | None:
        if len(self._times) < 2:
            return None
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return None
        return (len(self._times) - 1) / elapsed
