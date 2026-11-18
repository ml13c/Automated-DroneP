"""Mode + target in the image -> velocity command (vx, vy, vz, yaw_rate).

`Controller` is the pure part: one PID per axis on a normalised image error (box centre x/y,
box size), then clamps and slew limits. Gains and limits are in configs/controller.yaml.

`ControlLoop` runs a `Controller` in its own thread at a fixed rate (20 Hz), separate from the
camera loop, like the flight-controller link will on the Pi: the camera loop only posts the
latest mode/target; the control loop always sends a command on time and falls back to "hold"
(zero velocity) when the target goes stale.

Command frame: vx forward +, vy right +, vz up +, yaw_rate clockwise (right) +, in m/s and
deg/s. Image errors are measured on the camera's own (un-mirrored) image.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

AXES = ("vx", "vy", "vz", "yaw_rate")


@dataclass(frozen=True)
class Target:
    """What to keep centred: centre and size (box height or ball diameter) in px."""

    cx: float
    cy: float
    size: float
    frame_w: int
    frame_h: int


@dataclass(frozen=True)
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {a: getattr(self, a) for a in AXES}


def image_errors(target: Target, setpoint: float | None = None) -> dict[str, float]:
    """Normalised errors (about -1..1): x right +, y up +, size too-small (far) +."""
    hw, hh = target.frame_w / 2, target.frame_h / 2
    err = {"x": (target.cx - hw) / hw, "y": (hh - target.cy) / hh}
    if setpoint:
        s = target.size / target.frame_h
        err["size"] = (setpoint - s) / setpoint
    return err


class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        deadband: float = 0.0,
        i_limit: float = float("inf"),
    ) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.deadband, self.i_limit = deadband, i_limit
        self.reset()

    def reset(self) -> None:
        self._i = 0.0
        self._prev: float | None = None

    def step(self, err: float, dt: float) -> float:
        if abs(err) < self.deadband:
            err = 0.0
        d = 0.0 if self._prev is None or dt <= 0 else (err - self._prev) / dt
        self._prev = err
        if self.ki:
            self._i += self.ki * err * dt
            self._i = max(-self.i_limit, min(self.i_limit, self._i))
        return self.kp * err + self._i + self.kd * d


class Controller:
    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg
        self.limits = {a: float(cfg["limits"][a]) for a in AXES}
        self.slew = {a: float(cfg["slew"][a]) for a in AXES}
        self.pids: dict[str, dict[str, PID]] = {}
        self.axes: dict[str, dict[str, Mapping]] = {}
        for mode, axes in cfg["modes"].items():
            self.axes[mode] = dict(axes)
            self.pids[mode] = {
                a: PID(
                    p["kp"],
                    p.get("ki", 0.0),
                    p.get("kd", 0.0),
                    p.get("deadband", 0.0),
                    p.get("i_limit", float("inf")),
                )
                for a, p in axes.items()
            }
            bad = set(axes) - set(AXES)
            if bad:
                raise ValueError(f"controller.modes.{mode}: unknown axes {sorted(bad)}")
        self.reset()

    def reset(self) -> None:
        self.cmd = VelocityCommand()
        self._mode: str | None = None
        self._last_t: float | None = None
        for pids in self.pids.values():
            for pid in pids.values():
                pid.reset()

    def _raw(self, mode: str, target: Target | None, dt: float) -> dict[str, float]:
        raw = dict.fromkeys(AXES, 0.0)
        if mode == "land":
            raw["vz"] = float(self.cfg["land"]["vz"])
        elif mode in self.axes and target is not None:
            for axis, p in self.axes[mode].items():
                err = image_errors(target, p.get("setpoint"))[p["error"]]
                raw[axis] = self.pids[mode][axis].step(err, dt)
        elif mode in self.pids:  # no target: hold, and don't wind up
            for pid in self.pids[mode].values():
                pid.reset()
        return raw

    def step(self, mode: str, target: Target | None, t: float) -> VelocityCommand:
        """One control tick at time t (s). target=None -> hold position (zero velocity)."""
        dt = 1.0 / self.cfg["rate_hz"] if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        if mode != self._mode:
            for pid in self.pids.get(mode, {}).values():
                pid.reset()
            self._mode = mode
        raw = self._raw(mode, target, dt)
        prev = self.cmd.as_dict()
        out = {}
        for a in AXES:
            v = max(-self.limits[a], min(self.limits[a], raw[a]))
            step = self.slew[a] * dt
            out[a] = max(prev[a] - step, min(prev[a] + step, v))
        self.cmd = VelocityCommand(**out)
        return self.cmd


class ControlLoop:
    """Runs a Controller at `rate_hz` in a background thread.

    The camera loop calls `post(mode, target)` whenever it has a new frame; `latest` is the
    command most recently sent. Targets older than `target_timeout_s` are treated as gone."""

    def __init__(
        self,
        controller: Controller,
        rate_hz: float | None = None,
        target_timeout_s: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
        on_command: Callable[[float, VelocityCommand], None] | None = None,
    ) -> None:
        cfg = controller.cfg
        self.controller = controller
        self.period = 1.0 / (rate_hz or cfg["rate_hz"])
        self.timeout = cfg["target_timeout_s"] if target_timeout_s is None else target_timeout_s
        self.clock, self.on_command = clock, on_command
        self._lock = threading.Lock()
        self._mode = "hover"
        self._target: Target | None = None
        self._target_t = float("-inf")
        self.latest = VelocityCommand()
        self.ticks = 0
        self.late_ticks = 0  # ticks that started more than one period late
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t_start: float | None = None
        self.t_stop: float | None = None

    @classmethod
    def from_config(cls, cfg: Mapping, **kw) -> ControlLoop:
        return cls(Controller(cfg), **kw)

    def post(self, mode: str, target: Target | None) -> None:
        with self._lock:
            self._mode = mode
            if target is not None:
                self._target, self._target_t = target, self.clock()

    def tick(self) -> VelocityCommand:
        """One control step now (the thread calls this; tests can call it directly)."""
        now = self.clock()
        with self._lock:
            mode = self._mode
            fresh = now - self._target_t <= self.timeout
            target = self._target if fresh else None
        cmd = self.controller.step(mode, target, now)
        self.latest = cmd
        self.ticks += 1
        if self.on_command is not None:
            self.on_command(now, cmd)
        return cmd

    def _run(self) -> None:
        next_t = self.clock()
        while not self._stop.is_set():
            self.tick()
            next_t += self.period
            delay = next_t - self.clock()
            if delay < -self.period:  # fell behind (e.g. the OS paused us): don't burst
                self.late_ticks += 1
                next_t = self.clock()
            elif delay > 0:
                self._stop.wait(delay)

    @property
    def measured_hz(self) -> float:
        """Achieved tick rate between start() and stop() (or now)."""
        end = self.t_stop if self.t_stop is not None else self.clock()
        return self.ticks / (end - self.t_start) if self.t_start is not None else 0.0

    def start(self) -> ControlLoop:
        self.t_start, self.t_stop = self.clock(), None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="control-loop", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.t_stop = self.clock()

    def __enter__(self) -> ControlLoop:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
