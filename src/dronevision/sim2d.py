"""2D top-down drone sim driven by the controller's velocity command (configs/sim.yaml).

`Sim2D.step()` integrates a kinematic toy model: each command axis (vx forward, vy right,
vz up, yaw_rate clockwise) is followed with a first-order lag, the body velocity is rotated
into the world frame (x east, y north, heading clockwise from north) and integrated. Landing
descends to the ground and stays there; leaving `land` takes off again to the start altitude.

`camera_target()` closes the loop the way the real drone does. On the drone the camera moves
with the body, so flying towards the hand re-centres it in the image and the drone stops. A
laptop webcam never moves, so its image alone would say "keep going" forever. Instead the
webcam view places the user's hand in the sim world (the webcam stands where the drone
started, facing the user, so at the start both views agree), and the hand is re-projected into
the SIM drone's camera from wherever the drone is now. The controller steers on that view.

`render()` draws the panel with pygame on an off-screen surface and returns a BGR image, so it
can be shown next to the camera frame in the same OpenCV window and recorded with it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from dronevision.controller import AXES, Target, VelocityCommand, image_errors


@dataclass
class DroneState:
    x: float = 0.0  # m, east
    y: float = 0.0  # m, north
    z: float = 0.0  # m, altitude
    heading: float = 0.0  # deg, clockwise from north
    vel: dict[str, float] = field(default_factory=lambda: dict.fromkeys(AXES, 0.0))
    landed: bool = False
    taking_off: bool = False
    t: float = 0.0
    trail: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)


class Sim2D:
    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = DroneState(z=float(self.cfg["start_altitude_m"]))
        self.hand: tuple[float, float, float] | None = None  # world (x, y, z) of the user's hand

    # ---- the drone's own camera view of the user's hand ---------------------------------
    def camera_target(self, webcam: Target, setpoint: float) -> Target:
        """Where the hand seen in the (fixed) webcam would appear in the sim drone's camera.

        The webcam stands at the drone's start pose (origin, facing north, start altitude) with
        the user `user_distance_m` ahead; a hand of the set size is at that distance. Its image
        offset and size give its world position, which is projected into the drone's camera
        (same field of view, fixed to the body)."""
        c = self.cfg
        tan_h = math.tan(math.radians(c["camera_hfov_deg"]) / 2)
        tan_v = tan_h * webcam.frame_h / webcam.frame_w
        e = image_errors(webcam)
        s_w = max(webcam.size / webcam.frame_h, 1e-3)
        dist = c["user_distance_m"] * setpoint / s_w  # webcam -> hand, from its apparent size
        hx = e["x"] * dist * tan_h
        hz = c["start_altitude_m"] + e["y"] * dist * tan_v
        self.hand = (hx, dist, hz)

        s = self.state
        h = math.radians(s.heading)
        rx, ry = hx - s.x, dist - s.y
        fwd = max(rx * math.sin(h) + ry * math.cos(h), c["min_view_distance_m"])
        right = rx * math.cos(h) - ry * math.sin(h)
        ex, ey = right / (fwd * tan_h), (hz - s.z) / (fwd * tan_v)
        w, hh = webcam.frame_w, webcam.frame_h
        size = s_w * dist / fwd * hh  # apparent size falls with distance
        return Target(w / 2 * (1 + ex), hh / 2 * (1 - ey), size, w, hh)

    def step(self, cmd: VelocityCommand, dt: float, mode: str = "") -> DroneState:
        s, c = self.state, self.cfg
        dt = max(dt, 0.0)
        s.t += dt
        want = cmd.as_dict()
        if s.landed and mode != "land":  # leaving land: take off again
            s.landed, s.taking_off = False, True
        if s.landed or s.taking_off:
            want = dict.fromkeys(AXES, 0.0)
        if s.taking_off:
            want["vz"] = float(c["takeoff_speed_mps"])
            if s.z >= c["start_altitude_m"]:
                s.taking_off = False
        k = min(dt / c["response_tau_s"], 1.0) if c["response_tau_s"] > 0 else 1.0
        for a in AXES:
            s.vel[a] += (want[a] - s.vel[a]) * k

        s.heading = (s.heading + s.vel["yaw_rate"] * dt) % 360.0
        h = math.radians(s.heading)
        fwd, right = s.vel["vx"], s.vel["vy"]
        s.x += (fwd * math.sin(h) + right * math.cos(h)) * dt
        s.y += (fwd * math.cos(h) - right * math.sin(h)) * dt
        s.z = min(max(s.z + s.vel["vz"] * dt, 0.0), c["max_altitude_m"])
        if s.z <= 0.0 and mode == "land":
            s.landed = True
            s.vel = dict.fromkeys(AXES, 0.0)

        s.trail.append((s.t, s.x, s.y))
        keep = c["view"]["trail_s"]
        while s.trail and s.t - s.trail[0][0] > keep:
            s.trail.pop(0)
        return s

    # ---- drawing ------------------------------------------------------------------------
    def render(
        self,
        height: int,
        mode: str = "",
        target_err_x: float | None = None,
        cmd: VelocityCommand | None = None,
    ) -> np.ndarray:
        """Panel image (BGR, height x width). target_err_x: the controller's normalised x error
        (-1..1) of the current target, drawn as a ray inside the camera's field of view."""
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        if not pygame.font.get_init():
            pygame.font.init()
        v = self.cfg["view"]
        width = int(v["width_px"]) or height
        surf = pygame.Surface((width, height))
        surf.fill((24, 28, 34))
        s = self.state
        ppm = width / v["metres_across"]  # pixels per metre
        cx, cy = width / 2, height / 2

        def to_px(x: float, y: float) -> tuple[float, float]:
            return (cx + (x - s.x) * ppm, cy - (y - s.y) * ppm)

        # grid (world-fixed, so it scrolls as the drone moves)
        g = v["grid_m"]
        half_w, half_h = width / 2 / ppm, height / 2 / ppm
        gx = math.floor((s.x - half_w) / g) * g
        while gx <= s.x + half_w:
            px = to_px(gx, 0)[0]
            pygame.draw.line(surf, (45, 52, 62), (px, 0), (px, height))
            gx += g
        gy = math.floor((s.y - half_h) / g) * g
        while gy <= s.y + half_h:
            py = to_px(0, gy)[1]
            pygame.draw.line(surf, (45, 52, 62), (0, py), (width, py))
            gy += g

        # flown path
        if len(s.trail) > 1:
            pygame.draw.lines(surf, (90, 160, 255), False, [to_px(x, y) for _, x, y in s.trail], 2)

        h = math.radians(s.heading)
        fwd = (math.sin(h), -math.cos(h))  # screen coords (y down)
        right = (math.cos(h), math.sin(h))

        # the user (standing where the webcam looks) and their hand
        ux, uy = to_px(0.0, float(self.cfg["user_distance_m"]))
        pygame.draw.circle(surf, (120, 200, 120), (ux, uy), 9, 2)
        label = pygame.font.Font(None, 18).render("user", True, (120, 200, 120))
        surf.blit(label, (ux + 12, uy - 8))
        if self.hand is not None:
            hx, hy = to_px(self.hand[0], self.hand[1])
            pygame.draw.circle(surf, (60, 230, 60), (hx, hy), 5)
            pygame.draw.line(surf, (60, 160, 60), (ux, uy), (hx, hy), 1)

        # camera field of view and the target bearing inside it
        half = math.radians(self.cfg["camera_hfov_deg"] / 2)
        reach = min(width, height) * 0.42
        for sgn in (-1, 1):
            a = h + sgn * half
            pygame.draw.line(
                surf,
                (70, 80, 95),
                (cx, cy),
                (cx + math.sin(a) * reach, cy - math.cos(a) * reach),
                1,
            )
        if target_err_x is not None:
            a = h + max(-1.0, min(1.0, target_err_x)) * half
            end = (cx + math.sin(a) * reach, cy - math.cos(a) * reach)
            pygame.draw.line(surf, (255, 170, 0), (cx, cy), end, 2)
            pygame.draw.circle(surf, (255, 170, 0), end, 6, 2)

        # drone: triangle pointing along the heading; size shrinks a little with altitude
        size = 14 * (0.8 + 0.4 * min(s.z / max(self.cfg["start_altitude_m"], 1e-6), 1.5))
        nose = (cx + fwd[0] * size, cy + fwd[1] * size)
        left = (
            cx - fwd[0] * size * 0.7 - right[0] * size * 0.7,
            cy - fwd[1] * size * 0.7 - right[1] * size * 0.7,
        )
        rgt = (
            cx - fwd[0] * size * 0.7 + right[0] * size * 0.7,
            cy - fwd[1] * size * 0.7 + right[1] * size * 0.7,
        )
        body = (120, 120, 120) if s.landed else (240, 240, 240)
        pygame.draw.polygon(surf, body, [nose, left, rgt])

        # velocity vector (world frame), 1 m/s = 1 m on the map
        vxw = s.vel["vx"] * math.sin(h) + s.vel["vy"] * math.cos(h)
        vyw = s.vel["vx"] * math.cos(h) - s.vel["vy"] * math.sin(h)
        if abs(vxw) + abs(vyw) > 0.02:
            pygame.draw.line(surf, (255, 255, 0), (cx, cy), (cx + vxw * ppm, cy - vyw * ppm), 3)

        # altitude bar (right edge)
        bx, top, bot = width - 22, 40, height - 40
        pygame.draw.line(surf, (90, 90, 90), (bx, top), (bx, bot), 2)
        alt_y = bot - (bot - top) * s.z / self.cfg["max_altitude_m"]
        pygame.draw.line(surf, (0, 255, 255), (bx - 8, alt_y), (bx + 8, alt_y), 4)

        font = pygame.font.Font(None, max(18, height // 26))
        small = pygame.font.Font(None, max(16, height // 32))
        title = "LANDED" if s.landed else (mode.replace("_", " ").upper() or "SIM")
        surf.blit(font.render(f"SIM  {title}", True, (255, 255, 255)), (10, 8))
        lines = [
            f"alt {s.z:4.1f} m   heading {s.heading:5.1f} deg",
            f"pos ({s.x:+.1f}, {s.y:+.1f}) m",
        ]
        if cmd is not None:
            lines.append(
                f"cmd vx {cmd.vx:+.2f} vy {cmd.vy:+.2f} vz {cmd.vz:+.2f} yaw {cmd.yaw_rate:+.0f}"
            )
        for i, text in enumerate(lines):
            surf.blit(
                small.render(text, True, (200, 200, 200)), (10, height - 22 * (len(lines) - i))
            )
        # scale bar: 1 grid cell
        pygame.draw.line(surf, (200, 200, 200), (10, 40), (10 + g * ppm, 40), 2)
        surf.blit(small.render(f"{g:g} m", True, (200, 200, 200)), (14 + g * ppm, 33))

        rgb = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
        return np.ascontiguousarray(rgb[:, :, ::-1])
