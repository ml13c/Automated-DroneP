import copy
import time

import pytest

from dronevision.config import load_config
from dronevision.controller import PID, Controller, ControlLoop, Target, image_errors

CFG = load_config("controller")
W, H = 640, 480
DT = 1 / CFG["rate_hz"]
SP = CFG["modes"]["follow_hand"]["vx"]["setpoint"]


def tgt(cx=W / 2, cy=H / 2, size=SP * H) -> Target:
    return Target(cx, cy, size, W, H)


def settle(ctrl, mode, target, steps=60):
    cmd = None
    for k in range(steps):
        cmd = ctrl.step(mode, target, k * DT)
    return cmd


def test_image_errors_signs():
    e = image_errors(Target(W, 0, SP * H / 2, W, H), setpoint=SP)
    assert e["x"] == pytest.approx(1) and e["y"] == pytest.approx(1)
    assert e["size"] == pytest.approx(0.5)  # half the set size -> too far -> positive


def test_centred_target_at_setpoint_gives_zero():
    cmd = settle(Controller(CFG), "follow_hand", tgt())
    assert cmd.as_dict() == pytest.approx(dict.fromkeys(cmd.as_dict(), 0.0))


@pytest.mark.parametrize(
    ("target", "axis", "sign"),
    [
        (tgt(cx=W * 0.9), "vy", +1),  # hand right of centre -> move right until centred
        (tgt(cx=W * 0.1), "vy", -1),
        (tgt(cy=H * 0.1), "vz", +1),  # hand high -> climb
        (tgt(size=SP * H * 0.5), "vx", +1),  # hand small (far) -> forward
        (tgt(size=SP * H * 1.6), "vx", -1),  # hand big (close) -> back off
    ],
)
def test_follow_hand_axis_signs(target, axis, sign):
    cmd = settle(Controller(CFG), "follow_hand", target).as_dict()
    assert sign * cmd[axis] > 0
    assert all(v == 0 for a, v in cmd.items() if a != axis)


def test_track_ball_ball_ahead_moves_forward_and_turns():
    cmd = settle(Controller(CFG), "track_ball", Target(W * 0.8, H * 0.2, 14, W, H))
    assert cmd.vx > 0 and cmd.yaw_rate > 0 and cmd.vz == 0 and cmd.vy == 0


def test_limits_and_slew():
    cfg = copy.deepcopy(CFG)
    for p in cfg["modes"]["follow_hand"].values():
        p["kp"] *= 10  # saturate every axis
    ctrl = Controller(cfg)
    far = tgt(cx=W, cy=0, size=1)  # max error everywhere
    first = ctrl.step("follow_hand", far, 0.0)
    assert abs(first.vy) <= CFG["slew"]["vy"] * DT + 1e-9  # ramps, no jump
    cmd = settle(ctrl, "follow_hand", far)
    for a, lim in CFG["limits"].items():
        assert abs(cmd.as_dict()[a]) <= lim + 1e-9
    assert cmd.vz == pytest.approx(CFG["limits"]["vz"])
    assert cmd.vy == pytest.approx(CFG["limits"]["vy"])


def test_deadband():
    db = CFG["modes"]["follow_hand"]["vy"]["deadband"]
    cmd = settle(Controller(CFG), "follow_hand", tgt(cx=W / 2 * (1 + db * 0.5)))
    assert cmd.vy == 0


def test_hover_and_lost_target_ramp_to_zero_land_descends():
    ctrl = Controller(CFG)
    settle(ctrl, "follow_hand", tgt(cx=W))
    assert settle(ctrl, "follow_hand", None).as_dict() == pytest.approx(
        dict.fromkeys(("vx", "vy", "vz", "yaw_rate"), 0.0)
    )
    settle(ctrl, "follow_hand", tgt(cx=W))
    assert settle(ctrl, "hover", tgt(cx=W)).vy == 0  # hover ignores the target
    assert settle(ctrl, "land", None).vz == pytest.approx(CFG["land"]["vz"])


def test_pid_integral_is_clamped():
    pid = PID(kp=0, ki=1.0, i_limit=0.3)
    for _ in range(100):
        out = pid.step(1.0, 0.1)
    assert out == pytest.approx(0.3)


def test_unknown_axis_rejected():
    cfg = copy.deepcopy(CFG)
    cfg["modes"]["follow_hand"]["roll"] = {"error": "x", "kp": 1}
    with pytest.raises(ValueError):
        Controller(cfg)


def test_control_loop_stale_target_holds():
    now = [0.0]
    loop = ControlLoop.from_config(CFG, clock=lambda: now[0])
    loop.post("follow_hand", tgt(cx=W))
    for _ in range(10):
        now[0] += DT
        cmd = loop.tick()
    assert cmd.vy > 0
    for _ in range(40):  # no new target for 2 s -> stale -> ramps back to zero
        now[0] += DT
        cmd = loop.tick()
    assert cmd.vy == 0


def test_control_loop_runs_at_its_own_rate():
    """Real thread, real clock: ~20 Hz regardless of how often post() is called."""
    with ControlLoop.from_config(CFG) as loop:
        loop.post("follow_hand", tgt(cx=W))
        time.sleep(0.5)
    expected = 0.5 * CFG["rate_hz"]
    assert expected * 0.6 <= loop.ticks <= expected * 1.4 + 1
