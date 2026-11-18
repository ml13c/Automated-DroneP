import pytest

from dronevision.config import load_config
from dronevision.controller import Controller, Target, VelocityCommand, image_errors
from dronevision.sim2d import Sim2D

CFG = load_config("sim")
DT = 0.05


def run(sim, cmd, seconds, mode="follow_hand"):
    for _ in range(round(seconds / DT)):
        sim.step(cmd, DT, mode)
    return sim.state


def test_starts_hovering_at_start_altitude():
    s = Sim2D(CFG).state
    assert s.z == CFG["start_altitude_m"] and (s.x, s.y, s.heading) == (0, 0, 0)


def test_forward_moves_north_then_east_after_turning_right():
    sim = Sim2D(CFG)
    s = run(sim, VelocityCommand(vx=1.0), 5.0)
    assert s.y > 3.5 and abs(s.x) < 1e-6  # heading 0 = north
    s = run(sim, VelocityCommand(yaw_rate=45.0), 2.0)  # turn right (clockwise), ~72 deg after lag
    assert 60 < s.heading < 100
    x0 = s.x
    s = run(sim, VelocityCommand(vx=1.0), 3.0)
    assert s.x - x0 > 1.5  # now flying roughly east


def test_lateral_right_is_east_at_heading_zero():
    s = run(Sim2D(CFG), VelocityCommand(vy=1.0), 3.0)
    assert s.x > 1.5 and abs(s.y) < 1e-6


def test_command_is_followed_with_lag():
    sim = Sim2D(CFG)
    sim.step(VelocityCommand(vx=1.0), DT)
    assert 0 < sim.state.vel["vx"] < 1.0
    s = run(sim, VelocityCommand(vx=1.0), 5 * CFG["response_tau_s"])
    assert s.vel["vx"] == pytest.approx(1.0, abs=0.02)


def test_land_reaches_ground_stays_then_takes_off_again():
    sim = Sim2D(CFG)
    s = run(sim, VelocityCommand(vz=-0.4), 15.0, mode="land")
    assert s.landed and s.z == 0
    s = run(sim, VelocityCommand(vx=1.0), 1.0, mode="land")  # ignores commands on the ground
    assert s.landed and s.vel["vx"] == 0
    s = run(sim, VelocityCommand(), 20.0, mode="hover")  # re-armed: climbs back up
    assert not s.landed and s.z >= CFG["start_altitude_m"] - 0.05


def test_altitude_is_clamped():
    s = run(Sim2D(CFG), VelocityCommand(vz=1.0), 30.0)
    assert s.z == CFG["max_altitude_m"]


def test_render_returns_bgr_panel():
    pytest.importorskip("pygame")
    sim = Sim2D(CFG)
    run(sim, VelocityCommand(vx=0.5, yaw_rate=20), 2.0)
    img = sim.render(360, "track_ball", target_err_x=0.4, cmd=VelocityCommand(vx=0.5))
    assert img.shape == (360, 360, 3) and img.dtype.name == "uint8"
    assert img.flags["C_CONTIGUOUS"] and img.std() > 5  # something was drawn


# ---- closed loop: the drone's own camera moves with it ----------------------------------

CTRL = load_config("controller")
SP = CTRL["modes"]["follow_hand"]["vx"]["setpoint"]
W, H = 640, 480


def webcam(cx=W / 2, cy=H / 2, size=SP * H):
    return Target(cx, cy, size, W, H)


def fly(sim, ctrl, cam, seconds, t0=0.0):
    """Webcam view fixed at `cam`; controller steers on the sim drone's own camera view."""
    t, view = t0, None
    for _ in range(round(seconds / DT)):
        view = sim.camera_target(cam, SP)
        cmd = ctrl.step("follow_hand", view, t)
        sim.step(cmd, DT, "follow_hand")
        t += DT
    return view, t


def test_camera_target_matches_webcam_at_start():
    sim = Sim2D(CFG)
    cam = webcam(cx=W * 0.3, cy=H * 0.6, size=SP * H * 0.8)
    v = sim.camera_target(cam, SP)
    assert (v.cx, v.cy, v.size) == pytest.approx((cam.cx, cam.cy, cam.size), rel=1e-6)


def test_hand_left_moves_drone_left_until_centred_then_stops():
    sim, ctrl = Sim2D(CFG), Controller(CTRL)
    cam = webcam(cx=W * 0.2)  # hand held left of centre (camera's view) and kept there
    view, t = fly(sim, ctrl, cam, 0.5)
    assert ctrl.cmd.vy < -0.2  # moving left (its left = towards the hand)
    view, _ = fly(sim, ctrl, cam, 12.0, t)
    assert abs(image_errors(view)["x"]) < 0.08  # hand centred in the drone's own camera
    assert abs(ctrl.cmd.vy) < 0.05  # and it has stopped
    assert sim.state.x == pytest.approx(sim.hand[0], abs=0.25)  # right in front of the hand


def test_hand_pushed_closer_drone_backs_off_to_set_distance():
    sim, ctrl = Sim2D(CFG), Controller(CTRL)
    y0 = sim.state.y
    view, _ = fly(sim, ctrl, webcam(size=SP * H * 1.5), 12.0)  # hand looks bigger: closer
    assert sim.state.y < y0 - 0.3  # backed away
    assert view.size / H == pytest.approx(SP, rel=0.15)  # hand back at the set size


def test_hand_back_to_centre_brings_drone_back():
    sim, ctrl = Sim2D(CFG), Controller(CTRL)
    _, t = fly(sim, ctrl, webcam(cx=W * 0.8), 10.0)
    assert sim.state.x > 0.5
    fly(sim, ctrl, webcam(), 12.0, t)
    assert sim.state.x == pytest.approx(0.0, abs=0.25)
