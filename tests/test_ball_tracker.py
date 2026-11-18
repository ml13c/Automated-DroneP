import math
import random

import pytest

from dronevision.ball_tracker import BallState, BallTracker, ball_roi
from dronevision.config import load_config

DT = 1 / 30
CFG = load_config("tracking")


def tracker() -> BallTracker:
    return BallTracker(CFG["ball_tracker"])


def ball_path(i: int) -> tuple[float, float]:
    """Ball rolling across the frame with a gentle curve, ~120 px/s."""
    return (80 + 4.0 * i, 300 - 60 * math.sin(i / 40))


def run(frames: int, seed: int = 0, miss: float = 0.35, lamp=True, fps=True, spare=None):
    rng = random.Random(seed)
    bt, states = tracker(), []
    for i in range(frames):
        dets = []
        if rng.random() > miss:
            x, y = ball_path(i)
            dets.append((x + rng.gauss(0, 1), y + rng.gauss(0, 1), 14, rng.uniform(0.2, 0.6)))
        if lamp:
            dets.append((500, 60, 16, 0.45))  # ceiling lamp: detected every frame, never moves
        if spare:
            dets.append((*spare, 14, 0.5))  # spare ball resting on the sideline
        if fps and rng.random() < 0.15:  # random one-frame false positives
            dets.append((rng.uniform(0, 640), rng.uniform(0, 480), 12, 0.2))
        states.append(bt.update(dets, i * DT))
    return bt, states


def test_locks_moving_ball_not_lamp_and_holds_through_misses():
    bt, states = run(240)
    first = next(i for i, s in enumerate(states) if s.state == "tracking")
    assert first < 45  # acquired within ~1.5 s
    after = states[first:]
    assert all(s.state == "tracking" for s in after)  # 35 % misses never broke the lock
    errs = [math.dist(s.center, ball_path(first + k)) for k, s in enumerate(after)]
    assert max(errs) < 20 and sum(errs) / len(errs) < 6  # follows the ball, not lamp/FPs
    assert any(math.dist(p, (500, 60)) < 20 for p, _, _ in bt.static_spots)  # learned meanwhile


def test_static_object_alone_is_never_acquired():
    bt = tracker()
    for i in range(120):
        s = bt.update([(300, 200, 15, 0.9)], i * DT)
    assert s.state == "searching" and len(bt.static_spots) >= 1


def test_spare_ball_on_sideline_is_ignored_even_when_play_passes_by():
    # the spare sits right on the ball's path at frame ~100
    spare = ball_path(100)
    spare = (spare[0], spare[1] + 25)
    _, states = run(200, seed=3, miss=0.2, lamp=False, fps=False, spare=spare)
    tracked = [(i, s) for i, s in enumerate(states) if s.state == "tracking"]
    assert tracked and all(math.dist(s.center, ball_path(i)) < 20 for i, s in tracked)


def test_long_dropout_returns_to_searching():
    bt = tracker()
    for i in range(30):
        bt.update([(*ball_path(i), 14, 0.5)], i * DT)
    assert bt.state == "tracking"
    t = 30 * DT
    s = None
    for k in range(int(CFG["ball_tracker"]["max_coast_s"] / DT) + 3):
        s = bt.update([], t + k * DT)
    assert s.state == "searching"


def test_roi_contains_prediction_and_stays_in_frame():
    roi_cfg = CFG["ball_roi"]
    s = BallState("tracking", (630, 10), (300, 0), 14, True, 0.0)
    x, y, w, h = ball_roi(s, 640, 480, roi_cfg)
    assert w == h == max(roi_cfg["size_diam"] * 14, roi_cfg["min_px"])
    assert x >= 0 and x + w <= 640 and y >= 0 and y + h <= 480
    big = ball_roi(BallState("tracking", (320, 240), (0, 0), 30, False, 1.0), 640, 480, roi_cfg)
    assert big[2] == pytest.approx(min(30 * roi_cfg["size_diam"] * 2.5, 480), abs=1)
    assert ball_roi(BallState("searching"), 640, 480, roi_cfg) is None


def test_robust_over_many_random_scenes():
    """35 % misses + a lamp + random false positives, 40 seeds: the lock is never lost and
    the worst excursion stays under 3 ball diameters (no runaway)."""
    worst, errs = 0.0, []
    for seed in range(40):
        _, st = run(240, seed=seed)
        first = next(i for i, s in enumerate(st) if s.state == "tracking")
        assert all(s.state == "tracking" for s in st[first:])
        e = [math.dist(s.center, ball_path(first + k)) for k, s in enumerate(st[first:])]
        worst, errs = max(worst, max(e)), errs + e
    assert worst < 3 * 14 and sum(errs) / len(errs) < 3
