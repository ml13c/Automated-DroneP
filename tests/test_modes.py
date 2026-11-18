"""Every mode transition (PLAN.md M5 done-when: unit tests cover every mode transition)."""

import copy

import pytest

from dronevision.config import load_config
from dronevision.glove_sim import KeyboardGlove
from dronevision.modes import ModeMachine

CFG = load_config("modes")
DT = 1 / 30
HOLD = CFG["vision"]["hold_s"]


def machine(**over):
    cfg = copy.deepcopy(CFG)
    cfg.update(over)
    return ModeMachine(cfg)


def hold(m, label, seconds, t0=0.0, prob=0.95, **kw):
    """Feed the same finger reading every frame for `seconds`; returns (last status, end t)."""
    t, st = t0, None
    for _ in range(round(seconds / DT) + 1):
        st = m.update(t, finger=(label, prob), **kw)
        t += DT
    return st, t


def test_starts_in_initial_mode():
    assert machine().update(0.0).mode == CFG["initial_mode"] == "hover"


@pytest.mark.parametrize(
    ("label", "mode"), [("5", "follow_hand"), ("3", "track_ball"), ("4", "land")]
)
def test_vision_hold_switches_mode(label, mode):
    st, _ = hold(machine(), label, HOLD + DT)
    assert st.mode == mode and st.reason.startswith("vision")


def test_short_hold_does_not_switch_and_reports_progress():
    st, _ = hold(machine(), "5", HOLD * 0.6)
    assert st.mode == "hover" and st.pending == "5" and 0.5 < st.hold_progress < 0.7


def test_changing_count_restarts_the_hold():
    m = machine()
    _, t = hold(m, "5", HOLD * 0.6)
    st, t = hold(m, "3", HOLD * 0.6, t0=t)
    assert st.mode == "hover" and st.pending == "3"
    st, _ = hold(m, "3", HOLD * 0.5, t0=t)
    assert st.mode == "track_ball"


def test_low_probability_is_ignored():
    st, _ = hold(machine(), "5", HOLD * 2, prob=CFG["vision"]["min_prob"] - 0.05)
    assert st.mode == "hover" and st.pending is None


def test_brief_dropout_keeps_hold_long_gap_resets():
    m = machine()
    _, t = hold(m, "5", HOLD * 0.5)
    m.update(t, finger=None)  # one missed frame (< max_gap_s)
    st, _ = hold(m, "5", HOLD * 0.6, t0=t + DT)
    assert st.mode == "follow_hand"
    m = machine()
    _, t = hold(m, "5", HOLD * 0.5)
    for k in range(10):  # 0.33 s of nothing (> max_gap_s)
        st = m.update(t + k * DT, finger=None)
    assert st.pending is None
    st, _ = hold(m, "5", HOLD * 0.6, t0=t + 10 * DT)
    assert st.mode == "hover"


@pytest.mark.parametrize("label", ["0", "1", "none"])
def test_unmapped_counts_do_nothing(label):
    st, _ = hold(machine(), label, HOLD * 2)
    assert st.mode == "hover" and st.pending is None


def test_land_is_latched_against_vision_glove_and_failsafes():
    m = machine()
    _, t = hold(m, "4", HOLD + DT)
    assert m.mode == "land"
    st, t = hold(m, "5", HOLD * 2, t0=t)
    assert st.mode == "land"
    assert m.update(t, glove="3").mode == "land"
    assert m.update(t + 1, user_lost_s=0.0).mode == "land"
    m.reset()
    assert m.mode == "hover"


def test_glove_acts_immediately_and_emergency_land_beats_vision():
    m = machine()
    assert m.update(0.0, glove="3").mode == "track_ball"
    st = m.update(DT, glove="5")
    assert st.mode == "follow_hand" and st.reason == "glove: 5"
    _, t = hold(m, "3", HOLD * 0.9, t0=2 * DT)  # vision about to switch...
    st = m.update(t, finger=("3", 0.99), glove="land")  # ...emergency land wins
    assert st.mode == "land" and "emergency" in st.reason


def test_glove_unmapped_command_does_nothing():
    assert machine().update(0.0, glove="1").mode == "hover"


def test_user_lost_follow_hand_goes_hover_then_land():
    m = machine()
    _, t = hold(m, "5", HOLD + DT)
    assert m.mode == "follow_hand"
    lost = CFG["user_lost"]
    assert m.update(t, user_lost_s=lost["hover_after_s"] - 0.1).mode == "follow_hand"
    st = m.update(t + 0.1, user_lost_s=lost["hover_after_s"] + 0.1)
    assert st.mode == "hover" and "user lost" in st.reason
    assert m.update(t + 1, user_lost_s=lost["land_after_s"] + 0.1).mode == "land"


def test_user_lost_does_not_interrupt_ball_tracking():
    m = machine()
    m.update(0.0, glove="3")
    assert m.update(1.0, user_lost_s=CFG["user_lost"]["land_after_s"] + 5).mode == "track_ball"


def test_user_back_before_timeout_changes_nothing():
    m = machine()
    _, t = hold(m, "5", HOLD + DT)
    m.update(t, user_lost_s=1.5)
    assert m.update(t + DT, user_lost_s=0.0).mode == "follow_hand"


def test_keyboard_glove_map():
    kb = KeyboardGlove.from_config(CFG)
    assert kb.command(ord("3")) == "3" and kb.command(ord(" ")) == "land"
    assert kb.command(ord("x")) is None
