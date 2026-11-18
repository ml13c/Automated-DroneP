"""Every mode transition, including the mode lock and the failsafes."""

import copy

import pytest

from dronevision.config import load_config
from dronevision.glove_sim import KeyboardGlove
from dronevision.modes import ModeMachine

CFG = load_config("modes")
DT = 1 / 30
HOLD = CFG["vision"]["hold_s"]
UNLOCK = CFG["lock"]["unlock_command"]
TIMEOUT = CFG["lock"]["unlock_timeout_s"]
LINK = CFG["glove_link"]
BALL = CFG["ball_lost"]


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


def switch(m, label, t0=0.0):
    """Unlock (hold 1), then hold `label`: the normal way to change mode by hand."""
    _, t = hold(m, UNLOCK, HOLD + DT, t0)
    return hold(m, label, HOLD + DT, t)


def glove_to(m, key, t):
    m.update(t, glove=UNLOCK)
    return m.update(t + DT, glove=key)


# ---- lock / unlock ----------------------------------------------------------------------
def test_starts_locked_in_initial_mode():
    st = machine().update(0.0)
    assert st.mode == CFG["initial_mode"] == "hover" and st.locked


def test_locked_ignores_counts_until_unlocked():
    st, _ = hold(machine(), "5", HOLD * 3)
    assert st.mode == "hover" and st.locked and st.pending is None


def test_unlock_then_count_switches_and_locks_again():
    m = machine()
    st, t = hold(m, UNLOCK, HOLD + DT)
    assert not st.locked and st.mode == "hover" and st.unlock_left_s > 0
    st, t = hold(m, "5", HOLD + DT, t)
    assert st.mode == "follow_hand" and st.locked and st.reason.startswith("vision")
    st, _ = hold(m, "3", HOLD * 3, t)  # locked again: another count does nothing
    assert st.mode == "follow_hand" and st.pending is None


@pytest.mark.parametrize(
    ("label", "mode"), [("5", "follow_hand"), ("3", "track_ball"), ("4", "land"), ("2", "hover")]
)
def test_every_count_reaches_its_mode(label, mode):
    st, _ = switch(machine(), label)
    assert st.mode == mode and st.locked


def test_still_holding_after_switch_does_not_retrigger():
    m = machine()
    st, t = switch(m, "5")
    st, _ = hold(m, "5", HOLD * 3, t)
    assert st.mode == "follow_hand" and st.pending is None and st.reason.startswith("vision")


def test_holding_unlock_longer_does_not_restart_it():
    m = machine()
    st, _ = hold(m, UNLOCK, HOLD * 3)
    assert not st.locked and st.pending is None


def test_unlock_times_out_and_locks_again_unchanged():
    m = machine()
    _, t = switch(m, "5")
    st, t = hold(m, UNLOCK, HOLD + DT, t)
    assert not st.locked
    st = m.update(t + TIMEOUT + 0.1)
    assert st.locked and st.mode == "follow_hand" and "locked again" in st.reason


def test_land_works_while_locked_and_needs_unlock_to_leave():
    m = machine()
    _, t = switch(m, "5")
    st, t = hold(m, "4", HOLD + DT, t)  # no unlock needed for land
    assert st.mode == "land" and st.locked
    st, t = hold(m, "2", HOLD * 3, t)  # locked in land
    assert st.mode == "land"
    st, _ = switch(m, "2", t)  # unlock, then hover: back in the air
    assert st.mode == "hover" and st.locked


def test_reset_returns_to_initial_locked():
    m = machine()
    switch(m, "4")
    m.reset(10.0)
    assert m.mode == "hover" and m.locked


# ---- vision hold details ----------------------------------------------------------------
def test_short_hold_does_not_switch_and_reports_progress():
    m = machine()
    _, t = hold(m, UNLOCK, HOLD + DT)
    st, _ = hold(m, "5", HOLD * 0.6, t)
    assert st.mode == "hover" and st.pending == "5" and 0.5 < st.hold_progress < 0.7


def test_changing_count_restarts_the_hold():
    m = machine()
    _, t = hold(m, UNLOCK, HOLD + DT)
    _, t = hold(m, "5", HOLD * 0.6, t)
    st, t = hold(m, "3", HOLD * 0.6, t)
    assert st.mode == "hover" and st.pending == "3"
    st, _ = hold(m, "3", HOLD * 0.5, t)
    assert st.mode == "track_ball"


def test_low_probability_is_ignored():
    m = machine()
    _, t = hold(m, UNLOCK, HOLD + DT)
    st, _ = hold(m, "5", HOLD * 2, t, prob=CFG["vision"]["min_prob"] - 0.05)
    assert st.mode == "hover" and st.pending is None


def test_brief_dropout_keeps_hold_long_gap_resets():
    m = machine()
    _, t = hold(m, UNLOCK, HOLD + DT)
    _, t = hold(m, "5", HOLD * 0.5, t)
    m.update(t, finger=None)  # one missed frame (< max_gap_s)
    st, _ = hold(m, "5", HOLD * 0.6, t + DT)
    assert st.mode == "follow_hand"
    m = machine()
    _, t = hold(m, UNLOCK, HOLD + DT)
    _, t = hold(m, "5", HOLD * 0.5, t)
    for k in range(10):  # 0.33 s of nothing (> max_gap_s)
        st = m.update(t + k * DT, finger=None)
    assert st.pending is None
    st, _ = hold(m, "5", HOLD * 0.6, t + 10 * DT)
    assert st.mode == "hover"


@pytest.mark.parametrize("label", ["0", "none"])
def test_unmapped_counts_do_nothing(label):
    m = machine()
    _, t = hold(m, UNLOCK, HOLD + DT)
    st, _ = hold(m, label, HOLD * 2, t)
    assert st.mode == "hover" and st.pending is None


def test_unlock_command_cannot_also_be_a_mode():
    with pytest.raises(ValueError):
        machine(commands={**CFG["commands"], UNLOCK: "hover"})


# ---- glove ------------------------------------------------------------------------------
def test_glove_obeys_lock_and_acts_immediately():
    m = machine()
    assert m.update(0.0, glove="3").mode == "hover"  # locked
    st = glove_to(m, "3", 0.1)
    assert st.mode == "track_ball" and st.locked and st.reason == "glove: 3"
    assert m.update(0.3, glove="5").mode == "track_ball"  # locked again


def test_glove_emergency_land_always_wins():
    m = machine()
    _, t = hold(m, UNLOCK, HOLD + DT)
    _, t = hold(m, "3", HOLD * 0.9, t)  # vision about to switch...
    st = m.update(t, finger=("3", 0.99), glove="land")  # ...emergency land wins
    assert st.mode == "land" and "emergency" in st.reason


def test_glove_unmapped_command_does_nothing():
    m = machine()
    m.update(0.0, glove=UNLOCK)
    assert m.update(0.1, glove="0").mode == "hover"


# ---- user lost --------------------------------------------------------------------------
def test_user_lost_follow_hand_goes_hover_then_land():
    m = machine()
    _, t = switch(m, "5")
    lost = CFG["user_lost"]
    assert m.update(t, user_lost_s=lost["hover_after_s"] - 0.1).mode == "follow_hand"
    st = m.update(t + 0.1, user_lost_s=lost["hover_after_s"] + 0.1)
    assert st.mode == "hover" and "user lost" in st.reason and st.locked
    assert m.update(t + 1, user_lost_s=lost["land_after_s"] + 0.1).mode == "land"


def test_user_lost_does_not_interrupt_ball_tracking():
    m = machine()
    glove_to(m, "3", 0.0)
    assert m.update(1.0, user_lost_s=CFG["user_lost"]["land_after_s"] + 5).mode == "track_ball"


def test_user_back_before_timeout_changes_nothing():
    m = machine()
    _, t = switch(m, "5")
    m.update(t, user_lost_s=1.5)
    assert m.update(t + DT, user_lost_s=0.0).mode == "follow_hand"


# ---- glove link lost (all modes) --------------------------------------------------------
@pytest.mark.parametrize("start", ["5", "3", "2"])
def test_glove_link_lost_hovers_then_lands_in_every_mode(start):
    m = machine()
    glove_to(m, start, 0.0)
    mode = m.commands[start]
    assert m.update(0.5, glove_lost_s=LINK["hover_after_s"] - 0.1).mode == mode
    st = m.update(1.0, glove_lost_s=LINK["hover_after_s"] + 0.1)
    assert st.mode == "hover" and ("glove link" in st.reason or start == "2")
    st = m.update(5.0, glove_lost_s=LINK["land_after_s"] + 0.1)
    assert st.mode == "land" and "glove link" in st.reason


def test_glove_link_down_blocks_vision_except_land():
    m = machine()
    lost = LINK["hover_after_s"] + 0.1
    st, t = hold(m, UNLOCK, HOLD * 2, glove_lost_s=lost)
    assert st.locked and st.pending is None  # can't even unlock
    st, t = hold(m, "5", HOLD * 2, t, glove_lost_s=lost)
    assert st.mode == "hover" and st.pending is None
    st, _ = hold(m, "4", HOLD + DT, t, glove_lost_s=lost)
    assert st.mode == "land"


def test_glove_link_back_after_hover_stays_hover_until_commanded():
    m = machine()
    glove_to(m, "5", 0.0)
    assert m.update(0.5, glove_lost_s=0.5).mode == "follow_hand"
    m.update(2.0, glove_lost_s=LINK["hover_after_s"] + 0.1)
    st = m.update(2.1, glove_lost_s=0.0)
    assert st.mode == "hover" and st.locked
    assert glove_to(m, "5", 2.2).mode == "follow_hand"


def test_failsafes_never_override_land():
    m = machine()
    _, t = switch(m, "4")
    st = m.update(t, glove_lost_s=LINK["hover_after_s"] + 0.1, user_lost_s=3.0)
    assert st.mode == "land"


# ---- ball lost (track_ball) -------------------------------------------------------------
def test_ball_lost_in_track_ball_goes_hover():
    m = machine()
    glove_to(m, "3", 0.0)
    assert m.update(2.0, ball_lost_s=0.0).mode == "track_ball"
    t = 2.0 + BALL["hover_after_s"] + 0.1
    assert m.update(t - 0.2, ball_lost_s=BALL["hover_after_s"] - 0.1).mode == "track_ball"
    st = m.update(t, ball_lost_s=BALL["hover_after_s"] + 0.1)
    assert st.mode == "hover" and "ball lost" in st.reason


def test_ball_lost_counts_from_entering_track_ball():
    # the ball tracker has been searching for a long time, but the mode was just entered
    m = machine()
    glove_to(m, "3", 10.0)
    assert m.update(11.0, ball_lost_s=11.0).mode == "track_ball"
    assert m.update(10.0 + BALL["hover_after_s"] + 0.2, ball_lost_s=13.2).mode == "hover"


def test_ball_lost_ignored_outside_track_ball():
    m = machine()
    glove_to(m, "5", 0.0)
    assert m.update(20.0, ball_lost_s=20.0).mode == "follow_hand"


# ---- keyboard glove ---------------------------------------------------------------------
def test_keyboard_glove_map():
    kb = KeyboardGlove.from_config(CFG)
    assert kb.command(ord("3")) == "3" and kb.command(ord(" ")) == "land"
    assert kb.command(ord("x")) is None


def test_keyboard_glove_link_toggle():
    kb = KeyboardGlove.from_config(CFG)
    g = ord(CFG["keyboard_glove"]["link_toggle_key"])
    assert kb.poll(ord("3"), 0.0) == "3" and kb.link_lost_s(0.5) == pytest.approx(0.5)
    assert kb.poll(-1, 1.0) is None and kb.link_lost_s(1.0) == 0.0
    kb.poll(g, 2.0)  # link drops: no more heartbeats or commands
    assert not kb.connected
    assert kb.poll(ord("5"), 3.0) is None and kb.link_lost_s(3.0) == pytest.approx(1.0)
    kb.poll(g, 4.0)  # link back
    assert kb.connected and kb.link_lost_s(4.0) == 0.0 and kb.poll(ord("5"), 4.1) == "5"
