import cv2
import numpy as np
import pytest

from dronevision.config import load_config
from dronevision.markers import (
    AprilTagMarkerDetector,
    Marker,
    MarkerDetector,
    make_marker_detector,
)
from dronevision.tracker import HandTracker, greedy_match, iou_matrix
from dronevision.user_lock import LockState, UserLock, marker_hand_distance

DT = 1 / 30


def tracker(**kw) -> HandTracker:
    cfg = dict(load_config("tracking")["hand_tracker"])
    cfg.update(kw)
    return HandTracker.from_config(cfg)


def box_at(cx, cy, s=40):
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


# ---------------------------------------------------------------------------------- tracker
def test_iou_and_greedy_match():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    b = np.array([[21, 21, 31, 31], [0, 0, 10, 10]], float)
    m = iou_matrix(a, b)
    assert m[0, 1] == pytest.approx(1.0) and m[0, 0] == 0
    assert sorted(greedy_match(m, 0.3)) == [(0, 1), (1, 0)]


def test_track_keeps_id_while_moving():
    tr = tracker(min_hits=3)
    ids = set()
    for i in range(30):
        tracks = tr.update([box_at(100 + 4 * i, 200)], [0.9], i * DT)
        ids |= {t.id for t in tracks}
    assert ids == {1} and tr.confirmed()[0].confirmed


def test_weak_detection_extends_track_but_never_starts_one():
    tr = tracker(min_hits=1, high_thr=0.4, low_thr=0.1)
    tr.update([box_at(100, 100)], [0.9], 0.0)
    tracks = tr.update([box_at(102, 100)], [0.2], DT)  # blurred: weak but overlapping
    assert len(tracks) == 1 and tracks[0].updated_now
    tracks = tr.update([box_at(102, 100), box_at(400, 400)], [0.2, 0.2], 2 * DT)
    assert [t.id for t in tracks] == [1]  # the far weak box did not start a track


def test_track_coasts_through_dropout_then_is_dropped():
    tr = tracker(min_hits=1, max_coast_s=1.0)
    for i in range(10):
        tr.update([box_at(100 + 3 * i, 100)], [0.9], i * DT)
    t_last = 9 * DT
    tracks = tr.update([], [], t_last + 0.5)  # 0.5 s gap: coasting, still alive
    assert len(tracks) == 1 and not tracks[0].updated_now
    assert box_at(127, 100)[0] < tracks[0].box[0] < box_at(127 + 45, 100)[0]  # decayed drift
    # re-detected a little further on (the hand slowed while hidden): same ID
    tracks = tr.update([box_at(150, 100)], [0.9], t_last + 0.6)
    assert [t.id for t in tracks] == [1]
    assert tr.update([], [], t_last + 0.6 + 1.1) == []  # > max_coast_s: dropped


def test_tentative_track_needs_min_hits():
    tr = tracker(min_hits=3)
    tr.update([box_at(50, 50)], [0.9], 0)
    tr.update([box_at(50, 50)], [0.9], DT)
    assert tr.confirmed() == []
    tr.update([box_at(50, 50)], [0.9], 2 * DT)
    assert len(tr.confirmed()) == 1


# ---------------------------------------------------------------------------------- markers
def _scene_with_tag(tag_id=0, at=(300, 220), side=80):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(d, tag_id, side)
    img = np.full((480, 640, 3), 200, np.uint8)
    x, y = at[0] - side // 2, at[1] - side // 2
    img[y - 10 : y + side + 10, x - 10 : x + side + 10] = 255  # white quiet zone
    img[y : y + side, x : x + side] = tag[..., None]
    return img


def test_apriltag_detector_finds_configured_id_only():
    det = make_marker_detector(load_config("tracking")["marker"])
    assert isinstance(det, MarkerDetector) and isinstance(det, AprilTagMarkerDetector)
    found = det.detect(_scene_with_tag(0))
    assert len(found) == 1 and found[0].id == 0
    assert found[0].center == pytest.approx((300, 220), abs=2)
    assert found[0].size_px == pytest.approx(80, rel=0.1)
    assert det.detect(_scene_with_tag(7)) == []  # not in ids: ignored


def test_marker_registry_errors():
    with pytest.raises(NotImplementedError, match="planned"):
        make_marker_detector({"type": "color_patch"})
    with pytest.raises(ValueError, match="unknown"):
        make_marker_detector({"type": "laser"})


# -------------------------------------------------------------------------------- user lock
def marker_at(x, y):
    return Marker(0, (x, y), np.zeros((4, 2), np.float32), 20.0, 100.0)


def test_marker_hand_distance():
    assert marker_hand_distance((50, 50), (40, 40, 60, 60)) == 0
    assert marker_hand_distance((50, 80), (40, 40, 60, 60)) == pytest.approx(1.0)


def test_lock_needs_marker_then_holds_without_it():
    tr, lock = tracker(min_hits=1), UserLock(acquire_frames=3, max_marker_dist=1.0)
    user, other = (100, 200), (400, 200)
    t = 0.0
    # no marker: never locks, even with hands present
    for _ in range(5):
        tracks = tr.update([box_at(*user), box_at(*other)], [0.9, 0.9], t)
        st = lock.update(tracks, [], t)
        t += DT
    assert st.state is LockState.SEARCHING and lock.needs_marker
    # marker on the user's wrist (just below the hand) for 3 frames -> LOCKED on that track
    for _ in range(3):
        tracks = tr.update([box_at(*user), box_at(*other)], [0.9, 0.9], t)
        st = lock.update(tracks, [marker_at(user[0], user[1] + 35)], t)
        t += DT
    user_id = st.track_id
    assert st.state is LockState.LOCKED and not lock.needs_marker
    assert user_id == next(x.id for x in tracks if abs(x.box[0] - (user[0] - 20)) < 1)
    # marker gone, user moves, blurred (weak) detections, short dropout: lock holds
    for i in range(20):
        dets = [box_at(user[0] + 3 * i, user[1]), box_at(*other)]
        scores = [0.2 if i % 3 else 0.9, 0.9]
        if 8 <= i < 14:  # 6-frame dropout of the user's hand
            dets, scores = [box_at(*other)], [0.9]
        tracks = tr.update(dets, scores, t)
        st = lock.update(tracks, None, t)
        t += DT
        assert st.state is LockState.LOCKED and st.track_id == user_id
    # the other person's hand never becomes the user
    assert st.track_id != next(x.id for x in tracks if x.box[0] > 300)


def test_lock_released_when_track_dropped_and_not_stolen():
    tr, lock = tracker(min_hits=1, max_coast_s=0.5), UserLock(acquire_frames=1, max_marker_dist=1.0)
    tracks = tr.update([box_at(100, 100), box_at(400, 100)], [0.9, 0.9], 0.0)
    assert lock.update(tracks, [marker_at(100, 135)], 0.0).state is LockState.LOCKED
    t = 0.0
    for _ in range(25):  # user's hand gone for > max_coast_s; other hand stays
        t += DT
        st = lock.update(tr.update([box_at(400, 100)], [0.9], t), None, t)
    assert st.state is LockState.SEARCHING and st.track_id is None
    assert lock.needs_marker


def test_acquire_requires_consecutive_sightings_of_same_hand():
    tr, lock = tracker(min_hits=1), UserLock(acquire_frames=3, max_marker_dist=1.0)
    t = 0.0
    seq = [(100, 135), (400, 135), (100, 135), (100, 135)]  # marker jumps between hands once
    for mx, my in seq:
        tracks = tr.update([box_at(100, 100), box_at(400, 100)], [0.9, 0.9], t)
        st = lock.update(tracks, [marker_at(mx, my)], t)
        t += DT
    assert st.state is LockState.ACQUIRING and st.progress == pytest.approx(2 / 3)


def test_steady_tracking_has_no_decay_lag():
    tr = tracker(min_hits=1)
    for i in range(40):
        tr.update([box_at(100 + 5 * i, 100)], [0.9], i * DT)
    tracks = tr.update([box_at(100 + 5 * 40, 100)], [0.9], 40 * DT)
    t = tracks[0]
    tr._predict(t, 41 * DT)  # one frame ahead, still inside the grace period
    assert (t.box[0] + t.box[2]) / 2 == pytest.approx(100 + 5 * 41, abs=1.0)


def test_effective_dt_closed_form():
    tr = tracker(coast_grace_s=0.1, coast_vel_halflife_s=0.2)
    assert tr._effective_dt(0.0, 0.1) == pytest.approx(0.1)  # all inside grace
    tau = 0.2 / np.log(2)
    assert tr._effective_dt(0.0, 0.3) == pytest.approx(
        0.1 + tau * (1 - 0.5)
    )  # 0.2 s past grace = 1 half-life
    assert tr._effective_dt(0.0, 100.0) == pytest.approx(0.1 + tau, rel=1e-6)  # bounded drift
