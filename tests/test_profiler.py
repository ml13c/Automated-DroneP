import pytest

from dronevision.profiler import FpsMeter


def test_fps_none_until_two_frames():
    m = FpsMeter(window=5)
    assert m.tick(0.0) is None


def test_fps_constant_rate():
    m = FpsMeter(window=10)
    for i in range(20):
        fps = m.tick(i / 25.0)
    assert fps == pytest.approx(25.0)


def test_fps_uses_only_recent_window():
    m = FpsMeter(window=3)
    for t in (0.0, 1.0, 2.0):  # 1 fps
        m.tick(t)
    for t in (2.1, 2.2):  # now 10 fps over last 3 frames
        fps = m.tick(t)
    assert fps == pytest.approx(10.0)


def test_window_must_be_at_least_two():
    with pytest.raises(ValueError):
        FpsMeter(window=1)
