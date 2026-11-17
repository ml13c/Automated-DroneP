import pytest

from dronevision.config import load_config
from dronevision.datasets import (
    box_big_enough,
    count_yolo_labels,
    finger_class,
    parse_yolo_line,
    sample,
    scaled,
    split_counts,
    square_crop_px,
    xywh_norm_to_yolo,
    xywh_px_to_yolo,
    yolo_line,
)


def test_hagrid_box_to_yolo():
    assert xywh_norm_to_yolo([0.1, 0.2, 0.4, 0.2]) == pytest.approx((0.3, 0.3, 0.4, 0.2))


def test_box_clipped_to_image():
    cx, cy, w, h = xywh_norm_to_yolo([0.9, -0.1, 0.2, 0.3])
    assert (cx, cy, w, h) == pytest.approx((0.95, 0.1, 0.1, 0.2))


def test_coco_px_box_to_yolo():
    assert xywh_px_to_yolo([10, 20, 30, 40], 100, 200) == pytest.approx((0.25, 0.2, 0.3, 0.2))


def test_box_big_enough():
    assert box_big_enough((0.5, 0.5, 0.01, 0.1), 640, 480, 4)  # 6.4 px wide
    assert not box_big_enough((0.5, 0.5, 0.005, 0.1), 640, 480, 4)  # 3.2 px wide


def test_yolo_line_roundtrip():
    line = yolo_line(1, (0.5, 0.25, 0.1, 0.2))
    assert line == "1 0.500000 0.250000 0.100000 0.200000"
    cls, box = parse_yolo_line(line)
    assert cls == 1 and box == pytest.approx((0.5, 0.25, 0.1, 0.2))
    with pytest.raises(ValueError):
        parse_yolo_line("1 0.5 0.5")


def test_finger_class_mapping():
    fmap = {"fist": "0", "palm": "5", "call": "none"}
    assert finger_class("fist", fmap) == "0"
    assert finger_class("call", fmap) == "none"
    assert finger_class("point", fmap) is None


def test_dataset_config_finger_map_is_consistent():
    cfg = load_config("dataset")
    classes = set(cfg["fingers"]["classes"])
    assert set(map(str, cfg["hagrid"]["finger_map"].values())) <= classes
    assert {"0", "1", "2", "3", "4", "5", "none"} <= classes


def test_square_crop_inside_and_square():
    x0, y0, x1, y1 = square_crop_px((0.5, 0.5, 0.1, 0.2), 640, 480, 0.25)
    assert x1 - x0 == y1 - y0 == 144  # 0.2*480=96, *1.5
    assert (x0 + x1) / 2 == 320 and (y0 + y1) / 2 == 240


def test_square_crop_shifted_at_border():
    x0, y0, x1, y1 = square_crop_px((0.02, 0.98, 0.04, 0.04), 640, 480, 0.25)
    assert x0 == 0 and y1 == 480
    assert x1 - x0 == y1 - y0


def test_square_crop_never_exceeds_image():
    x0, y0, x1, y1 = square_crop_px((0.5, 0.5, 1.0, 1.0), 640, 480, 0.5)
    assert (x0, y0, x1, y1) == (80, 0, 560, 480)


def test_split_counts_even_and_total():
    out = split_counts(10, ["b", "a", "c"])
    assert out == {"a": 4, "b": 3, "c": 3}
    assert split_counts(5, []) == {}


def test_sample_deterministic_and_order_independent():
    items = [f"k{i}" for i in range(100)]
    a = sample(items, 10, "s")
    assert a == sample(list(reversed(items)), 10, "s")
    assert a != sample(items, 10, "other")
    assert sample(items[:3], 10, "s") == items[:3]


def test_scaled():
    assert scaled(300, 0.05) == 15
    assert scaled(10, 0.01) == 1
    assert scaled(0, 1.0) == 0


def test_count_yolo_labels(tmp_path):
    (tmp_path / "a.txt").write_text("0 .5 .5 .1 .1\n1 .5 .5 .1 .1\n0 .2 .2 .1 .1\n")
    (tmp_path / "b.txt").write_text("\n")
    assert count_yolo_labels(tmp_path, 2) == [2, 1]
    assert count_yolo_labels(tmp_path / "missing", 2) == [0, 0]
