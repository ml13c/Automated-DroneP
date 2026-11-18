from pathlib import Path

import numpy as np
import pytest

from dronevision.config import REPO_ROOT, load_config
from dronevision.detector import Letterbox, decode, letterbox, nms, to_tensor

NAMES = ["hand", "ball"]
CONF = {"hand": 0.4, "ball": 0.25}


def _raw(rows):
    """rows: (cx, cy, w, h, s_hand, s_ball) in 320-px input space -> [1, 6, A]."""
    return np.array(rows, dtype=np.float32).T[None]


def test_letterbox_landscape_pads_vertically():
    img = np.zeros((480, 640, 3), np.uint8)
    out, lb = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert lb.scale == pytest.approx(0.5) and lb.pad_x == 0 and lb.pad_y == 40
    assert (out[:40] == 114).all() and (out[280:] == 114).all() and (out[40:280] == 0).all()


def test_to_tensor_is_rgb_nchw_unit_range():
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0] = 255  # blue in BGR
    t = to_tensor(img)
    assert t.shape == (1, 3, 2, 2) and t.dtype == np.float32
    assert t[0, 2].max() == 1.0 and t[0, 0].max() == 0.0  # blue ends up in channel 2


def test_nms_suppresses_overlap_keeps_disjoint():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_decode_thresholds_per_class_and_maps_back_to_image():
    lb = Letterbox(scale=0.5, pad_x=0, pad_y=40)
    raw = _raw(
        [
            (160, 160, 40, 40, 0.90, 0.01),  # hand, kept
            (162, 161, 40, 40, 0.85, 0.01),  # duplicate hand, suppressed by NMS
            (100, 100, 10, 10, 0.02, 0.30),  # ball 0.30 >= 0.25, kept
            (200, 200, 10, 10, 0.35, 0.02),  # hand 0.35 < 0.40, dropped
        ]
    )
    dets = decode(raw, NAMES, CONF, iou=0.5, max_det=10, lb=lb, image_size=(640, 480))
    assert [d.name for d in dets] == ["hand", "ball"]
    hand = dets[0]
    # input (140..180, 140..180) -> image x: /0.5 ; y: (v - 40) / 0.5
    assert (hand.x1, hand.y1, hand.x2, hand.y2) == pytest.approx((280, 200, 360, 280))
    assert dets[1].center == pytest.approx((200, 120))


def test_decode_class_aware_nms_keeps_overlapping_different_classes():
    lb = Letterbox(1.0, 0, 0)
    raw = _raw([(50, 50, 20, 20, 0.9, 0.0), (50, 50, 20, 20, 0.0, 0.8)])
    dets = decode(raw, NAMES, CONF, iou=0.5, max_det=10, lb=lb, image_size=(320, 320))
    assert sorted(d.name for d in dets) == ["ball", "hand"]


def test_decode_empty_and_max_det():
    lb = Letterbox(1.0, 0, 0)
    assert decode(_raw([(5, 5, 2, 2, 0.1, 0.1)]), NAMES, CONF, 0.5, 10, lb, (320, 320)) == []
    raw = _raw([(10 + 30 * i, 10, 5, 5, 0.9 - 0.01 * i, 0) for i in range(8)])
    assert len(decode(raw, NAMES, CONF, 0.5, 3, lb, (320, 320))) == 3


def test_config_has_threshold_for_every_class():
    cfg = load_config("detector")
    assert set(cfg["inference"]["conf"]) >= set(load_config("dataset")["detector"]["classes"])


ONNX = REPO_ROOT / load_config("detector")["inference"]["onnx"]


@pytest.mark.skipif(not Path(ONNX).is_file(), reason="trained ONNX model not present (CI)")
def test_onnx_detector_runs_end_to_end():
    from dronevision.detector import OnnxDetector

    det = OnnxDetector.from_config(load_config("detector"), REPO_ROOT)
    assert det.names == ["hand", "ball"] and det.imgsz == 320
    out = det(np.full((480, 640, 3), 127, np.uint8))
    assert isinstance(out, list)
    assert set(det.last_ms) == {"pre", "infer", "post"}
