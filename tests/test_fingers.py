import numpy as np
import pytest

from dronevision.config import REPO_ROOT, load_config
from dronevision.fingers import MEAN, STD, hand_crop, preprocess, softmax


def test_preprocess_shape_rgb_and_normalisation():
    crop = np.zeros((50, 70, 3), np.uint8)
    crop[..., 2] = 255  # red in BGR
    x = preprocess(crop, 128)
    assert x.shape == (1, 3, 128, 128) and x.dtype == np.float32
    assert x[0, 0].mean() == pytest.approx((1 - MEAN[0]) / STD[0], abs=1e-5)  # R channel first
    assert x[0, 2].mean() == pytest.approx((0 - MEAN[2]) / STD[2], abs=1e-5)


def test_hand_crop_is_square_padded_and_inside_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    crop = hand_crop(frame, (300, 200, 340, 260), pad=0.25)  # 40x60 box -> side 90
    assert crop.shape[:2] == (90, 90)
    edge = hand_crop(frame, (0, 0, 30, 30), pad=0.25)
    assert edge.shape[0] == edge.shape[1] > 0


def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 2


def test_classifier_and_dataset_crop_settings_agree():
    assert (
        load_config("classifier")["inference"]["crop_pad"]
        == (load_config("dataset")["fingers"]["crop_pad"])
    )
    assert [str(c) for c in load_config("classifier")["classes"]] == [
        str(c) for c in load_config("dataset")["fingers"]["classes"]
    ]


@pytest.mark.weights(load_config("classifier")["inference"]["onnx"])
def test_onnx_classifier_end_to_end():
    from dronevision.fingers import OnnxFingerClassifier

    clf = OnnxFingerClassifier.from_config(load_config("classifier"), REPO_ROOT)
    pred = clf(np.full((100, 100, 3), 127, np.uint8))
    assert pred.label in clf.classes and 0 <= pred.prob <= 1
    assert sum(pred.probs.values()) == pytest.approx(1.0, abs=1e-4)
