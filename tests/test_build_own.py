"""End-to-end test of importing a labelled own-clip export (Roboflow/CVAT YOLO layout)."""

import importlib.util
import json

import cv2
import numpy as np
import pytest

from dronevision.config import REPO_ROOT, load_config


@pytest.fixture(scope="module")
def build_dataset():
    spec = importlib.util.spec_from_file_location(
        "build_dataset", REPO_ROOT / "scripts" / "build_dataset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("names_yaml", ["names: ['ball', 'hand']", "names: {0: ball, 1: hand}"])
def test_build_own_remaps_classes_and_crops(build_dataset, tmp_path, names_yaml):
    labeled, rec = tmp_path / "labeled", tmp_path / "recordings"
    (labeled / "test" / "images").mkdir(parents=True)
    (labeled / "test" / "labels").mkdir(parents=True)
    rec.mkdir()
    (labeled / "data.yaml").write_text(names_yaml + "\nnc: 2\n")

    clip = "20260101-120000_d2m_indoor_c4"
    (rec / f"{clip}.json").write_text(json.dumps({"finger_count": "4"}))
    for i, stem in enumerate([f"{clip}__f00000_jpg.rf.abc", "unrelated_frame"]):
        cv2.imwrite(
            str(labeled / "test" / "images" / f"{stem}.jpg"),
            np.full((240, 320, 3), 100 + i, np.uint8),
        )
        # class 1 = hand (big), class 1 = hand (small), class 0 = ball (in export's order)
        (labeled / "test" / "labels" / f"{stem}.txt").write_text(
            "1 0.5 0.5 0.4 0.5\n1 0.1 0.1 0.05 0.05\n0 0.8 0.8 0.1 0.1\n"
        )

    cfg = load_config("dataset")
    cfg["root"] = str(tmp_path / "data")
    cfg["own"] = {"labeled_dir": str(labeled), "recordings_dir": str(rec)}
    paths = build_dataset.Paths(cfg)
    build_dataset.build_own(cfg, paths)

    lbl = paths.det_lbl("test_own") / f"own_{clip}__f00000_jpg.rf.abc.txt"
    classes = [int(line.split()[0]) for line in lbl.read_text().splitlines()]
    assert classes == [0, 0, 1]  # hand -> 0, ball -> 1 in our order
    assert len(list(paths.det_img("test_own").iterdir())) == 2

    crops = list((paths.fingers / "test_own" / "4").glob("*.jpg"))
    assert len(crops) == 1  # only the frame whose clip has a finger_count
    assert cv2.imread(str(crops[0])).shape[:2] == (cfg["fingers"]["crop_size"],) * 2


def test_build_own_without_export_is_a_noop(build_dataset, tmp_path):
    cfg = load_config("dataset")
    cfg["root"] = str(tmp_path / "data")
    cfg["own"] = {"labeled_dir": str(tmp_path / "nope"), "recordings_dir": str(tmp_path)}
    build_dataset.build_own(cfg, build_dataset.Paths(cfg))
    assert not (tmp_path / "data").exists()
