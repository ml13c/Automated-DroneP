import numpy as np
import pytest

from dronevision.config import CONFIG_DIR, load_config
from dronevision.hud import draw_stats


def test_camera_config_loads_by_name():
    cfg = load_config("camera")
    for key in ("source", "width", "height", "backend", "mirror"):
        assert key in cfg


def test_load_config_by_path_matches_name():
    assert load_config(CONFIG_DIR / "camera.yaml") == load_config("camera")


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_draw_stats_draws_in_place():
    img = np.full((120, 200, 3), 128, dtype=np.uint8)
    out = draw_stats(img, {"FPS": "30.0", "src": "test"})
    assert out is img
    assert (img != 128).any()


def test_draw_stats_tolerates_tiny_image():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    draw_stats(img, {"a very long key": "a very long value"})


def test_platform_dataset_yaml_rewrites_path(tmp_path):
    import sys

    import yaml

    from dronevision.config import platform_dataset_yaml

    src = tmp_path / "dataset.yaml"
    src.write_text("path: /built/on/another/os\ntrain: images/train\nnames: {0: hand}\n")
    out = platform_dataset_yaml(src)
    assert out.name == f"dataset_{sys.platform}.yaml"
    doc = yaml.safe_load(out.read_text())
    assert doc["path"] == str(tmp_path.resolve())
    assert doc["train"] == "images/train" and doc["names"] == {0: "hand"}
