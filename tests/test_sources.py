from pathlib import Path

import cv2
import numpy as np
import pytest

from dronevision.sources import FrameSource, VideoFileSource, open_source, parse_source_spec


def _write_video(path: Path, n: int, w: int = 64, h: int = 48) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
    assert writer.isOpened(), "OpenCV build cannot write MJPG/AVI"
    for i in range(n):
        writer.write(np.full((h, w, 3), i * 20 % 255, dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("0", 0), (" 2 ", 2), (1, 1), ("clip.mp4", Path("clip.mp4"))],
)
def test_parse_source_spec(spec, expected):
    assert parse_source_spec(spec) == expected


def test_video_file_source_reads_all_frames(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, n=7)

    with open_source(str(video)) as src:
        assert isinstance(src, VideoFileSource)
        assert isinstance(src, FrameSource)
        assert src.resolution == (64, 48)
        frames = list(src)

    assert [f.index for f in frames] == list(range(7))
    assert all(f.image.shape == (48, 64, 3) for f in frames)
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "nope.mp4"))
