import zipfile

import pytest

from dronevision.remote_zip import RemoteZip, file_range_reader


@pytest.fixture
def sample_zip(tmp_path):
    path = tmp_path / "a.zip"
    files = {
        "dir/stored.txt": b"hello stored",
        "dir/deflated.bin": bytes(range(256)) * 50,
        "other/x.json": b'{"a": 1}',
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("dir/"), b"")
        zf.writestr("dir/stored.txt", files["dir/stored.txt"], compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "dir/deflated.bin", files["dir/deflated.bin"], compress_type=zipfile.ZIP_DEFLATED
        )
        zf.writestr("other/x.json", files["other/x.json"], compress_type=zipfile.ZIP_DEFLATED)
    return path, files


def test_index_and_read(sample_zip):
    path, files = sample_zip
    rz = RemoteZip.from_reader(file_range_reader(path), path.stat().st_size)
    assert sorted(rz.names()) == sorted(files)  # directories are skipped
    for name, data in files.items():
        assert rz.read(name) == data


def test_read_many_keeps_order_and_reports_errors(sample_zip):
    path, files = sample_zip
    rz = RemoteZip.from_reader(file_range_reader(path), path.stat().st_size)
    names = ["other/x.json", "missing.txt", "dir/stored.txt"]
    out = list(rz.read_many(names, workers=3))
    assert [n for n, _ in out] == names
    assert out[0][1] == files["other/x.json"]
    assert isinstance(out[1][1], KeyError)
    assert out[2][1] == files["dir/stored.txt"]


def test_crc_mismatch_detected(sample_zip, tmp_path):
    path, _ = sample_zip
    rz = RemoteZip.from_reader(file_range_reader(path), path.stat().st_size)
    m = rz.members["dir/stored.txt"]
    rz.members["dir/stored.txt"] = type(m)(m.offset, m.compress_size, m.method, m.crc ^ 1)
    with pytest.raises(ValueError, match="CRC"):
        rz.read("dir/stored.txt")
