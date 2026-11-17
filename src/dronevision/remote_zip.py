"""Read individual files out of a huge remote ZIP using HTTP range requests.

HaGRIDv2 ships as one 128 GB zip; we need a few GB of it. Reading the central directory
(~160 MB for 1M entries) once gives every member's offset, after which each image costs a
single range request. The index is cached as gzipped JSON so later runs skip that step.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

RangeReader = Callable[[int, int], bytes]  # (start, end_inclusive) -> bytes

_LOCAL_HEADER = 30
_HEADER_MARGIN = 1024  # bytes fetched for local name + extra fields (rarely > 100)


@dataclass(frozen=True)
class Member:
    offset: int
    compress_size: int
    method: int  # 0 = stored, 8 = deflate
    crc: int


def http_size(url: str, timeout: float = 60) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError(f"{url} does not advertise byte-range support")
        return int(r.headers["Content-Length"])


def http_range_reader(url: str, retries: int = 4, timeout: float = 120) -> RangeReader:
    def read(start: int, end: int) -> bytes:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    if r.status != 206:
                        raise RuntimeError(f"expected 206 Partial Content, got {r.status}")
                    return r.read()
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    return read


def file_range_reader(path: str | Path) -> RangeReader:
    """Local-file equivalent of http_range_reader (tests, or a fully downloaded zip)."""
    path = Path(path)

    def read(start: int, end: int) -> bytes:
        with path.open("rb") as f:
            f.seek(start)
            return f.read(end - start + 1)

    return read


class _RangeFile(io.RawIOBase):
    """Seekable read-only file over a RangeReader, so `zipfile` can parse the directory."""

    def __init__(self, read: RangeReader, size: int) -> None:
        self._read, self._size, self._pos = read, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = base + offset
        return self._pos

    def readinto(self, b) -> int:  # type: ignore[override]
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        data = self._read(self._pos, self._pos + n - 1)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


class RemoteZip:
    def __init__(self, read: RangeReader, members: dict[str, Member]) -> None:
        self._read = read
        self.members = members

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_reader(cls, read: RangeReader, size: int) -> RemoteZip:
        raw = io.BufferedReader(_RangeFile(read, size), buffer_size=1 << 20)
        with zipfile.ZipFile(raw) as zf:
            members = {
                i.filename: Member(i.header_offset, i.compress_size, i.compress_type, i.CRC)
                for i in zf.infolist()
                if not i.is_dir()
            }
        return cls(read, members)

    @classmethod
    def open_url(cls, url: str, index_cache: str | Path | None = None) -> RemoteZip:
        read = http_range_reader(url)
        cache = Path(index_cache) if index_cache else None
        if cache and cache.is_file():
            with gzip.open(cache, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("url") == url:
                return cls(read, {k: Member(*v) for k, v in payload["members"].items()})
        rz = cls.from_reader(read, http_size(url))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                members = {
                    k: [m.offset, m.compress_size, m.method, m.crc] for k, m in rz.members.items()
                }
                json.dump({"url": url, "members": members}, f)
            tmp.replace(cache)
        return rz

    # ---- reading ------------------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self.members)

    def read(self, name: str) -> bytes:
        m = self.members[name]
        blob = self._read(m.offset, m.offset + _LOCAL_HEADER + _HEADER_MARGIN + m.compress_size - 1)
        if blob[:4] != b"PK\x03\x04":
            raise ValueError(f"{name}: bad local header signature")
        n_name, n_extra = struct.unpack("<HH", blob[26:30])
        start = _LOCAL_HEADER + n_name + n_extra
        end = start + m.compress_size
        if end > len(blob):  # unusually long extra field: fetch the exact range
            blob = self._read(m.offset, m.offset + end - 1)
        data = blob[start:end]
        if m.method == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        elif m.method != zipfile.ZIP_STORED:
            raise ValueError(f"{name}: unsupported compression method {m.method}")
        if zlib.crc32(data) != m.crc:
            raise ValueError(f"{name}: CRC mismatch")
        return data

    def read_many(
        self, names: Iterable[str], workers: int = 16
    ) -> Iterator[tuple[str, bytes | Exception]]:
        """Yield (name, data) in input order, fetched in parallel; errors are yielded."""

        def safe(n: str) -> tuple[str, bytes | Exception]:
            try:
                return n, self.read(n)
            except Exception as e:  # noqa: BLE001 - surface per-file failures to caller
                return n, e

        with ThreadPoolExecutor(max_workers=workers) as ex:
            yield from ex.map(safe, names)
