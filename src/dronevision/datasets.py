"""Pure helpers for building the detector (YOLO) and finger (ImageFolder) datasets.

No I/O beyond what callers pass in, so everything here is unit-tested.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path

Box = tuple[float, float, float, float]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clip01(v: float) -> float:
    return min(1.0, max(0.0, v))


def xywh_norm_to_yolo(box: Sequence[float]) -> Box:
    """Normalized top-left [x, y, w, h] (HaGRID) -> clipped YOLO (cx, cy, w, h)."""
    x, y, w, h = box
    x0, y0, x1, y1 = clip01(x), clip01(y), clip01(x + w), clip01(y + h)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def xywh_px_to_yolo(box: Sequence[float], width: int, height: int) -> Box:
    """Pixel top-left [x, y, w, h] (COCO) -> clipped YOLO (cx, cy, w, h)."""
    x, y, w, h = box
    return xywh_norm_to_yolo((x / width, y / height, w / width, h / height))


def box_big_enough(yolo: Box, width: int, height: int, min_px: float) -> bool:
    return yolo[2] * width >= min_px and yolo[3] * height >= min_px


def yolo_line(cls: int, box: Box) -> str:
    return f"{cls} " + " ".join(f"{v:.6f}" for v in box)


def parse_yolo_line(line: str) -> tuple[int, Box]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
    c, *v = parts
    return int(c), (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def finger_class(gesture: str, finger_map: Mapping[str, str]) -> str | None:
    """Gesture label -> finger class ("0".."5" / "none"), or None if excluded."""
    value = finger_map.get(gesture)
    return None if value is None else str(value)


def square_crop_px(yolo: Box, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    """Square pixel crop (x0, y0, x1, y1) around a YOLO box, enlarged by ``pad`` per side.

    The square is shifted (not shrunk) to stay inside the image where possible.
    """
    cx, cy = yolo[0] * width, yolo[1] * height
    side = max(yolo[2] * width, yolo[3] * height) * (1 + 2 * pad)
    side = min(side, width, height)
    x0 = min(max(cx - side / 2, 0), width - side)
    y0 = min(max(cy - side / 2, 0), height - side)
    return int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side))


def split_counts(n_total: int, sources: Sequence[str]) -> dict[str, int]:
    """Spread ``n_total`` over ``sources`` as evenly as possible (deterministic)."""
    if not sources:
        return {}
    base, extra = divmod(n_total, len(sources))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(sorted(sources))}


def sample(items: Sequence, n: int, seed: int | str) -> list:
    """Deterministic sample of up to ``n`` items (sorted first, so input order doesn't matter)."""
    items = sorted(items)
    if n >= len(items):
        return items
    return random.Random(seed).sample(items, n)


def scaled(n: int, scale: float) -> int:
    return max(1, round(n * scale)) if n > 0 else 0


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def count_yolo_labels(label_dir: Path, n_classes: int) -> list[int]:
    """Number of boxes per class over all .txt files in ``label_dir``."""
    counts = [0] * n_classes
    if not label_dir.is_dir():
        return counts
    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls, _ = parse_yolo_line(line)
                if 0 <= cls < n_classes:
                    counts[cls] += 1
    return counts
