"""Build the detector and finger-classifier datasets.

    python scripts/build_dataset.py all              # hagrid + coco + own + summary
    python scripts/build_dataset.py hagrid --scale 0.05   # quick smoke run
    python scripts/build_dataset.py coco
    python scripts/build_dataset.py own              # your labelled clips -> test_own
    python scripts/build_dataset.py summary

Outputs (gitignored, under data/):
    detector/images/{train,val,test,test_own}/*.jpg  + labels/<split>/*.txt  (YOLO: 0 hand, 1 ball)
    detector/dataset.yaml                            (Ultralytics dataset file)
    fingers/{train,val,test,test_own}/{0..5,none}/*.jpg   (square hand crops)
    raw/                                             (download cache; safe to delete)
Also writes results/dataset_summary.md (committed): counts per split/class, sources, licenses.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from dronevision.config import REPO_ROOT, load_config
from dronevision.datasets import (
    IMAGE_EXTS,
    box_big_enough,
    count_images,
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
from dronevision.remote_zip import RemoteZip

SPLITS = ("train", "val", "test")
HAND, BALL = 0, 1


# ---------------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def clean_prefix(root: Path, prefix: str) -> None:
    """Delete previously generated files that start with ``prefix`` (keeps other sources)."""
    if not root.is_dir():
        return
    for p in root.rglob(f"{prefix}*"):
        if p.is_file():
            p.unlink()


def save_crop(image: np.ndarray, yolo, pad: float, size: int, dst: Path) -> None:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = square_crop_px(yolo, w, h, pad)
    crop = cv2.resize(image[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


class Paths:
    def __init__(self, cfg: dict) -> None:
        self.root = REPO_ROOT / cfg["root"]
        self.raw = self.root / "raw"
        self.det = self.root / "detector"
        self.fingers = self.root / "fingers"

    def det_img(self, split: str) -> Path:
        return self.det / "images" / split

    def det_lbl(self, split: str) -> Path:
        return self.det / "labels" / split


# ----------------------------------------------------------------------------------- HaGRID
def _hagrid_annotations(cfg: dict, paths: Paths) -> dict[str, dict[str, dict]]:
    """{split: {gesture: {image_id: {bboxes, labels}}}}, cached without landmarks/meta."""
    h = cfg["hagrid"]
    cache = paths.raw / "hagrid" / "annotations.json.gz"
    if cache.is_file():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return json.load(f)

    log("HaGRID: reading annotation zip index")
    rz = RemoteZip.open_url(h["annotations_zip"])
    pattern = re.compile(
        rf"^{re.escape(h['annotations_prefix'])}/(train|val|test)/([a-z0-9_]+)\.json$"
    )
    wanted = [(m.group(1), m.group(2), n) for n in rz.names() if (m := pattern.match(n))]
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for i, (split, gesture, name) in enumerate(sorted(wanted), 1):
        data = json.loads(rz.read(name))
        out[split][gesture] = {
            k: {"bboxes": v["bboxes"], "labels": v["labels"]} for k, v in data.items()
        }
        log(f"  annotations {i}/{len(wanted)} {split}/{gesture}: {len(data)} images")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump(out, f)
    return dict(out)


def build_hagrid(cfg: dict, paths: Paths, scale: float) -> None:
    h, seed = cfg["hagrid"], cfg["seed"]
    ann = _hagrid_annotations(cfg, paths)
    gestures = sorted(ann["train"])
    log(f"HaGRID: {len(gestures)} gesture classes in annotations: {', '.join(gestures)}")

    finger_map = {str(k): str(v) for k, v in h["finger_map"].items()}
    unknown = sorted(set(finger_map) - set(gestures))
    if unknown:
        sys.exit(f"finger_map names not in this HaGRID version: {unknown}")

    # ---- choose images
    det_sel: dict[str, list[tuple[str, str]]] = {s: [] for s in SPLITS}  # (gesture, id)
    fin_sel: dict[str, list[tuple[str, str, str]]] = {s: [] for s in SPLITS}  # + class
    by_class: dict[str, list[str]] = defaultdict(list)
    for g, c in finger_map.items():
        by_class[c].append(g)
    for s in SPLITS:
        n_det = scaled(h["detector_per_gesture"][s], scale)
        for g in gestures:
            for k in sample(list(ann[s].get(g, {})), n_det, f"{seed}-det-{s}-{g}"):
                det_sel[s].append((g, k))
        for c, srcs in by_class.items():
            for g, n in split_counts(scaled(h["fingers_per_class"][s], scale), srcs).items():
                for k in sample(list(ann[s].get(g, {})), n, f"{seed}-fin-{s}-{g}"):
                    fin_sel[s].append((g, k, c))

    needed = {(g, k) for s in SPLITS for g, k in det_sel[s]}
    needed |= {(g, k) for s in SPLITS for g, k, _ in fin_sel[s]}
    img_dir = paths.raw / "hagrid" / "images"
    todo = sorted(x for x in needed if not (img_dir / x[0] / f"{x[1]}.jpg").is_file())
    log(f"HaGRID: {len(needed)} images selected, {len(todo)} to download")

    # ---- download only the selected members of the 128 GB zip
    if todo:
        rz = RemoteZip.open_url(h["images_zip"], paths.raw / "hagrid" / "images_index.json.gz")
        names = {f"{h['images_prefix']}/{g}/{k}.jpg": (g, k) for g, k in todo}
        missing = [n for n in names if n not in rz.members]
        if missing:
            log(f"  {len(missing)} selected images are not in the zip; skipping them")
        fetch = [n for n in names if n in rz.members]
        t0, failed = time.time(), 0
        for i, (name, data) in enumerate(rz.read_many(fetch, cfg["download_workers"]), 1):
            g, k = names[name]
            if isinstance(data, Exception):
                failed += 1
                log(f"  failed {name}: {data}")
            else:
                dst = img_dir / g / f"{k}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
            if i % 500 == 0 or i == len(fetch):
                rate = i / max(time.time() - t0, 1e-6)
                log(f"  downloaded {i}/{len(fetch)} ({rate:.1f} img/s, {failed} failed)")

    # ---- write detector (every hand box -> class hand)
    min_px = cfg["detector"]["min_box_px"]
    for s in SPLITS:
        clean_prefix(paths.det_img(s), "hagrid_")
        clean_prefix(paths.det_lbl(s), "hagrid_")
        for g, k in det_sel[s]:
            src = img_dir / g / f"{k}.jpg"
            im = cv2.imread(str(src)) if src.is_file() else None
            if im is None:
                continue
            hh, ww = im.shape[:2]
            boxes = [xywh_norm_to_yolo(b) for b in ann[s][g][k]["bboxes"]]
            lines = [yolo_line(HAND, b) for b in boxes if box_big_enough(b, ww, hh, min_px)]
            if not lines:
                continue
            link_or_copy(src, paths.det_img(s) / f"hagrid_{k}.jpg")
            lbl = paths.det_lbl(s) / f"hagrid_{k}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("HaGRID: detector split written")

    # ---- write finger crops (only hands whose label is the folder's gesture)
    f = cfg["fingers"]
    for s in SPLITS:
        clean_prefix(paths.fingers / s, "hagrid_")
        for g, k, c in fin_sel[s]:
            src = img_dir / g / f"{k}.jpg"
            im = cv2.imread(str(src)) if src.is_file() else None
            if im is None:
                continue
            hh, ww = im.shape[:2]
            rec = ann[s][g][k]
            for i, (b, lab) in enumerate(zip(rec["bboxes"], rec["labels"], strict=True)):
                if lab != g or finger_class(lab, finger_map) != c:
                    continue
                yb = xywh_norm_to_yolo(b)
                if not box_big_enough(yb, ww, hh, f["min_hand_px"]):
                    continue
                save_crop(
                    im,
                    yb,
                    f["crop_pad"],
                    f["crop_size"],
                    paths.fingers / s / c / f"hagrid_{k}_{i}.jpg",
                )
    log("HaGRID: finger crops written")


# ------------------------------------------------------------------------------------- COCO
def _coco_ball_index(cfg: dict, paths: Paths) -> dict[str, list[dict]]:
    """{"train2017"|"val2017": [{file, url, w, h, balls: [[x,y,w,h]], person_frac}]}."""
    c = cfg["coco_ball"]
    cache = paths.raw / "coco" / f"ball_index_cat{c['category_id']}.json.gz"
    if cache.is_file():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return json.load(f)
    log("COCO: reading annotation zip (instances only)")
    rz = RemoteZip.open_url(c["annotations_zip"])
    out = {}
    for coco_split in ("train2017", "val2017"):
        data = json.loads(rz.read(f"annotations/instances_{coco_split}.json"))
        balls: dict[int, list] = defaultdict(list)
        person_area: dict[int, float] = defaultdict(float)
        for a in data["annotations"]:
            if a["category_id"] == c["category_id"] and not a["iscrowd"]:
                balls[a["image_id"]].append(a["bbox"])
            elif a["category_id"] == 1:
                bw, bh = a["bbox"][2], a["bbox"][3]
                person_area[a["image_id"]] = max(person_area[a["image_id"]], bw * bh)
        images = {im["id"]: im for im in data["images"]}
        out[coco_split] = [
            {
                "file": images[i]["file_name"],
                "url": images[i]["coco_url"],
                "w": images[i]["width"],
                "h": images[i]["height"],
                "balls": bs,
                "person_frac": person_area[i] / (images[i]["width"] * images[i]["height"]),
            }
            for i, bs in sorted(balls.items())
        ]
        log(f"  {coco_split}: {len(out[coco_split])} images with category {c['category_id']}")
        del data
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def _download(url: str, dst: Path) -> Exception | None:
    if dst.is_file():
        return None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            return None
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return e
            time.sleep(2**attempt)
    return None


def build_coco(cfg: dict, paths: Paths, scale: float) -> None:
    c, seed = cfg["coco_ball"], cfg["seed"]
    idx = _coco_ball_index(cfg, paths)
    keep = lambda recs: [r for r in recs if r["person_frac"] <= c["max_person_area_frac"]]  # noqa: E731
    train_pool = keep(idx["train2017"])
    test_pool = keep(idx["val2017"])
    log(
        f"COCO: after person filter (<= {c['max_person_area_frac']:.0%} of image): "
        f"train2017 {len(train_pool)}/{len(idx['train2017'])}, "
        f"val2017 {len(test_pool)}/{len(idx['val2017'])}"
    )

    by_file = {r["file"]: r for r in train_pool + test_pool}
    files = sorted(r["file"] for r in train_pool)
    n_val = round(len(files) * c["val_fraction"])
    val_files = set(sample(files, n_val, f"{seed}-coco-val"))
    pools = {
        "train": [f for f in files if f not in val_files],
        "val": sorted(val_files),
        "test": sorted(r["file"] for r in test_pool),
    }
    sel = {
        s: sample(pools[s], scaled(c["max_images"][s], scale), f"{seed}-coco-{s}") for s in SPLITS
    }

    img_dir = paths.raw / "coco" / "images"
    jobs = [(by_file[f]["url"], img_dir / f) for s in SPLITS for f in sel[s]]
    log(f"COCO: {len(jobs)} images selected, {sum(not d.is_file() for _, d in jobs)} to download")
    with ThreadPoolExecutor(cfg["download_workers"]) as ex:
        errors = [e for e in ex.map(lambda j: _download(*j), jobs) if e]
    if errors:
        log(f"  {len(errors)} downloads failed (first: {errors[0]})")

    min_px = cfg["detector"]["min_box_px"]
    for s in SPLITS:
        clean_prefix(paths.det_img(s), "coco_")
        clean_prefix(paths.det_lbl(s), "coco_")
        for f in sel[s]:
            r, src = by_file[f], img_dir / f
            if not src.is_file():
                continue
            boxes = [xywh_px_to_yolo(b, r["w"], r["h"]) for b in r["balls"]]
            lines = [yolo_line(BALL, b) for b in boxes if box_big_enough(b, r["w"], r["h"], min_px)]
            if not lines:
                continue
            stem = f"coco_{Path(f).stem}"
            link_or_copy(src, paths.det_img(s) / f"{stem}.jpg")
            (paths.det_lbl(s) / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("COCO: detector split written")


# -------------------------------------------------------------------------------------- own
_FRAME_RE = re.compile(r"^(?P<clip>.+?)__f(?P<frame>\d+)")


def _read_class_names(root: Path) -> list[str] | None:
    for name in ("data.yaml", "dataset.yaml"):
        for p in root.rglob(name):
            names = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if isinstance(names, list):
                return names
    for name in ("obj.names", "classes.txt"):
        for p in root.rglob(name):
            return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return None


def build_own(cfg: dict, paths: Paths) -> None:
    o = cfg["own"]
    root = REPO_ROOT / o["labeled_dir"]
    if not root.is_dir():
        log(
            f"own: {root} not found. Record clips (scripts/record_clips.py), label them in "
            "CVAT/Roboflow, export as YOLO into that folder, then re-run `build_dataset.py own`."
        )
        return
    names = _read_class_names(root)
    if not names:
        sys.exit(f"own: no data.yaml / obj.names / classes.txt with class names in {root}")
    ours = [n.lower() for n in cfg["detector"]["classes"]]
    remap = {i: ours.index(n.lower()) for i, n in enumerate(names) if n.lower() in ours}
    log(f"own: export classes {names} -> {remap}")

    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    clip_meta = {}
    rec_dir = REPO_ROOT / o["recordings_dir"]
    if rec_dir.is_dir():
        for js in rec_dir.glob("*.json"):
            clip_meta[js.stem] = json.loads(js.read_text(encoding="utf-8"))

    split, f = "test_own", cfg["fingers"]
    for d in (paths.det_img(split), paths.det_lbl(split), paths.fingers / split):
        shutil.rmtree(d, ignore_errors=True)
    n_img = n_crop = 0
    for txt in sorted(root.rglob("*.txt")):
        img = images.get(txt.stem)
        if img is None or txt.name in ("classes.txt", "obj.names"):
            continue
        boxes = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cls, b = parse_yolo_line(line)
                if cls in remap:
                    boxes.append((remap[cls], b))
        stem = f"own_{txt.stem}"
        link_or_copy(img, paths.det_img(split) / f"{stem}{img.suffix.lower()}")
        lbl = paths.det_lbl(split) / f"{stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("".join(yolo_line(c, b) + "\n" for c, b in boxes), encoding="utf-8")
        n_img += 1

        # finger label comes from the clip's metadata; crop the largest hand in the frame
        m = _FRAME_RE.match(txt.stem)
        count = clip_meta.get(m.group("clip"), {}).get("finger_count") if m else None
        hands = [b for c, b in boxes if c == HAND]
        if count is not None and hands:
            im = cv2.imread(str(img))
            if im is not None:
                big = max(hands, key=lambda b: b[2] * b[3])
                save_crop(
                    im,
                    big,
                    f["crop_pad"],
                    f["crop_size"],
                    paths.fingers / split / str(count) / f"{stem}.jpg",
                )
                n_crop += 1
    log(f"own: {n_img} labelled frames -> detector/test_own, {n_crop} finger crops")


# ---------------------------------------------------------------------------------- summary
def write_detector_yaml(cfg: dict, paths: Paths) -> None:
    classes = cfg["detector"]["classes"]
    doc = {
        "path": str(paths.det.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",  # public test; the honest one is images/test_own
        "names": dict(enumerate(classes)),
    }
    paths.det.mkdir(parents=True, exist_ok=True)
    (paths.det / "dataset.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _source_id(stem: str) -> str:
    """Original-image id behind a file: crops `hagrid_<id>_<i>` -> `hagrid_<id>`;
    own frames -> the whole clip (`own:<clip>`), so a clip can never straddle splits."""
    if stem.startswith("own_"):
        m = _FRAME_RE.match(stem[4:])
        return f"own:{m.group('clip') if m else stem[4:]}"
    if stem.startswith("hagrid_"):
        parts = stem.split("_")
        return "_".join(parts[:2])
    return stem


def leakage_check(paths: Paths, splits: list[str]) -> list[str]:
    """Overlap of source images/clips and identical file contents between splits."""
    import hashlib

    ids: dict[str, set[str]] = {s: set() for s in splits}
    hashes: dict[str, dict[str, str]] = {s: {} for s in splits}
    for s in splits:
        files = [p for p in paths.det_img(s).glob("*") if p.suffix.lower() in IMAGE_EXTS]
        files += [p for p in (paths.fingers / s).glob("*/*") if p.suffix.lower() in IMAGE_EXTS]
        for p in files:
            ids[s].add(_source_id(p.stem))
            hashes[s][hashlib.md5(p.read_bytes()).hexdigest()] = p.name
    lines = [
        "| splits | shared source images/clips | identical files |",
        "|---|---:|---:|",
    ]
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            shared = len(ids[a] & ids[b])
            same = len(hashes[a].keys() & hashes[b].keys())
            flag = " ⚠️" if shared or same else ""
            lines.append(f"| {a} ↔ {b} | {shared}{flag} | {same}{flag} |")
    own_in_training = sum(1 for s in ("train", "val") for i in ids[s] if i.startswith("own:"))
    lines += [
        "",
        f"Own-clip frames in train/val: **{own_in_training}** (must be 0; `own` only writes "
        "`test_own`). Detector and finger splits are pooled per split, so a source image used "
        "for training either model counts as training data.",
        "",
        "HaGRID images keep HaGRID's subject-disjoint split (by `user_id`, per the dataset "
        "authors), so no person appears in more than one split.",
    ]
    return lines


def low_count_flags(cfg: dict, stats: dict, splits: list[str]) -> list[str]:
    warn = cfg.get("summary_warn", {})
    out = []
    for s in splits:
        lim = warn.get("detector_boxes", {}).get(s)
        for c in cfg["detector"]["classes"]:
            if lim is not None and stats["detector"][s][c] < lim:
                out.append(f"- detector `{s}` / **{c}**: {stats['detector'][s][c]} boxes (< {lim})")
        lim = warn.get("finger_crops", {}).get(s)
        for c in cfg["fingers"]["classes"]:
            if lim is not None and stats["fingers"][s][c] < lim:
                out.append(f"- fingers `{s}` / **{c}**: {stats['fingers'][s][c]} crops (< {lim})")
    return out or ["- none"]


def summary(cfg: dict, paths: Paths) -> None:
    write_detector_yaml(cfg, paths)
    det_classes, fin_classes = cfg["detector"]["classes"], cfg["fingers"]["classes"]
    splits = [*SPLITS, "test_own"]
    stats: dict = {"detector": {}, "fingers": {}}
    for s in splits:
        boxes = count_yolo_labels(paths.det_lbl(s), len(det_classes))
        img = paths.det_img(s)
        srcs = defaultdict(int)
        if img.is_dir():
            for p in img.iterdir():
                srcs[p.name.split("_", 1)[0]] += 1
        stats["detector"][s] = {
            "images": count_images(img),
            "sources": dict(srcs),
            **dict(zip(det_classes, boxes, strict=True)),
        }
        stats["fingers"][s] = {c: count_images(paths.fingers / s / c) for c in fin_classes}

    lines = [
        "# Dataset summary",
        "",
        "Generated by `scripts/build_dataset.py summary`; counts are what is on disk in `data/`.",
        "",
        "## Detector (YOLO)",
        "",
        "| split | images | " + " | ".join(f"{c} boxes" for c in det_classes) + " | by source |",
        "|---|---:|" + "---:|" * len(det_classes) + "---|",
    ]
    for s in splits:
        d = stats["detector"][s]
        src = ", ".join(f"{k}: {v}" for k, v in sorted(d["sources"].items())) or "-"
        lines.append(
            f"| {s} | {d['images']} | " + " | ".join(str(d[c]) for c in det_classes) + f" | {src} |"
        )
    lines += [
        "",
        "## Finger classifier (ImageFolder crops)",
        "",
        "| split | " + " | ".join(fin_classes) + " | total |",
        "|---|" + "---:|" * (len(fin_classes) + 1),
    ]
    for s in splits:
        f = stats["fingers"][s]
        lines.append(
            f"| {s} | " + " | ".join(str(f[c]) for c in fin_classes) + f" | {sum(f.values())} |"
        )
    fm = cfg["hagrid"]["finger_map"]
    max_person = cfg["coco_ball"]["max_person_area_frac"]
    by_c = defaultdict(list)
    for g, c in fm.items():
        by_c[str(c)].append(g)
    lines += [
        "",
        "Finger class sources (HaGRID gesture → class): "
        + "; ".join(f"**{c}** ← {', '.join(sorted(by_c[c]))}" for c in fin_classes if c in by_c),
        "",
        "## Classes with too few examples",
        "",
        "Thresholds: `summary_warn` in `configs/dataset.yaml`.",
        "",
        *low_count_flags(cfg, stats, splits),
        "",
        "## Leakage check",
        "",
        *leakage_check(paths, splits),
        "",
        "## Sources and licenses",
        "",
        f"- **HaGRIDv2** (512 px version): hand boxes and gestures. {cfg['hagrid']['license']}. "
        "Kapitanov et al., *HaGRID: HAnd Gesture Recognition Image Dataset*, WACV 2024; "
        "Nuzhdin et al., *HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition*, "
        "arXiv:2412.01508. Splits follow HaGRID's subject-disjoint train/val/test.",
        "- **COCO 2017** `sports ball`: annotations CC BY 4.0, images under their Flickr "
        f"licenses. Images whose largest person covers > {max_person:.0%} "
        "of the frame are dropped because COCO does not label hands.",
        "- **Own clips** (`test_own`): recorded with `scripts/record_clips.py`; the only held-out "
        "test data that matches the drone use case.",
        "",
    ]
    out = REPO_ROOT / "results" / "dataset_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    (paths.root / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n".join(lines[4 : lines.index("## Sources and licenses")]))
    log(f"wrote {out.relative_to(REPO_ROOT)} and {paths.det / 'dataset.yaml'}")


# ------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("step", choices=["all", "hagrid", "coco", "own", "summary"])
    p.add_argument("--config", default="dataset")
    p.add_argument("--scale", type=float, default=1.0, help="multiply all subset sizes")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    paths = Paths(cfg)
    if args.step in ("all", "hagrid"):
        build_hagrid(cfg, paths, args.scale)
    if args.step in ("all", "coco"):
        build_coco(cfg, paths, args.scale)
    if args.step in ("all", "own"):
        build_own(cfg, paths)
    summary(cfg, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
