"""Record your own webcam clips for the held-out test set (the honest benchmark).

    python scripts/record_clips.py --distance 2 --place indoor --count 3
    python scripts/record_clips.py --distance 4 --place outdoor --ball --seconds 15
    python scripts/record_clips.py --extract recordings/<clip>.mp4 --every 10

Each clip is saved as recordings/<stamp>_d<dist>m_<place>_c<count>[_ball].mp4 with a JSON
sidecar (distance, place, finger count, ball, measured FPS). `--every N` also dumps every Nth
frame to data/own/frames/<clip>__f<idx>.jpg for labelling in CVAT/Roboflow. Keep the
`<clip>__f<idx>` prefix in exported file names: build_dataset.py uses it to find each frame's
finger count. Hold up ONE counting hand per clip; the largest labelled hand is used.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2

from dronevision.config import REPO_ROOT, load_config
from dronevision.hud import draw_stats
from dronevision.sources import WebcamSource, open_source

FRAMES_DIR = REPO_ROOT / "data" / "own" / "frames"
WINDOW = "record_clips"


def extract_frames(clip: Path, every: int, out_dir: Path = FRAMES_DIR) -> int:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    out_dir.mkdir(parents=True, exist_ok=True)
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            cv2.imwrite(str(out_dir / f"{clip.stem}__f{i:05d}.jpg"), frame)
            n += 1
        i += 1
    cap.release()
    return n


def record(args: argparse.Namespace, cam: dict) -> Path | None:
    out_dir = REPO_ROOT / cam.get("recordings_dir", "recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    dist = f"{args.distance:g}".replace(".", "p")
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_d{dist}m_{args.place}_c{args.count}"
    if args.ball:
        stem += "_ball"
    video = out_dir / f"{stem}.mp4"

    src = open_source(
        args.source or str(cam.get("source", "0")),
        width=cam.get("width"),
        height=cam.get("height"),
        fps=cam.get("fps"),
        backend=cam.get("backend", "auto"),
    )
    mirror = bool(cam.get("mirror", False)) and isinstance(src, WebcamSource)
    w, h = src.resolution
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(cam.get("fps", 30)), (w, h)
    )
    if not writer.isOpened():
        src.close()
        raise RuntimeError(f"cannot write {video}")

    n, t_start, aborted = 0, None, False
    countdown_end = time.perf_counter() + args.countdown
    with src:
        for frame in src:
            now = frame.timestamp
            recording = now >= countdown_end
            if recording:
                t_start = now if t_start is None else t_start
                writer.write(frame.image)  # raw frame, never mirrored or annotated
                n += 1
            shown = cv2.flip(frame.image, 1) if mirror else frame.image.copy()
            status = (
                f"REC {now - t_start:4.1f}/{args.seconds:g}s"
                if recording
                else f"starting in {countdown_end - now:3.1f}s"
            )
            draw_stats(
                shown,
                {
                    "status": status,
                    "dist": f"{args.distance:g} m",
                    "count": args.count,
                    "place": args.place,
                },
            )
            cv2.imshow(WINDOW, shown)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                aborted = True
                break
            if recording and now - t_start >= args.seconds:
                break
    writer.release()
    cv2.destroyAllWindows()

    if aborted or n == 0:
        video.unlink(missing_ok=True)
        print("aborted; nothing saved")
        return None
    duration = max(frame.timestamp - t_start, 1e-6)
    meta = {
        "clip": stem,
        "distance_m": args.distance,
        "place": args.place,
        "finger_count": None if args.count == "-" else args.count,
        "ball": args.ball,
        "frames": n,
        "duration_s": round(duration, 3),
        "fps_measured": round((n - 1) / duration, 2) if n > 1 else None,
        "resolution": [w, h],
        "camera": src.name,
        "host": platform.node(),
        "note": args.note,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {video.relative_to(REPO_ROOT)} ({n} frames, {meta['fps_measured']} fps)")
    return video


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--distance", type=float, help="metres from camera to hand (1, 2, 3, 4...)")
    p.add_argument("--place", choices=["indoor", "outdoor"])
    p.add_argument(
        "--count",
        default="-",
        choices=["0", "1", "2", "3", "4", "5", "none", "-"],
        help="finger count held for the whole clip ('-' = mixed/unknown)",
    )
    p.add_argument("--ball", action="store_true", help="a ball is in the clip")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--countdown", type=float, default=3)
    p.add_argument("--source", help="webcam index or video path (default: configs/camera.yaml)")
    p.add_argument("--note", default="")
    p.add_argument("--every", type=int, default=0, help="also dump every Nth frame for labelling")
    p.add_argument("--extract", type=Path, help="only extract frames from an existing clip")
    args = p.parse_args(argv)

    if args.extract:
        n = extract_frames(args.extract, args.every or 10)
        print(f"wrote {n} frames to {FRAMES_DIR.relative_to(REPO_ROOT)}")
        return 0
    if args.distance is None or args.place is None:
        p.error("--distance and --place are required when recording")

    video = record(args, load_config("camera"))
    if video and args.every:
        n = extract_frames(video, args.every)
        print(f"wrote {n} frames to {FRAMES_DIR.relative_to(REPO_ROOT)}")
    return 0 if video else 1


if __name__ == "__main__":
    sys.exit(main())
