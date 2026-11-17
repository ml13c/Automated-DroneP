"""Live demo: open a camera or video file and show it with a HUD.

M0: raw frames + measured FPS. Later milestones add detection, tracking, modes and control.

    python scripts/run_demo.py --source 0
    python scripts/run_demo.py --source path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import sys

import cv2

from dronevision.config import load_config
from dronevision.hud import draw_stats
from dronevision.profiler import FpsMeter
from dronevision.sources import WebcamSource, open_source

WINDOW = "dronevision"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="camera", help="camera config name or path")
    p.add_argument("--source", help="webcam index or video path (overrides config)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    spec = args.source if args.source is not None else str(cfg.get("source", "0"))

    try:
        src = open_source(
            spec,
            width=cfg.get("width"),
            height=cfg.get("height"),
            fps=cfg.get("fps"),
            backend=cfg.get("backend", "auto"),
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    mirror = bool(cfg.get("mirror", False)) and isinstance(src, WebcamSource)
    meter = FpsMeter(window=int(cfg.get("hud", {}).get("fps_window_frames", 30)))
    w, h = src.resolution
    print(f"Opened {src.name} at {w}x{h}. Press q or Esc to quit.")

    first_ts = last_ts = None
    n = 0
    with src:
        for frame in src:
            fps = meter.tick(frame.timestamp)
            first_ts = frame.timestamp if first_ts is None else first_ts
            last_ts = frame.timestamp
            n += 1

            image = cv2.flip(frame.image, 1) if mirror else frame.image
            draw_stats(
                image,
                {
                    "FPS": f"{fps:.1f}" if fps else "--",
                    "src": src.name,
                    "res": f"{w}x{h}",
                },
            )
            cv2.imshow(WINDOW, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.max_frames and n >= args.max_frames:
                break
    cv2.destroyAllWindows()

    if n >= 2 and last_ts > first_ts:
        print(
            f"{n} frames, mean {(n - 1) / (last_ts - first_ts):.1f} FPS (display loop, "
            f"{w}x{h}, {src.name})"
        )
    else:
        print(f"{n} frames read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
