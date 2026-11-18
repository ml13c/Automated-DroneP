"""Live demo: open a camera or video file and show it with a HUD.

M0: raw frames + measured FPS. M2: `--detector` adds the ONNX hand/ball detector
(configs/detector.yaml) with its per-frame latency. Later: tracking, modes, control.

    python scripts/run_demo.py --source 0
    python scripts/run_demo.py --source 0 --detector
    python scripts/run_demo.py --source path/to/clip.mp4 --detector
    python scripts/run_demo.py --source 0 --fingers     # M3: + finger count above each hand
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2

from dronevision.config import REPO_ROOT, load_config
from dronevision.hud import draw_detections, draw_stats
from dronevision.profiler import FpsMeter
from dronevision.sources import WebcamSource, open_source

WINDOW = "dronevision"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="camera", help="camera config name or path")
    p.add_argument("--source", help="webcam index or video path (overrides config)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run)")
    p.add_argument("--detector", action="store_true", help="run the ONNX hand/ball detector")
    p.add_argument("--detector-config", default="detector")
    p.add_argument(
        "--fingers", action="store_true", help="classify finger count (implies --detector)"
    )
    p.add_argument("--classifier-config", default="classifier")
    args = p.parse_args(argv)
    args.detector = args.detector or args.fingers

    cfg = load_config(args.config)
    detector = None
    if args.detector:
        from dronevision.detector import OnnxDetector

        try:
            detector = OnnxDetector.from_config(load_config(args.detector_config), REPO_ROOT)
        except Exception as e:  # noqa: BLE001 - e.g. model not exported yet
            print(f"error: cannot load detector: {e}", file=sys.stderr)
            return 1
        print(f"Detector: {detector.names} imgsz {detector.imgsz} on {detector.provider}")
    fingers = None
    if args.fingers:
        from dronevision.fingers import OnnxFingerClassifier, hand_crop

        clf_cfg = load_config(args.classifier_config)
        try:
            fingers = OnnxFingerClassifier.from_config(clf_cfg, REPO_ROOT)
        except Exception as e:  # noqa: BLE001 - e.g. classifier not trained yet
            print(f"error: cannot load finger classifier: {e}", file=sys.stderr)
            return 1
        crop_pad, min_prob = clf_cfg["inference"]["crop_pad"], clf_cfg["inference"]["min_prob"]
        print(f"Fingers: {fingers.classes} input {fingers.size} on {fingers.provider}")
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
    det_ms: list[float] = []
    fin_ms: list[float] = []
    with src:
        for frame in src:
            fps = meter.tick(frame.timestamp)
            first_ts = frame.timestamp if first_ts is None else first_ts
            last_ts = frame.timestamp
            n += 1

            image = cv2.flip(frame.image, 1) if mirror else frame.image
            stats = {"FPS": f"{fps:.1f}" if fps else "--", "src": src.name, "res": f"{w}x{h}"}
            if detector is not None:
                dets = detector(image)
                ms = sum(detector.last_ms.values())
                det_ms.append(ms)
                labels = []
                if fingers is not None:
                    t0 = time.perf_counter()
                    for d in (d for d in dets if d.name == "hand"):
                        # crop from the clean full-resolution frame (before anything is drawn)
                        pred = fingers(hand_crop(image, (d.x1, d.y1, d.x2, d.y2), crop_pad))
                        text = pred.label if pred.prob >= min_prob else "?"
                        labels.append((d, f"fingers {text} ({pred.prob:.2f})"))
                    fin_ms.append((time.perf_counter() - t0) * 1e3)
                draw_detections(image, dets)
                for d, text in labels:
                    cv2.putText(
                        image,
                        text,
                        (int(d.x1), max(14, int(d.y1) - 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                stats["detect"] = (
                    f"{ms:.1f} ms ({detector.provider.replace('ExecutionProvider', '')})"
                )
                stats["hands/balls"] = (
                    f"{sum(d.name == 'hand' for d in dets)}/{sum(d.name == 'ball' for d in dets)}"
                )
            draw_stats(image, stats)
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
    if len(det_ms) > 1:
        warm = det_ms[1:]  # first call includes ORT warm-up
        p95 = sorted(warm)[int(0.95 * (len(warm) - 1))]
        print(
            f"detector: mean {statistics.mean(warm):.1f} ms, p95 {p95:.1f} ms per frame "
            f"(pre+infer+post, {detector.provider}, {len(warm)} frames)"
        )
    if len(fin_ms) > 1:
        print(f"fingers: mean {statistics.mean(fin_ms[1:]):.1f} ms per frame (all hands, crop+clf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
