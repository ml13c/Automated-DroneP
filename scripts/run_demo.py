"""Live demo: open a camera or video file and show it with a HUD.

M0: raw frames + measured FPS. M2: `--detector` adds the ONNX hand/ball detector
(configs/detector.yaml) with its per-frame latency. M3: `--fingers` adds the finger count.
M4: `--lock` adds hand tracking + the user lock (configs/tracking.yaml): show the wrist
marker to acquire; the lock then follows your hand's track ID without the marker.

    python scripts/run_demo.py --source 0
    python scripts/run_demo.py --source 0 --detector
    python scripts/run_demo.py --source 0 --fingers          # count above each hand
    python scripts/run_demo.py --source 0 --lock --fingers   # count only the user's hand
    python scripts/run_demo.py --source clip.mp4 --ball      # lock onto the moving ball
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2

from dronevision.config import REPO_ROOT, load_config
from dronevision.hud import draw_ball, draw_detections, draw_markers, draw_stats, draw_tracks
from dronevision.profiler import FpsMeter
from dronevision.sources import WebcamSource, open_source

WINDOW = "dronevision"


def _summary(name: str, ms: list[float], extra: str) -> None:
    if len(ms) > 1:
        warm = sorted(ms[1:])  # first call includes warm-up
        p95 = warm[int(0.95 * (len(warm) - 1))]
        print(f"{name}: mean {statistics.mean(warm):.1f} ms, p95 {p95:.1f} ms per frame ({extra})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="camera", help="camera config name or path")
    p.add_argument("--source", help="webcam index or video path (overrides config)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run)")
    p.add_argument("--detector", action="store_true", help="run the ONNX hand/ball detector")
    p.add_argument("--detector-config", default="detector")
    p.add_argument("--fingers", action="store_true", help="finger count (implies --detector)")
    p.add_argument("--classifier-config", default="classifier")
    p.add_argument("--lock", action="store_true", help="tracking + user lock (implies --detector)")
    p.add_argument("--tracking-config", default="tracking")
    p.add_argument("--ball", action="store_true", help="ball tracker + ROI (implies --detector)")
    args = p.parse_args(argv)
    args.detector = args.detector or args.fingers or args.lock or args.ball

    cfg = load_config(args.config)
    detector = fingers = tracker = lock = marker_det = ball_tracker = None
    if args.detector:
        from dronevision.detector import OnnxDetector

        try:
            detector = OnnxDetector.from_config(load_config(args.detector_config), REPO_ROOT)
        except Exception as e:  # noqa: BLE001 - e.g. model not exported yet
            print(f"error: cannot load detector: {e}", file=sys.stderr)
            return 1
        print(f"Detector: {detector.names} imgsz {detector.imgsz} on {detector.provider}")
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
    if args.lock:
        from dronevision.markers import make_marker_detector
        from dronevision.tracker import HandTracker
        from dronevision.user_lock import UserLock

        tcfg = load_config(args.tracking_config)
        tracker = HandTracker.from_config(tcfg["hand_tracker"])
        lock = UserLock.from_config(tcfg["user_lock"])
        marker_det = make_marker_detector(tcfg["marker"])
        # weak (blurred) hand detections must reach the tracker's second stage
        detector.conf["hand"] = min(detector.conf["hand"], tcfg["hand_tracker"]["low_thr"])
        print(
            f"User lock: marker '{tcfg['marker']['type']}', tracker coasts "
            f"{tcfg['hand_tracker']['max_coast_s']} s"
        )

    if args.ball:
        from dronevision.ball_tracker import BallTracker, ball_roi

        bcfg = load_config(args.tracking_config)
        ball_tracker = BallTracker(bcfg["ball_tracker"])
        detector.conf["ball"] = min(detector.conf["ball"], bcfg["ball_tracker"]["low_conf"])
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
    mark_ms: list[float] = []
    track_ms: list[float] = []
    with src:
        for frame in src:
            fps = meter.tick(frame.timestamp)
            first_ts = frame.timestamp if first_ts is None else first_ts
            last_ts = t = frame.timestamp
            n += 1

            image = cv2.flip(frame.image, 1) if mirror else frame.image.copy()
            stats = {"FPS": f"{fps:.1f}" if fps else "--", "src": src.name, "res": f"{w}x{h}"}
            if detector is not None:
                dets = detector(image)
                det_ms.append(sum(detector.last_ms.values()))
                stats["detect"] = (
                    f"{det_ms[-1]:.1f} ms ({detector.provider.replace('ExecutionProvider', '')})"
                )
                hands = [d for d in dets if d.name == "hand"]
                balls = [d for d in dets if d.name == "ball"]
                labels: list[tuple[tuple[float, float, float, float], str]] = []

                if tracker is not None:
                    t0 = time.perf_counter()
                    tracks = tracker.update(
                        [(d.x1, d.y1, d.x2, d.y2) for d in hands], [d.score for d in hands], t
                    )
                    track_ms.append((time.perf_counter() - t0) * 1e3)
                    markers = None
                    if lock.needs_marker:
                        t0 = time.perf_counter()
                        # tags must be read un-mirrored (a mirrored tag doesn't decode)
                        markers = marker_det.detect(frame.image)
                        if mirror:
                            markers = [_mirror_marker(m, w) for m in markers]
                        mark_ms.append((time.perf_counter() - t0) * 1e3)
                    status = lock.update(tracks, markers, t)
                    targets = [
                        tr.box
                        for tr in tracks
                        if tr.id == status.track_id
                        and status.state.value == "locked"
                        and tr.updated_now
                    ]
                    draw_markers(image, markers or [])
                    stats["user"] = (
                        f"{status.state.value} #{status.track_id}"
                        if status.track_id
                        else status.state.value
                    )
                    if status.state.value == "acquiring":
                        stats["user"] += f" {status.progress:.0%}"
                    if status.state.value == "locked" and not status.visible:
                        stats["user"] += f" (coast {status.coast_s:.2f} s)"
                else:
                    tracks, status = None, None
                    targets = [(d.x1, d.y1, d.x2, d.y2) for d in hands]

                if fingers is not None:
                    t0 = time.perf_counter()
                    for box in targets:
                        # crop from the clean full-resolution frame (before anything is drawn)
                        pred = fingers(hand_crop(image, box, crop_pad))
                        text = pred.label if pred.prob >= min_prob else "?"
                        labels.append((box, f"fingers {text} ({pred.prob:.2f})"))
                    fin_ms.append((time.perf_counter() - t0) * 1e3)

                ball_state = roi = None
                if ball_tracker is not None:
                    ball_state = ball_tracker.update(
                        [
                            (
                                (d.x1 + d.x2) / 2,
                                (d.y1 + d.y2) / 2,
                                max(d.x2 - d.x1, d.y2 - d.y1),
                                d.score,
                            )
                            for d in balls
                        ],
                        t,
                    )
                    roi = ball_roi(ball_state, w, h, bcfg["ball_roi"])
                    stats["ball"] = ball_state.state + (
                        f" (coast {ball_state.coast_s:.1f} s)"
                        if ball_state.state == "tracking" and not ball_state.visible
                        else ""
                    )
                if tracks is not None:
                    draw_tracks(image, [tr for tr in tracks if tr.confirmed], status, t)
                else:
                    draw_detections(image, hands)
                if ball_tracker is not None:
                    draw_ball(image, ball_state, roi)
                else:
                    draw_detections(image, balls)
                for box, text in labels:
                    cv2.putText(
                        image,
                        text,
                        (int(box[0]), max(14, int(box[1]) - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                stats["hands/balls"] = f"{len(hands)}/{len(balls)}"
            draw_stats(image, stats)
            cv2.imshow(WINDOW, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if lock is not None and key == ord("r"):  # manual release
                lock.reset()
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
    if detector is not None:
        _summary("detector", det_ms, f"pre+infer+post, {detector.provider}")
    _summary("tracker", track_ms, "hand tracks + association")
    _summary("marker", mark_ms, "only while not locked")
    _summary("fingers", fin_ms, "crop + classify, user hand only" if lock else "all hands")
    return 0


def _mirror_marker(m, width: int):
    from dronevision.markers import Marker

    c = m.corners.copy()
    c[:, 0] = width - 1 - c[:, 0]
    return Marker(m.id, (width - 1 - m.center[0], m.center[1]), c, m.size_px, m.score)


if __name__ == "__main__":
    raise SystemExit(main())
