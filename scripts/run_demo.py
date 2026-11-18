"""Live demo: open a camera or video file and show it with a HUD.

M0: raw frames + measured FPS. M2: `--detector` adds the ONNX hand/ball detector
(configs/detector.yaml) with its per-frame latency. M3: `--fingers` adds the finger count.
M4: `--lock` adds hand tracking + the user lock (configs/tracking.yaml): show the wrist
marker to acquire; the lock then follows your hand's track ID without the marker.
`--ball` locks onto the moving ball. M5: `--control` is the full demo: user lock + finger
count -> flight mode (configs/modes.yaml), keyboard glove, failsafes, and the 20 Hz
controller (configs/controller.yaml) drawn as a velocity command.

    python scripts/run_demo.py --source 0
    python scripts/run_demo.py --source 0 --detector
    python scripts/run_demo.py --source 0 --fingers          # count above each hand
    python scripts/run_demo.py --source 0 --lock --fingers   # count only the user's hand
    python scripts/run_demo.py --source clip.mp4 --ball      # lock onto the moving ball
    python scripts/run_demo.py --source 0 --control          # full demo (M5)

Keys: q/Esc quit, r release the user lock. With --control: 1-5 glove command,
space glove emergency land, g glove link off/on, t reset after landing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

from dronevision.config import REPO_ROOT, load_config
from dronevision.hud import (
    draw_ball,
    draw_command,
    draw_detections,
    draw_markers,
    draw_mode_panel,
    draw_stats,
    draw_target_line,
    draw_tracks,
)
from dronevision.profiler import FpsMeter
from dronevision.sources import VideoFileSource, WebcamSource, open_source

WINDOW = "dronevision"
RESET_KEY = ord("t")


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
    p.add_argument(
        "--control",
        action="store_true",
        help="full M5 demo: modes + glove + controller (implies --lock --fingers)",
    )
    p.add_argument("--modes-config", default="modes")
    p.add_argument("--start-mode", help="override modes.yaml initial_mode (e.g. track_ball)")
    p.add_argument("--controller-config", default="controller")
    p.add_argument("--save", type=Path, help="also write the annotated video here (.mp4)")
    p.add_argument("--no-window", action="store_true", help="don't open a window (files)")
    args = p.parse_args(argv)
    args.lock = args.lock or args.control
    args.fingers = args.fingers or args.control
    args.detector = args.detector or args.fingers or args.lock or args.ball

    cfg = load_config(args.config)
    detector = fingers = tracker = lock = marker_det = ball_tracker = None
    machine = glove = loop = None
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
    tcfg = load_config(args.tracking_config)
    if args.lock:
        from dronevision.markers import make_marker_detector
        from dronevision.tracker import HandTracker
        from dronevision.user_lock import UserLock

        tracker = HandTracker.from_config(tcfg["hand_tracker"])
        lock = UserLock.from_config(tcfg["user_lock"])
        marker_det = make_marker_detector(tcfg["marker"])
        # weak (blurred) hand detections must reach the tracker's second stage
        detector.conf["hand"] = min(detector.conf["hand"], tcfg["hand_tracker"]["low_thr"])
        print(
            f"User lock: marker '{tcfg['marker']['type']}', tracker coasts "
            f"{tcfg['hand_tracker']['max_coast_s']} s"
        )
    if args.ball or args.control:
        from dronevision.ball_tracker import BallTracker, ball_roi

        ball_tracker = BallTracker(tcfg["ball_tracker"])
        detector.conf["ball"] = min(detector.conf["ball"], tcfg["ball_tracker"]["low_conf"])
    if args.control:
        from dronevision.controller import ControlLoop, Target
        from dronevision.glove_sim import KeyboardGlove
        from dronevision.modes import ModeMachine

        mcfg = load_config(args.modes_config)
        if args.start_mode:
            mcfg["initial_mode"] = args.start_mode
        ccfg = load_config(args.controller_config)
        machine = ModeMachine(mcfg)
        glove = KeyboardGlove.from_config(mcfg)
        loop = ControlLoop.from_config(ccfg)
        print(
            f"Control: start in {mcfg['initial_mode']}, controller {ccfg['rate_hz']} Hz. Keys: "
            "1-5 glove, space emergency land, g glove link, t reset"
        )

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
    # files: track on media time (frame / fps) so speeds are real, however fast we decode
    file_fps = src.reported_fps if isinstance(src, VideoFileSource) else 0.0
    meter = FpsMeter(window=int(cfg.get("hud", {}).get("fps_window_frames", 30)))
    w, h = src.resolution
    print(f"Opened {src.name} at {w}x{h}. Press q or Esc to quit.")
    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        out_fps = file_fps or float(cfg.get("fps") or 30)
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    first_ts = last_ts = None
    n = 0
    det_ms: list[float] = []
    fin_ms: list[float] = []
    mark_ms: list[float] = []
    track_ms: list[float] = []
    ball_frames = {"with_det": 0, "tracking": 0, "visible": 0, "ran": 0}
    mode_log: list[tuple[float, str, str]] = []
    key = -1
    last_user_t: float | None = None  # last time the locked user hand was seen
    ball_seen_t = 0.0
    prev_mode = None
    t_start = None
    if loop is not None:
        loop.start()
    try:
        with src:
            for frame in src:
                fps = meter.tick(frame.timestamp)
                first_ts = frame.timestamp if first_ts is None else first_ts
                last_ts = frame.timestamp
                t = frame.index / file_fps if file_fps else frame.timestamp
                t_start = t if t_start is None else t_start
                n += 1

                image = cv2.flip(frame.image, 1) if mirror else frame.image.copy()
                stats = {
                    "FPS": f"{fps:.1f}" if fps else "--",
                    "src": src.name,
                    "res": f"{w}x{h}",
                }
                glove_cmd = None
                if glove is not None:
                    glove_cmd = glove.poll(key, t)
                    if key == RESET_KEY:
                        machine.reset(t)
                        last_user_t = None
                mode_now = machine.mode if machine is not None else None
                if detector is not None:
                    dets = detector(image)
                    det_ms.append(sum(detector.last_ms.values()))
                    prov = detector.provider.replace("ExecutionProvider", "")
                    stats["detect"] = f"{det_ms[-1]:.1f} ms ({prov})"
                    hands = [d for d in dets if d.name == "hand"]
                    balls = [d for d in dets if d.name == "ball"]
                    labels: list[tuple[tuple[float, float, float, float], str]] = []

                    user_box = None
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
                        locked = status.state.value == "locked"
                        user = next((tr for tr in tracks if tr.id == status.track_id), None)
                        if locked and user is not None:
                            user_box = user.box  # also while coasting (predicted)
                        targets = [user.box] if locked and user and user.updated_now else []
                        if targets:
                            last_user_t = t
                        draw_markers(image, markers or [])
                        stats["user"] = (
                            f"{status.state.value} #{status.track_id}"
                            if status.track_id
                            else status.state.value
                        )
                        if status.state.value == "acquiring":
                            stats["user"] += f" {status.progress:.0%}"
                        if locked and not status.visible:
                            stats["user"] += f" (coast {status.coast_s:.2f} s)"
                    else:
                        tracks, status = None, None
                        targets = [(d.x1, d.y1, d.x2, d.y2) for d in hands]

                    user_finger = None
                    if fingers is not None:
                        t0 = time.perf_counter()
                        for box in targets:
                            # crop from the clean full-resolution frame (before anything is drawn)
                            pred = fingers(hand_crop(image, box, crop_pad))
                            text = pred.label if pred.prob >= min_prob else "?"
                            labels.append((box, f"fingers {text} ({pred.prob:.2f})"))
                            if lock is not None:
                                user_finger = (pred.label, pred.prob)
                        fin_ms.append((time.perf_counter() - t0) * 1e3)

                    # ball: always with --ball; with --control only in track_ball
                    ball_state = roi = None
                    run_ball = ball_tracker is not None and (
                        machine is None or mode_now == "track_ball"
                    )
                    if machine is not None and mode_now == "track_ball" and prev_mode != mode_now:
                        ball_tracker.reset()  # fresh search every time the mode is entered
                        ball_seen_t = t
                    if run_ball:
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
                        roi = ball_roi(ball_state, w, h, tcfg["ball_roi"])
                        stats["ball"] = ball_state.state + (
                            f" (coast {ball_state.coast_s:.1f} s)"
                            if ball_state.state == "tracking" and not ball_state.visible
                            else ""
                        )
                        ball_frames["ran"] += 1
                        ball_frames["with_det"] += bool(balls)
                        if ball_state.state == "tracking":
                            ball_seen_t = t
                            ball_frames["tracking"] += 1
                            ball_frames["visible"] += ball_state.visible

                    # ---- M5: modes + controller ------------------------------------------
                    if machine is not None:
                        prev_mode = mode_now
                        mstatus = machine.update(
                            t,
                            finger=user_finger,
                            glove=glove_cmd,
                            user_lost_s=0.0 if last_user_t is None else t - last_user_t,
                            glove_lost_s=glove.link_lost_s(t),
                            ball_lost_s=t - ball_seen_t,
                        )
                        if not mode_log or mode_log[-1][1] != mstatus.mode:
                            mode_log.append((t - t_start, mstatus.mode, mstatus.reason))
                        target, target_xy = None, None
                        if mstatus.mode == "follow_hand" and user_box is not None:
                            x1, y1, x2, y2 = user_box
                            target_xy = ((x1 + x2) / 2, (y1 + y2) / 2)
                            target = (target_xy[0], target_xy[1], y2 - y1)
                        elif (
                            mstatus.mode == "track_ball"
                            and ball_state is not None
                            and ball_state.state == "tracking"
                        ):
                            target_xy = ball_state.center
                            target = (*ball_state.center, ball_state.diam)
                        if target is not None:
                            cx = w - 1 - target[0] if mirror else target[0]  # camera's own view
                            target = Target(cx, target[1], target[2], w, h)
                        loop.post(mstatus.mode, target)

                    if tracks is not None:
                        draw_tracks(image, [tr for tr in tracks if tr.confirmed], status, t)
                    else:
                        draw_detections(image, hands)
                    if ball_state is not None:
                        draw_ball(image, ball_state, roi)
                    elif machine is None or mode_now == "track_ball":
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
                    if machine is not None:
                        draw_target_line(image, target_xy)
                        draw_mode_panel(
                            image,
                            mstatus,
                            machine.commands,
                            glove.connected,
                            glove.link_lost_s(t),
                        )
                        draw_command(image, loop.latest, loop.controller.limits)
                draw_stats(image, stats)
                if writer is not None:
                    writer.write(image)

                key = -1
                if not args.no_window:
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
    finally:
        if loop is not None:
            loop.stop()
        if writer is not None:
            writer.release()
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
    if ball_frames["ran"]:
        r = ball_frames["ran"]
        print(
            f"ball: {r} frames, {ball_frames['with_det']} with a ball detection "
            f"({ball_frames['with_det'] / r:.0%}), tracking {ball_frames['tracking'] / r:.0%} "
            f"(matched {ball_frames['visible']}, coasting "
            f"{ball_frames['tracking'] - ball_frames['visible']})"
        )
    if loop is not None:
        print(
            f"control loop: {loop.ticks} ticks, {loop.measured_hz:.1f} Hz "
            f"({loop.late_ticks} late), independent of the camera loop"
        )
    for tm, mode, reason in mode_log:
        print(f"  mode {tm:7.2f} s  {mode:<12} ({reason})")
    if args.save:
        print(f"saved {args.save}")
    return 0


def _mirror_marker(m, width: int):
    from dronevision.markers import Marker

    c = m.corners.copy()
    c[:, 0] = width - 1 - c[:, 0]
    return Marker(m.id, (width - 1 - m.center[0], m.center[1]), c, m.size_px, m.score)


if __name__ == "__main__":
    raise SystemExit(main())
