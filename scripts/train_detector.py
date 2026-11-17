"""Train the YOLO11n hand/ball detector on the local GPU.

    python scripts/train_detector.py --name y11n_320_baseline
    python scripts/train_detector.py --name quick --epochs 1     # smoke test

Writes runs/detector/<name>/ (gitignored: weights, full logs) and copies the small, useful
outputs to results/detector/<name>/: results.csv, PR / confusion plots and train_summary.json
(wall time, peak VRAM, device, versions, final val/test metrics). All numbers are measured.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time

from dronevision.config import REPO_ROOT, load_config

KEEP = [
    "results.csv",
    "results.png",
    "args.yaml",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "labels.jpg",
    "val_batch0_pred.jpg",
]


def _metrics(m, names: dict) -> dict:
    out = {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "per_class_mAP50": {},
    }
    for i, c in enumerate(m.box.ap_class_index):
        out["per_class_mAP50"][names[int(c)]] = round(float(m.box.ap50[i]), 4)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", required=True, help="run name (folder under runs/detector)")
    p.add_argument("--config", default="detector")
    p.add_argument("--epochs", type=int, help="override configs/detector.yaml")
    p.add_argument("--imgsz", type=int, help="override configs/detector.yaml")
    args = p.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ul_version

    cfg = load_config(args.config)
    data = REPO_ROOT / cfg["dataset"]
    if not data.is_file():
        raise SystemExit(f"{data} missing: run scripts/build_dataset.py first")
    train_args = dict(cfg["train"])
    if args.epochs:
        train_args["epochs"] = args.epochs
    if args.imgsz:
        train_args["imgsz"] = args.imgsz

    model = YOLO(cfg["model"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train(
        data=str(data),
        project=str(REPO_ROOT / cfg["project"]),
        name=args.name,
        exist_ok=False,
        **train_args,
    )
    wall = time.time() - t0
    peak_vram = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None

    run_dir = model.trainer.save_dir
    best = run_dir / "weights" / "best.pt"
    best_model = YOLO(str(best))
    names = best_model.names
    common = dict(
        data=str(data),
        imgsz=train_args["imgsz"],
        device=train_args.get("device", 0),
        plots=False,
        verbose=False,
    )
    val = best_model.val(split="val", **common)
    test = best_model.val(split="test", **common)

    summary = {
        "run": args.name,
        "model": cfg["model"],
        "train_args": train_args,
        "epochs_completed": int(model.trainer.epoch) + 1,
        "wall_time_min": round(wall / 60, 1),
        "peak_vram_reserved_gib": round(peak_vram, 2) if peak_vram else None,
        "best_pt_mb": round(best.stat().st_size / 2**20, 2),
        "metrics": {"val": _metrics(val, names), "test_public": _metrics(test, names)},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {
            "torch": torch.__version__,
            "ultralytics": ul_version,
            "python": platform.python_version(),
        },
        "note": "test_public = HaGRID test split + COCO val2017 balls. Not own clips.",
    }
    out = REPO_ROOT / "results" / "detector" / args.name
    out.mkdir(parents=True, exist_ok=True)
    for f in KEEP:
        if (run_dir / f).is_file():
            shutil.copy2(run_dir / f, out / f)
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nweights: {best}\nsummary: {out / 'train_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
