"""Export a trained detector.

    # IMX500 (Linux only -> run inside WSL2; Python 3.10-3.12, Java 17+):
    python scripts/export_models.py --weights runs/detector/<run>/weights/best.pt --format imx
    # ONNX FP32 (Windows or Linux):
    python scripts/export_models.py --weights runs/detector/<run>/weights/best.pt --format onnx

IMX export: INT8 post-training quantization calibrated on our train split, then the Sony
converter. The size of every output file and the converter's memory report are written to
results/detector/<run>/imx_export.json (+ a copy of the raw memory report).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import yaml

from dronevision.config import REPO_ROOT, load_config


def local_dataset_yaml(cfg: dict) -> Path:
    """dataset.yaml stores an absolute path from the OS that built it (e.g. C:\\...), which
    WSL can't read. Write a copy pointing at this checkout's data/detector for this OS."""
    src = REPO_ROOT / cfg["dataset"]
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    doc["path"] = str(src.parent.resolve())
    dst = src.with_name(f"dataset_{sys.platform}.yaml")
    dst.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dst


def export_imx(model, weights: Path, cfg: dict) -> dict:
    ex = cfg["export"]["imx"]
    data = local_dataset_yaml(cfg)
    t0 = time.time()
    out = Path(
        model.export(format="imx", data=str(data), imgsz=ex["imgsz"], fraction=ex["fraction"])
    )
    out_dir = out if out.is_dir() else out.parent
    files = {
        p.name: round(p.stat().st_size / 2**20, 3) for p in sorted(out_dir.iterdir()) if p.is_file()
    }
    reports = sorted(out_dir.glob("*MemoryReport*.json"))
    memory = json.loads(reports[0].read_text(encoding="utf-8")) if reports else None
    return {
        "output_dir": str(out_dir),
        "export_time_s": round(time.time() - t0, 1),
        "files_mib": files,
        "memory_report_file": reports[0].name if reports else None,
        "memory_report": memory,
        "_report_path": str(reports[0]) if reports else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--format", choices=["imx", "onnx"], required=True)
    p.add_argument("--config", default="detector")
    args = p.parse_args(argv)

    from ultralytics import YOLO
    from ultralytics import __version__ as ul_version

    cfg = load_config(args.config)
    weights = args.weights.resolve()
    run = weights.parent.parent.name  # runs/detector/<run>/weights/best.pt
    model = YOLO(str(weights))

    if args.format == "imx":
        if platform.system() != "Linux":
            raise SystemExit("IMX export is Linux-only: run this inside WSL2.")
        info = export_imx(model, weights, cfg)
    else:
        out = Path(model.export(format="onnx", imgsz=cfg["train"]["imgsz"], dynamic=False))
        info = {"output": str(out), "size_mib": round(out.stat().st_size / 2**20, 3)}

    info.update(
        {
            "weights": str(weights),
            "format": args.format,
            "ultralytics": ul_version,
            "platform": platform.platform(),
        }
    )
    res = REPO_ROOT / "results" / "detector" / run
    res.mkdir(parents=True, exist_ok=True)
    report_path = info.pop("_report_path", None)
    if report_path:
        shutil.copy2(report_path, res / Path(report_path).name)
    (res / f"{args.format}_export.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
