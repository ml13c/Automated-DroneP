"""Detector benchmarks: CPU vs CUDA, FP32 vs INT8 -> results/benchmarks.{md,json,png}.

    python scripts/benchmark.py                       # latency (needs data/detector/images/val)
    python scripts/benchmark.py --accuracy            # + test-split mAP50 per model (slow, CPU)

FP32 = the exported ONNX model (configs/detector.yaml). INT8 = the same model statically
quantized with ONNX Runtime (QDQ, convolutions only, calibrated on val images); it is built
once next to the FP32 file. This is a laptop INT8, not the IMX500's INT8: the IMX500 accuracy
from scripts/eval_quantized.py is shown alongside for reference. Latency is measured on real
val images through the same OnnxDetector the demo uses (letterbox -> inference -> decode/NMS).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from dronevision.config import REPO_ROOT, load_config
from dronevision.detector import OnnxDetector, letterbox, to_tensor

PROVIDERS = {"CPU": ["CPUExecutionProvider"], "CUDA": ["CUDAExecutionProvider"]}


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def gpu_name() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return name.decode() if isinstance(name, bytes) else name
    except Exception:  # noqa: BLE001 - no NVML / no NVIDIA GPU
        return None


def load_images(folder: Path, n: int) -> list[np.ndarray]:
    files = sorted(folder.glob("*.jpg"))[:n]
    if not files:
        raise SystemExit(f"no images in {folder}: build the dataset first (build_dataset.py)")
    return [cv2.imread(str(f)) for f in files]


def build_int8(fp32: Path, out: Path, calib: list[np.ndarray], imgsz: int) -> None:
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(calib)

        def get_next(self):
            img = next(self.it, None)
            return None if img is None else {"images": to_tensor(letterbox(img, imgsz)[0])}

    pre = out.with_name(out.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    quantize_static(
        str(pre),
        str(out),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv"],  # keep the box/class decode head in float
    )
    pre.unlink(missing_ok=True)
    # keep Ultralytics metadata (class names, stride, imgsz) so every loader still works
    src, dst = onnx.load(str(fp32)), onnx.load(str(out))
    del dst.metadata_props[:]
    dst.metadata_props.extend(src.metadata_props)
    onnx.save(dst, str(out))


def _stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "mean": round(statistics.mean(xs), 2),
        "p50": round(xs[len(xs) // 2], 2),
        "p95": round(xs[int(0.95 * (len(xs) - 1))], 2),
    }


def time_all(models: dict, images: list[np.ndarray], args, cfg) -> list[dict]:
    """Interleaved timing: every (model, provider) runs a slice in each of `rounds` rounds, so
    laptop boost/thermal drift hits all configurations alike instead of whichever ran last."""
    inf = cfg["inference"]
    dets = {}
    for name, path in models.items():
        for prov in PROVIDERS:
            d = OnnxDetector(
                path, inf["conf"], inf["iou"], inf["max_det"], providers=PROVIDERS[prov]
            )
            if prov == "CUDA" and d.provider != "CUDAExecutionProvider":
                print(f"{name} {prov}: provider not available, skipped")
                continue
            for i in range(args.warmup):
                d(images[i % len(images)])
            dets[(name, prov)] = (d, path, {"infer": [], "total": []})
    per_round = max(args.runs // args.rounds, 1)
    for _ in range(args.rounds):
        for d, _, acc in dets.values():
            for i in range(per_round):
                d(images[i % len(images)])
                acc["infer"].append(d.last_ms["infer"])
                acc["total"].append(sum(d.last_ms.values()))
    rows = []
    for (name, prov), (_, path, acc) in dets.items():
        r = {"model": name, "provider": prov, "size_mb": round(path.stat().st_size / 2**20, 2)}
        r.update(infer_ms=_stats(acc["infer"]), total_ms=_stats(acc["total"]))
        rows.append(r)
        print(f"{name:11s} {prov:4s} infer {r['infer_ms']['mean']:6.2f} ms  "
              f"total {r['total_ms']['mean']:6.2f} ms")  # fmt: skip
    return rows


def accuracy(path: Path, data_yaml: Path, imgsz: int) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(path), task="detect").val(
        data=str(data_yaml), split="test", imgsz=imgsz, batch=1, device="cpu",
        plots=False, verbose=False,
    )  # fmt: skip
    names = m.names
    per = {
        names[int(c)]: round(float(m.box.ap50[i]), 4) for i, c in enumerate(m.box.ap_class_index)
    }
    return {"mAP50": round(float(m.box.map50), 4), "per_class_mAP50": per}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--detector-config", default="detector")
    p.add_argument("--images", type=Path, default=REPO_ROOT / "data/detector/images/val")
    p.add_argument("--n-images", type=int, default=200, help="distinct val images to time on")
    p.add_argument("--n-calib", type=int, default=100, help="val images for INT8 calibration")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--runs", type=int, default=300, help="timed frames per configuration")
    p.add_argument("--rounds", type=int, default=5, help="interleaved rounds (drift control)")
    p.add_argument("--accuracy", action="store_true", help="also test-split mAP50 (slow)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    args = p.parse_args(argv)

    cfg = load_config(args.detector_config)
    fp32 = REPO_ROOT / cfg["inference"]["onnx"]
    int8 = fp32.with_name(fp32.stem + "_int8_ort.onnx")
    imgsz = cfg["export"]["imx"]["imgsz"]
    images = load_images(args.images, args.n_images)
    if not int8.is_file():
        print(f"building INT8 model from {args.n_calib} val images ...")
        build_int8(fp32, int8, images[-args.n_calib :], imgsz)
    models = {"FP32": fp32, "INT8 (ORT)": int8}

    rows = time_all(models, images, args, cfg)

    acc = {}
    if args.accuracy:
        from dronevision.config import platform_dataset_yaml

        data_yaml = platform_dataset_yaml(REPO_ROOT / cfg["dataset"])
        for name, path in models.items():
            print(f"accuracy {name} (test split, CPU) ...")
            acc[name] = accuracy(path, data_yaml, imgsz)
    imx_q = REPO_ROOT / "results/detector" / fp32.parent.parent.name / "quantization.json"

    result = {
        "date": date.today().isoformat(),
        "cpu": cpu_name(),
        "gpu": gpu_name(),
        "onnxruntime": __import__("onnxruntime").__version__,
        "imgsz": imgsz,
        "images": (
            f"{len(images)} val images, {args.warmup} warm-up + {args.runs} timed frames per "
            f"configuration, interleaved over {args.rounds} rounds"
        ),
        "latency": rows,
        "accuracy_test_mAP50": acc,
        "imx500_reference": imx_q.relative_to(REPO_ROOT).as_posix() if imx_q.is_file() else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmarks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    chart(rows, result, args.out / "benchmarks.png")
    (args.out / "benchmarks.md").write_text(report(result, imx_q), encoding="utf-8")
    print(f"wrote {args.out / 'benchmarks.md'}")
    return 0


def chart(rows: list[dict], result: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"CPU": "#2a78d6", "CUDA": "#eb6834"}  # categorical slots 1, 2
    models = list(dict.fromkeys(r["model"] for r in rows))
    provs = [p for p in PROVIDERS if any(r["provider"] == p for r in rows)]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    width = 0.36
    for j, prov in enumerate(provs):
        for i, model in enumerate(models):
            r = next((r for r in rows if r["model"] == model and r["provider"] == prov), None)
            if r is None:
                continue
            x = i + (j - (len(provs) - 1) / 2) * (width + 0.03)
            v = r["total_ms"]["mean"]
            ax.bar(x, v, width, color=colors[prov], label=prov if i == 0 else None)
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=ink)
    ax.set_xticks(range(len(models)), models, color=ink)
    ax.set_ylabel("ms per frame (mean)", color=muted)
    ax.set_title(
        f"Detector latency, {result['imgsz']} px, pre + inference + post",
        color=ink, loc="left", fontsize=11,
    )  # fmt: skip
    ax.tick_params(colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.yaxis.grid(True, color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=ink)
    dev = f"CPU: {result['cpu']}" + (f"  ·  GPU: {result['gpu']}" if result["gpu"] else "")
    fig.text(0.01, 0.01, f"{dev}  ·  ONNX Runtime {result['onnxruntime']}", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor=surface)
    plt.close(fig)


def report(result: dict, imx_q: Path) -> str:
    lines = [
        "# Detector benchmarks",
        "",
        f"Generated by `scripts/benchmark.py` on {result['date']}. CPU: **{result['cpu']}**"
        + (f"; GPU: **{result['gpu']}**" if result["gpu"] else "")
        + f"; ONNX Runtime {result['onnxruntime']}; YOLO11n at {result['imgsz']} px; "
        f"{result['images']}.",
        "",
        "INT8 (ORT) is the same model statically quantized by ONNX Runtime (QDQ, convolutions "
        "only, calibrated on val images). It is a laptop INT8, not the IMX500's.",
        "",
        "![latency](benchmarks.png)",
        "",
        "| model | provider | size (MB) | inference mean / p95 (ms) | total mean / p95 (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in result["latency"]:
        lines.append(
            f"| {r['model']} | {r['provider']} | {r['size_mb']} | "
            f"{r['infer_ms']['mean']} / {r['infer_ms']['p95']} | "
            f"{r['total_ms']['mean']} / {r['total_ms']['p95']} |"
        )
    lines += ["", "Total = letterbox + inference + decode/NMS, as in the live demo.", ""]
    tot = {(r["model"], r["provider"]): r["total_ms"]["mean"] for r in result["latency"]}
    cmp = [
        (("FP32", "CPU"), ("FP32", "CUDA"), "FP32: CUDA vs CPU"),
        (("FP32", "CPU"), ("INT8 (ORT)", "CPU"), "CPU: INT8 (ORT) vs FP32"),
        (("FP32", "CUDA"), ("INT8 (ORT)", "CUDA"), "CUDA: INT8 (ORT) vs FP32"),
    ]
    found = [(t, tot[a], tot[b]) for a, b, t in cmp if a in tot and b in tot]
    if found:
        lines += ["Total time per frame, ratio (< 1 = faster):", ""]
        lines += [f"- {t}: {b / a:.2f}x ({a:.2f} -> {b:.2f} ms)" for t, a, b in found]
        lines.append("")
    acc = result["accuracy_test_mAP50"]
    if acc:
        lines += [
            "## Accuracy (test split, mAP50)",
            "",
            "| model | mAP50 | hand | ball |",
            "|---|---:|---:|---:|",
        ]
        for name, a in acc.items():
            pc = a["per_class_mAP50"]
            lines.append(
                f"| {name} | {a['mAP50']} | {pc.get('hand', '-')} | {pc.get('ball', '-')} |"
            )
        lines.append("")
    if imx_q.is_file():
        rel = imx_q.relative_to(imx_q.parent.parent.parent).with_suffix(".md").as_posix()
        lines += [
            f"IMX500 INT8 accuracy (MCT, CPU emulation): see [quantization.md]({rel}).",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
