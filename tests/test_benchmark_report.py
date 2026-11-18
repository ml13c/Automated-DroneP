"""The benchmark report is generated from measured JSON only; check its formatting logic."""

import importlib.util

from dronevision.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("benchmark", REPO_ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def _row(model, prov, total):
    ms = {"mean": total, "p50": total, "p95": total}
    return {"model": model, "provider": prov, "size_mb": 1.0, "infer_ms": ms, "total_ms": ms}


def test_report_tables_and_ratios(tmp_path):
    result = {
        "date": "2026-01-01",
        "cpu": "Test CPU",
        "gpu": None,
        "onnxruntime": "1.0",
        "imgsz": 320,
        "images": "n images",
        "latency": [_row("FP32", "CPU", 10.0), _row("INT8 (ORT)", "CPU", 5.0)],
        "accuracy_test_mAP50": {"FP32": {"mAP50": 0.6, "per_class_mAP50": {"hand": 0.9}}},
    }
    md = benchmark.report(result, tmp_path / "missing.json")
    assert "| FP32 | CPU | 1.0 | 10.0 / 10.0 | 10.0 / 10.0 |" in md
    assert "CPU: INT8 (ORT) vs FP32: 0.50x" in md
    assert "CUDA" not in md.split("Total time per frame")[1]  # no ratio without both rows
    assert "| FP32 | 0.6 | 0.9 | - |" in md
    assert "Test CPU" in md and "GPU" not in md.split("\n")[2]
