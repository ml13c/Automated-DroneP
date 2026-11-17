"""Print what this machine can run: Python, OpenCV, PyTorch/CUDA, ONNX Runtime providers, GPU, WSL.

    python scripts/check_env.py               # report only
    python scripts/check_env.py --require-gpu # exit 1 if PyTorch can't use CUDA

Missing optional packages are reported, not fatal, so this also runs in CPU-only CI.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import subprocess
import sys


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001 - report any import failure
        return e


def _row(label: str, value: object) -> None:
    print(f"  {label:<26} {value}")


def check_python() -> None:
    print("Python")
    _row("version", sys.version.split()[0])
    _row("executable", sys.executable)
    _row("platform", platform.platform())
    for mod in ("numpy", "cv2", "yaml", "psutil", "ultralytics", "onnx"):
        m = _try_import(mod)
        _row(
            mod, getattr(m, "__version__", "?") if not isinstance(m, Exception) else "not installed"
        )


def check_torch() -> bool:
    print("PyTorch")
    torch = _try_import("torch")
    if isinstance(torch, Exception):
        _row("torch", "not installed")
        return False
    _row("torch", torch.__version__)
    _row("built for CUDA", torch.version.cuda or "no (CPU-only build)")
    cuda = torch.cuda.is_available()
    _row("cuda available", cuda)
    if cuda:
        props = torch.cuda.get_device_properties(0)
        _row("device", props.name)
        _row("compute capability", f"{props.major}.{props.minor}")
        _row("VRAM (GiB)", f"{props.total_memory / 2**30:.1f}")
        _row("cuDNN", torch.backends.cudnn.version())
        # Real check, not just availability: run a tiny matmul on the GPU.
        x = torch.randn(256, 256, device="cuda")
        _row("matmul on GPU", "ok" if torch.isfinite(x @ x).all().item() else "FAILED")
    return cuda


def check_onnxruntime() -> None:
    print("ONNX Runtime")
    ort = _try_import("onnxruntime")
    if isinstance(ort, Exception):
        _row("onnxruntime", "not installed")
        return
    _row("version", ort.__version__)
    if hasattr(ort, "preload_dlls"):
        # onnxruntime-gpu >= 1.21: reuse the CUDA/cuDNN DLLs shipped with PyTorch.
        try:
            ort.preload_dlls()
        except Exception as e:  # noqa: BLE001
            _row("preload_dlls", f"failed: {e}")
    providers = ort.get_available_providers()
    _row("compiled-in providers", ", ".join(providers))

    onnx = _try_import("onnx")
    if isinstance(onnx, Exception):
        _row("provider smoke test", "skipped (install `onnx` to run it)")
        return
    import numpy as np
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    blob = model.SerializeToString()
    x = np.array([[-1, 0, 1, 2]], dtype=np.float32)
    for prov in providers:
        if prov not in ("CPUExecutionProvider", "CUDAExecutionProvider"):
            continue
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(blob, so, providers=[prov])
            used = sess.get_providers()[0]
            y = sess.run(None, {"x": x})[0]
            ok = used == prov and np.allclose(y, np.maximum(x, 0))
            _row(
                f"run on {prov.replace('ExecutionProvider', '')}",
                "ok" if ok else f"FAILED (fell back to {used})",
            )
        except Exception as e:  # noqa: BLE001
            _row(f"run on {prov.replace('ExecutionProvider', '')}", f"FAILED: {e}")


def check_nvml() -> None:
    print("NVIDIA driver (NVML)")
    pynvml = _try_import("pynvml")
    if isinstance(pynvml, Exception):
        _row("pynvml", "not installed (pip install nvidia-ml-py)")
        return
    try:
        pynvml.nvmlInit()
    except Exception as e:  # noqa: BLE001
        _row("nvml", f"unavailable: {e}")
        return
    try:
        _row("driver", pynvml.nvmlSystemGetDriverVersion())
        cuda_v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        _row("max CUDA (driver)", f"{cuda_v // 1000}.{(cuda_v % 1000) // 10}")
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        _row("gpu", pynvml.nvmlDeviceGetName(h))
        try:
            _row("power reading (W)", f"{pynvml.nvmlDeviceGetPowerUsage(h) / 1000:.1f} (now)")
        except pynvml.NVMLError as e:
            _row("power reading", f"not supported: {e}")
    finally:
        pynvml.nvmlShutdown()


def check_wsl() -> None:
    if platform.system() != "Windows":
        return
    print("WSL")
    if not shutil.which("wsl"):
        _row("wsl", "not found")
        return
    try:
        out = subprocess.run(["wsl", "-l", "-q"], capture_output=True, timeout=20).stdout
        distros = [d.strip() for d in out.decode("utf-16-le", errors="ignore").splitlines()]
        distros = [d.replace("\x00", "") for d in distros if d.strip("\x00 ")]
        _row("distros", ", ".join(distros) or "none installed")
    except Exception as e:  # noqa: BLE001
        _row("wsl", f"error: {e}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--require-gpu", action="store_true", help="exit 1 if CUDA is unusable")
    args = p.parse_args(argv)

    check_python()
    cuda = check_torch()
    check_onnxruntime()
    check_nvml()
    check_wsl()

    if args.require_gpu and not cuda:
        print("\nFAIL: PyTorch cannot use a CUDA GPU.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
