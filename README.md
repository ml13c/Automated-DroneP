# Automated-DroneP: gesture drone vision

The "brain" of a gesture-controlled drone, built and benchmarked on a laptop webcam **before**
any drone hardware is bought: custom detector (hand, ball), finger-count classifier, tracking,
single-user lock, mode state machine, simulated velocity commands, benchmarks and a battery
estimate.

> **Status: M0 (setup).** The webcam demo shows live FPS. Models, tracking and mode logic
> come in later milestones. See [PLAN.md](PLAN.md) for the full roadmap.
> Results tables and GIFs are added only from generated, measured outputs.

## Setup (Windows, NVIDIA GPU)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) PyTorch with CUDA first (otherwise pip installs the CPU-only build).
#    The CUDA major version must match onnxruntime-gpu (1.32 = CUDA 13), because ORT reuses
#    the CUDA/cuDNN DLLs that ship with PyTorch. check_env.py verifies this.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2) This package: gpu = onnxruntime-gpu + NVML, train = ultralytics + onnx, dev = pytest + ruff
pip install -e ".[gpu,train,dev]"
```

CPU-only machine / CI: `pip install -e ".[cpu,dev]"` (don't install `cpu` and `gpu` together;
both provide the `onnxruntime` module).

## Run

```powershell
python scripts/check_env.py --require-gpu   # GPU name, CUDA, ONNX Runtime providers, WSL
python scripts/run_demo.py --source 0       # webcam with FPS overlay (q / Esc to quit)
python scripts/run_demo.py --source clip.mp4
```

Camera settings (resolution, backend, mirroring) are in [configs/camera.yaml](configs/camera.yaml).

## Develop

```powershell
ruff check . ; ruff format --check . ; pytest
```

CI (GitHub Actions, CPU only) runs the same checks on every push.

## Layout

```
configs/          YAML settings (no hard-coded thresholds in code)
src/dronevision/  library: sources, profiler, hud, (later) detector, tracker, modes, ...
scripts/          entry points: check_env, run_demo, (later) training, benchmarks, GIFs
tests/            pytest
results/          small committed outputs: CSVs, charts, GIFs
legacy/           earlier MediaPipe / Pi / LIDAR prototypes, kept for reference
```

Datasets (`data/`), weights (`*.pt`, `*.onnx`), training runs and raw recordings are
gitignored. Final weights will be published as GitHub Release assets.
