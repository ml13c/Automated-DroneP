# Setup & running

## Install (Windows, NVIDIA GPU)

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

CPU-only machine / CI: `pip install -e ".[cpu,dev]"`. Don't install `cpu` and `gpu` together;
both provide the `onnxruntime` module.

## Check the environment and run the demo

```powershell
python scripts/check_env.py --require-gpu   # GPU, CUDA, ONNX Runtime providers (smoke-tested), WSL
python scripts/run_demo.py --source 0       # webcam with FPS overlay (q / Esc to quit)
python scripts/run_demo.py --source clip.mp4
```

Camera settings are in [configs/camera.yaml](../configs/camera.yaml).

## Build the datasets (M1)

```powershell
python scripts/build_dataset.py all --scale 0.02   # quick smoke run (a few hundred images)
python scripts/build_dataset.py all                # full subset, see configs/dataset.yaml
```

- **HaGRIDv2** ships as one 128 GB zip. The script reads its index once (about 160 MB, cached)
  and then fetches only the sampled images with HTTP range requests.
- **COCO** `sports ball` images are downloaded one by one from `images.cocodataset.org`.
- Everything lands in `data/` (gitignored). Sizes, the gesture → finger-count map and filters
  are in [configs/dataset.yaml](../configs/dataset.yaml).
- Counts per split are written to [results/dataset_summary.md](../results/dataset_summary.md).

### Own test clips (the honest benchmark)

```powershell
python scripts/record_clips.py --distance 2 --place indoor --count 3 --every 10
python scripts/record_clips.py --distance 3 --place outdoor --count - --ball --every 10
```

1. Record clips at 1, 2, 3 and 4 m, indoors and outdoors, some with a ball. Hold up **one**
   counting hand per clip, showing the `--count` you passed.
2. Label the frames in `data/own/frames/` (classes `hand`, `ball`) in CVAT or Roboflow.
   Keep the `<clip>__f<frame>` file-name prefix.
3. Export as YOLO into `data/own/labeled/`, then run `python scripts/build_dataset.py own`.
   This fills `test_own` for the detector and the finger classifier. It is never used for
   training.

## Train the detector (M2)

```powershell
python scripts/train_detector.py --name y11n_320_baseline   # settings: configs/detector.yaml
```

Weights land in `runs/detector/<name>/` (gitignored). The small outputs (metrics JSON, curves,
confusion matrix) are copied to `results/detector/<name>/`.

Export to ONNX and run the live detector (thresholds in `configs/detector.yaml`):

```powershell
python scripts/export_models.py --weights runs/detector/<run>/weights/best.pt --format onnx
python scripts/run_demo.py --source 0 --detector
```

## Train the finger classifier (M3)

```powershell
python scripts/train_classifier.py --name mnv3s_128_baseline   # settings: configs/classifier.yaml
python scripts/run_demo.py --source 0 --fingers                # count above each hand
```

## User lock (M4)

1. Print a tag at 100 % / "actual size" (check the 5 cm ruler), cut along the dashed line, glue
   it flat on cardboard and strap it to your wrist:
   - `docs/markers/tag36h11_id0_12cm.pdf`: the default, readable from further away.
   - `docs/markers/tag36h11_id0_6cm.pdf`: for close-range tests at the desk.

   For another size or ID: `python scripts/make_marker.py --id 0 --size-cm 8` (the ID must be
   in `configs/tracking.yaml`).
2. `python scripts/run_demo.py --source 0 --lock --fingers`
3. Show the tag close to the camera with your hand up until the box turns green ("USER #id").
   Step back: the lock follows your hand without the tag. Other hands are grey and ignored.
   Press `r` to release the lock.

## Full demo: modes + control (M5)

```powershell
python scripts/run_demo.py --source 0 --control
```

Lock on with the tag (as above), then hold up a finger count for 0.7 s to switch mode:
`5` follow hand, `3` track ball, `2` hover, `4` land. The top-right panel shows the mode, why
it changed, the hold-progress bar and the glove link. The bottom-right widget is the 20 Hz
velocity command (arrow = forward/right, arc = yaw, bar = up/down).

| Key | Action |
|---|---|
| `1`–`5` | glove command (acts immediately) |
| space | glove emergency land |
| `g` | glove link off/on (simulated heartbeat loss) |
| `t` | reset after landing (land is latched) |
| `r` | release the user lock |
| `q` / Esc | quit |

Failsafes ([configs/modes.yaml](../configs/modes.yaml)):

- User lost 2 s → hover, 10 s → land (follow hand / hover only).
- Glove link lost 1 s → hover, 5 s → land (all modes).
- Ball lost 3 s in track ball → hover.

Gains and limits are in [configs/controller.yaml](../configs/controller.yaml).

The command is in the drone's frame. The webcam image is mirrored for display, so when your
hand moves to *your* right, the drone (which faces you) yaws to *its* left.

### Ball tracking on a video clip

```powershell
python scripts/run_demo.py --source data/clips/<clip>.mp4 --ball --save recordings/ball.mp4
python scripts/run_demo.py --source data/clips/<clip>.mp4 --control --start-mode track_ball
```

Add `--no-window` to process a clip without showing it.

The test clips used so far are free Mixkit stock videos (mixkit.co, Mixkit License),
downloaded into the gitignored `data/clips/`. For example:
`https://assets.mixkit.co/videos/43482/43482-720.mp4`.

## IMX500 export (WSL2 Ubuntu)

The Sony converter is Linux-only (Python 3.10–3.12, Java 17+). One-time setup:

```bash
sudo apt-get install -y openjdk-17-jre-headless python3-venv python3-dev libgl1
python3 -m venv ~/venvs/dronevision && . ~/venvs/dronevision/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e "/mnt/c/<path to repo>[train,cpu,imx]"   # imx extra pins pulp<4 for MCT
```

Then, from the repo folder under `/mnt/c/...`:

```bash
python scripts/export_models.py --weights runs/detector/<run>/weights/best.pt --format imx
```

The first export installs Sony's toolchain (model-compression-toolkit, imx500-converter).
File sizes and the converter's memory report go to `results/detector/<run>/imx_export.json`.

## Develop

```powershell
ruff check . ; ruff format --check . ; pytest
```

CI (GitHub Actions, CPU only) runs the same checks on every push.
