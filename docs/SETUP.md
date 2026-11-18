# Setup

## Install (Windows or Linux, Python 3.11–3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux: source .venv/bin/activate
python -m pip install --upgrade pip

# NVIDIA GPU: PyTorch with CUDA first (its CUDA major must match onnxruntime-gpu: 1.32 = CUDA 13)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[gpu,train,dev]"

# CPU only (no training):
pip install -e ".[cpu,dev]"
```

Don't install the `cpu` and `gpu` extras together: both provide the `onnxruntime` module.
`python scripts/check_env.py` reports what was found (`--require-gpu` fails without CUDA).

## Run the demo

The demo needs the two ONNX models at the paths in `configs/detector.yaml` and
`configs/classifier.yaml`. Train and export them as shown below.

```powershell
python scripts/run_demo.py --source 0 --control              # full demo on webcam 0
python scripts/run_demo.py --source 0 --sim                  # + 2D drone sim (pip install -e ".[sim]")
python scripts/run_demo.py --source clip.mp4 --ball          # ball tracking on a video
python scripts/run_demo.py --source clip.mp4 --control --start-mode track_ball --save out.mp4
```

1. Print `docs/markers/tag36h11_id0_12cm.pdf` at 100 % (the 5 cm ruler must measure 5 cm). The
   6 cm version is for close-range tests. Cut along the dashed line, glue it flat on card, and
   strap it to your wrist.
2. Show the tag to the camera until your hand box turns green (`USER #id`). Step back: the lock
   follows your hand without the tag. Other hands are grey and ignored.
3. Every mode is locked once entered. Hold `1` for 0.7 s to unlock, then hold the next count
   for 0.7 s: `5` follow hand, `3` track ball, `2` hover, `4` land. Land also works while
   locked; after landing, unlock and pick a mode to take off again. Unlocked with no new
   count for 5 s locks again. In follow hand, the drone moves towards wherever your hand is
   until the hand is centred in its camera. With `--sim`, the sim drone's own camera view is
   used, so it stops once it's in front of your hand (the webcam itself never moves).

| Key | Action |
|---|---|
| `1` | simulated glove: unlock (acts immediately) |
| `2`–`5` | simulated glove: pick a mode after unlocking (acts immediately) |
| space | glove emergency land |
| `g` | glove link off/on (simulated radio loss) |
| `t` | reset to the start mode (hover, locked) |
| `r` | release the user lock |
| `q` / Esc | quit |

Everything that can be tuned lives in `configs/`: modes and failsafes (`modes.yaml`), PID gains
and limits (`controller.yaml`), and tracking and user lock (`tracking.yaml`).

## Train and export the models

```powershell
python scripts/build_dataset.py all          # HaGRIDv2 subset + COCO balls -> data/ (~15k images)
python scripts/train_detector.py --name y11n_320_baseline
python scripts/export_models.py --weights runs/detector/y11n_320_baseline/weights/best.pt --format onnx
python scripts/train_classifier.py --name mnv3s_128_baseline
```

`build_dataset.py all --scale 0.02` is a quick smoke run. Optional held-out test clips:
record them with `scripts/record_clips.py`, label them as YOLO in `data/own/labeled/`, and run
`build_dataset.py own`.

### IMX500 export (Linux / WSL2 only)

```bash
sudo apt-get install -y openjdk-17-jre-headless python3-venv python3-dev libgl1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,cpu,imx]"
python scripts/export_models.py --weights runs/detector/<run>/weights/best.pt --format imx
python scripts/eval_quantized.py --run <run>     # FP32 vs INT8 accuracy table in results/
```

## Develop

```powershell
ruff check . ; ruff format --check . ; pytest
```

Tests that need trained weights, datasets, a webcam or a GPU skip themselves when those are
missing (markers in `tests/conftest.py`), so CI runs the full suite on CPU.
