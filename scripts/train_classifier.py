"""Train the finger-count classifier (MobileNetV3-Small) on the local GPU, export to ONNX.

    python scripts/train_classifier.py --name mnv3s_128_baseline
    python scripts/train_classifier.py --name quick --epochs 1     # smoke test

Writes runs/classifier/<name>/ (gitignored: best.pt, best.onnx) and results/classifier/<name>/:
train_summary.json (time, peak VRAM, sizes, val/test accuracy for PyTorch AND ONNX Runtime,
per-class precision/recall), history.csv and a confusion matrix PNG. All numbers measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np

from dronevision.config import REPO_ROOT, load_config
from dronevision.datasets import IMAGE_EXTS
from dronevision.fingers import MEAN, STD, preprocess


def list_split(root: Path, classes: list[str]) -> list[tuple[str, int]]:
    items = []
    for k, c in enumerate(classes):
        d = root / c
        if d.is_dir():
            items += [(str(p), k) for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return items


class Crops:
    """Module-level (picklable) dataset: Windows DataLoader workers are spawned processes."""

    def __init__(self, items, size: int, transform=None):
        self.items, self.size, self.transform = items, size, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch

        path, label = self.items[i]
        bgr = cv2.imread(path)
        if self.transform is not None:
            t = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)))
            return self.transform(t), label
        return torch.from_numpy(preprocess(bgr, self.size)[0]), label  # exactly as at inference


def train_transform(cfg: dict):
    import torch
    from torchvision.transforms import v2

    size, aug = cfg["model"]["input"], cfg["train"]["aug"]
    return v2.Compose(
        [
            v2.RandomResizedCrop(size, scale=tuple(aug["scale"]), antialias=True),
            v2.RandomRotation(aug["rotate_deg"]),
            v2.RandomHorizontalFlip(aug["hflip"]),
            v2.ColorJitter(*aug["color_jitter"]),
            v2.RandomGrayscale(aug["grayscale"]),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=aug["blur"]),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(MEAN.tolist(), STD.tolist()),
        ]
    )


def build_model(cfg: dict):
    from torch import nn
    from torchvision import models

    weights = "DEFAULT" if cfg["model"]["pretrained"] else None
    net = getattr(models, cfg["model"]["arch"])(weights=weights)
    head = net.classifier[-1]
    net.classifier[-1] = nn.Linear(head.in_features, len(cfg["classes"]))
    return net


def evaluate_torch(net, loader, device, n_classes: int) -> np.ndarray:
    import torch

    cm = np.zeros((n_classes, n_classes), np.int64)
    net.eval()
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for x, y in loader:
            pred = net(x.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred, strict=True):
                cm[t, p] += 1
    return cm


def evaluate_onnx(onnx_path: Path, items, size: int, n_classes: int) -> tuple[np.ndarray, float]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    cm = np.zeros((n_classes, n_classes), np.int64)
    times = []
    for path, label in items:
        x = preprocess(cv2.imread(path), size)
        t0 = time.perf_counter()
        logits = sess.run(None, {name: x})[0]
        times.append(time.perf_counter() - t0)
        cm[label, int(logits.argmax())] += 1
    return cm, float(np.median(times) * 1e3)


def report(cm: np.ndarray, classes: list[str]) -> dict:
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    return {
        "accuracy": round(float(tp.sum() / max(cm.sum(), 1)), 4),
        "per_class": {
            c: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "n": int(cm[i].sum()),
            }
            for i, (c, p, r) in enumerate(zip(classes, prec, rec, strict=True))
        },
    }


def plot_cm(cm: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def export_onnx(net, size: int, classes: list[str], out: Path) -> None:
    import onnx
    import torch

    net = net.float().cpu().eval()
    torch.onnx.export(
        net,
        torch.zeros(1, 3, size, size),
        str(out),
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    m = onnx.load(str(out))
    for k, v in {"classes": repr(classes), "input": str(size), "task": "finger_count"}.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(out))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--config", default="classifier")
    p.add_argument("--epochs", type=int)
    args = p.parse_args(argv)

    import torch
    from torch import nn

    cfg = load_config(args.config)
    tr, classes = cfg["train"], [str(c) for c in cfg["classes"]]
    epochs = args.epochs or tr["epochs"]
    random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    torch.manual_seed(tr["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = REPO_ROOT / cfg["data"]
    splits = {s: list_split(root / s, classes) for s in ("train", "val", "test")}
    for s, items in splits.items():
        print(f"{s}: {len(items)} crops")
    size = cfg["model"]["input"]
    net = build_model(cfg)
    tf = train_transform(cfg)
    net.to(device)

    def loader(split: str, train: bool):
        return torch.utils.data.DataLoader(
            Crops(splits[split], size, tf if train else None),
            batch_size=tr["batch"],
            shuffle=train,
            num_workers=tr["workers"],
            persistent_workers=tr["workers"] > 0,
            pin_memory=device.type == "cuda",
            drop_last=train,
        )

    train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tr["lr"], total_steps=epochs * len(train_dl), pct_start=0.1
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=tr["label_smoothing"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    run_dir = REPO_ROOT / cfg["project"] / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    res_dir = REPO_ROOT / "results" / "classifier" / args.name
    res_dir.mkdir(parents=True, exist_ok=True)
    history, best_acc, best_epoch = [], -1.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        val_acc = report(evaluate_torch(net, val_dl, device, len(classes)), classes)["accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total / n, 4),
                "val_acc": val_acc,
                "epoch_s": round(time.time() - t0, 1),
            }
        )
        print(
            f"epoch {epoch}/{epochs} loss {total / n:.4f} val_acc {val_acc:.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(net.state_dict(), run_dir / "best.pt")
    wall = time.time() - t_start
    peak = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else None

    net.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    cm_val = evaluate_torch(net, val_dl, device, len(classes))
    cm_test = evaluate_torch(net, test_dl, device, len(classes))
    onnx_path = run_dir / "best.onnx"
    export_onnx(net, size, classes, onnx_path)
    cm_test_onnx, onnx_ms = evaluate_onnx(onnx_path, splits["test"], size, len(classes))

    plot_cm(
        cm_test_onnx,
        classes,
        f"{args.name}: test (public HaGRID crops), ONNX Runtime CPU",
        res_dir / "confusion_matrix_test.png",
    )
    with (res_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    summary = {
        "run": args.name,
        "model": cfg["model"],
        "train": {**tr, "epochs": epochs},
        "best_epoch": best_epoch,
        "wall_time_min": round(wall / 60, 1),
        "peak_vram_reserved_gib": round(peak, 2) if peak else None,
        "params": sum(p.numel() for p in net.parameters()),
        "onnx_mb": round(onnx_path.stat().st_size / 2**20, 2),
        "val_torch": report(cm_val, classes),
        "test_torch": report(cm_test, classes),
        "test_onnx_cpu": report(cm_test_onnx, classes),
        "onnx_cpu_median_ms_per_crop": round(onnx_ms, 2),
        "confusion_matrix_test_onnx": cm_test_onnx.tolist(),
        "classes": classes,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "cpu": platform.processor(),
        "versions": {"torch": torch.__version__, "python": platform.python_version()},
        "note": "test = HaGRID test-split crops (public, subject-disjoint). Not own clips.",
    }
    (res_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "best_epoch",
                    "wall_time_min",
                    "peak_vram_reserved_gib",
                    "onnx_mb",
                    "onnx_cpu_median_ms_per_crop",
                )
            },
            indent=2,
        )
    )
    print(
        "test acc torch",
        summary["test_torch"]["accuracy"],
        "| onnx",
        summary["test_onnx_cpu"]["accuracy"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
