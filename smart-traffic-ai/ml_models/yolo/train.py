"""
ml_models/yolo/train.py
=======================
YOLOv8 Training Pipeline for Traffic Vehicle Detection.

Usage:
    python train.py --model yolov8s --epochs 100 --imgsz 640 --batch 16
    python train.py --model yolov8n --epochs 50 --imgsz 416 --batch 32  # faster/lighter
"""

import argparse
import os
import yaml
import shutil
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DATA_YAML = ROOT / "ml_models" / "yolo" / "data.yaml"
WEIGHTS_DIR = ROOT / "ml_models" / "weights"
RUNS_DIR = ROOT / "ml_models" / "yolo" / "runs"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s", choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return p.parse_args()


def verify_dataset():
    """Verify that the dataset structure is correct before training."""
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    for split in ["train", "val"]:
        imgs = root / cfg[split]
        labels = Path(str(imgs).replace("images", "labels"))
        assert imgs.exists(), f"Missing images directory: {imgs}"
        assert labels.exists(), f"Missing labels directory: {labels}"

    n_train = len(list((root / cfg["train"]).glob("*.jpg"))) + \
              len(list((root / cfg["train"]).glob("*.png")))
    n_val = len(list((root / cfg["val"]).glob("*.jpg"))) + \
            len(list((root / cfg["val"]).glob("*.png")))

    print(f"✅ Dataset verified — Train: {n_train} | Val: {n_val}")
    return n_train, n_val


def train(args):
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {'GPU (CUDA)' if device == '0' else 'CPU'}")
    print(f"🏋️  Model: {args.model}.pt | Epochs: {args.epochs} | ImgSz: {args.imgsz}")

    # Load base model (pretrained on COCO)
    model = YOLO(f"{args.model}.pt")

    # ── Training ───────────────────────────────────────────────
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=0.01,              # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=args.patience,
        device=device,
        project=str(RUNS_DIR),
        name=f"traffic_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        save=True,
        save_period=10,
        cache=False,           # Set True if RAM allows for faster epochs
        workers=4,
        exist_ok=True,
        pretrained=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        resume=args.resume,
        verbose=True,
    )

    # ── Save best weights to weights/ ─────────────────────────
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        dest = WEIGHTS_DIR / "yolov8_traffic.pt"
        shutil.copy(best_pt, dest)
        print(f"✅ Best weights saved → {dest}")

    return results, run_dir


def evaluate(run_dir: Path):
    """Run validation and print metrics."""
    model = YOLO(run_dir / "weights" / "best.pt")
    metrics = model.val(data=str(DATA_YAML), imgsz=640, conf=0.001, iou=0.6, verbose=True)

    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print(f"  mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print("=" * 50)

    return metrics


def plot_results(run_dir: Path):
    """Generate training plots from results.csv."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print("⚠️  results.csv not found — skipping plots")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("YOLOv8 Training Results — Traffic Detection", fontsize=14, fontweight="bold")

    plots = [
        ("train/box_loss", "Train Box Loss", axes[0, 0]),
        ("train/cls_loss", "Train Class Loss", axes[0, 1]),
        ("val/box_loss", "Val Box Loss", axes[0, 2]),
        ("metrics/precision(B)", "Precision", axes[1, 0]),
        ("metrics/recall(B)", "Recall", axes[1, 1]),
        ("metrics/mAP50(B)", "mAP@0.5", axes[1, 2]),
    ]

    for col, title, ax in plots:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], linewidth=2, color="#E63946")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = run_dir / "training_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training plots saved → {plot_path}")


def export_model(run_dir: Path):
    """Export best model to ONNX and TorchScript for deployment."""
    best_pt = run_dir / "weights" / "best.pt"
    model = YOLO(best_pt)
    model.export(format="onnx", imgsz=640, optimize=True)
    model.export(format="torchscript", imgsz=640)
    print("📦 Model exported to ONNX + TorchScript")


if __name__ == "__main__":
    args = parse_args()

    print("\n🚦 SMART TRAFFIC — YOLOv8 Training Pipeline")
    print("=" * 50)

    # Step 1: Verify dataset
    try:
        verify_dataset()
    except AssertionError as e:
        print(f"⚠️  Dataset warning: {e}")
        print("   Create dataset first: python datasets/prepare_dataset.py")

    # Step 2: Train
    results, run_dir = train(args)

    # Step 3: Evaluate
    evaluate(run_dir)

    # Step 4: Plot
    plot_results(run_dir)

    # Step 5: Export
    export_model(run_dir)

    print("\n✅ Training pipeline complete!")
