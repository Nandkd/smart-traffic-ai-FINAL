"""
datasets/prepare_dataset.py
============================
Downloads / organises traffic + ambulance datasets for training.

Usage:
    python prepare_dataset.py --task yolo
    python prepare_dataset.py --task ambulance
    python prepare_dataset.py --task all
    python prepare_dataset.py --task synthetic   # generates synthetic CSV only
"""

import argparse
import os
import shutil
import random
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent.parent
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
ANNOTATIONS = ROOT / "datasets" / "annotations"
AUGMENTED = ROOT / "datasets" / "augmented"

for d in [RAW, PROCESSED, ANNOTATIONS, AUGMENTED]:
    d.mkdir(parents=True, exist_ok=True)

# ── YOLO dataset structure ─────────────────────────────────────

def create_yolo_structure():
    """Create YOLO-format directory tree with placeholder images and labels."""
    print("📁 Creating YOLO dataset structure...")
    for split in ["train", "val", "test"]:
        (PROCESSED / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Generate synthetic YOLO images + labels for demo
    classes = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck", 4: "ambulance"}
    splits = {"train": 800, "val": 150, "test": 50}
    class_weights = [0.50, 0.15, 0.12, 0.12, 0.11]  # ambulance rare

    for split, n in splits.items():
        img_dir = PROCESSED / "images" / split
        lbl_dir = PROCESSED / "labels" / split
        for i in range(n):
            # Blank 640×640 image (white + noise)
            img = Image.new("RGB", (640, 640), color=(random.randint(80,130),)*3)
            draw = ImageDraw.Draw(img)

            n_objects = random.randint(1, 8)
            labels = []
            for _ in range(n_objects):
                cls = random.choices(list(classes.keys()), weights=class_weights)[0]
                cx = random.uniform(0.1, 0.9)
                cy = random.uniform(0.1, 0.9)
                w  = random.uniform(0.05, 0.2)
                h  = random.uniform(0.04, 0.15)
                x1 = int((cx - w/2) * 640)
                y1 = int((cy - h/2) * 640)
                x2 = int((cx + w/2) * 640)
                y2 = int((cy + h/2) * 640)
                c = (0,200,0) if cls != 4 else (255,0,0)
                draw.rectangle([x1,y1,x2,y2], outline=c, width=2)
                labels.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            img.save(img_dir / f"frame_{i:05d}.jpg", quality=85)
            (lbl_dir / f"frame_{i:05d}.txt").write_text("\n".join(labels))

    print(f"✅ YOLO dataset: {sum(splits.values())} synthetic images with labels")
    print("   Replace with real traffic images from:")
    print("   • VisDrone2019: https://github.com/VisDrone/VisDrone-Dataset")
    print("   • COCO (vehicles): https://cocodataset.org")
    print("   • UA-DETRAC: http://detrac-db.rit.albany.edu")


def create_ambulance_structure():
    """Create ImageFolder-style ambulance dataset."""
    print("📁 Creating ambulance CNN dataset...")
    for split in ["train", "val"]:
        for cls in ["ambulance", "non_ambulance"]:
            (PROCESSED / "ambulance" / split / cls).mkdir(parents=True, exist_ok=True)

    # Generate placeholder images
    for split, n in [("train", 400), ("val", 80)]:
        # ambulance images (red-tinted, wide)
        for i in range(n // 2):
            img = Image.new("RGB", (224, 224),
                color=(random.randint(200,255), random.randint(0,50), random.randint(0,50)))
            draw = ImageDraw.Draw(img)
            draw.rectangle([20,60,200,160], fill=(255,255,255))
            draw.text((80,90), "AMB", fill=(255,0,0))
            img.save(PROCESSED / "ambulance" / split / "ambulance" / f"amb_{i:04d}.jpg")

        # non-ambulance images (varied)
        for i in range(n // 2):
            color = tuple(random.randint(30,200) for _ in range(3))
            img = Image.new("RGB", (224,224), color=color)
            draw = ImageDraw.Draw(img)
            draw.rectangle([20,50,190,170], fill=tuple(c+20 for c in color))
            img.save(PROCESSED / "ambulance" / split / "non_ambulance" / f"nonamb_{i:04d}.jpg")

    print("✅ Ambulance dataset created (synthetic)")
    print("   Replace with real images from:")
    print("   • Google Images / Kaggle ambulance datasets")
    print("   • Roboflow Universe: roboflow.com/universe")


def create_synthetic_traffic_csv():
    """Generate synthetic traffic CSV for congestion ML training."""
    from ml_models.congestion.train_models import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_samples=15000)
    print(f"✅ Synthetic traffic CSV: {len(df)} rows → {PROCESSED / 'traffic_dataset.csv'}")


def create_dataset_summary():
    """Write a JSON summary of all datasets."""
    summary = {
        "yolo": {
            "train": len(list((PROCESSED / "images" / "train").glob("*.jpg"))),
            "val": len(list((PROCESSED / "images" / "val").glob("*.jpg"))),
            "test": len(list((PROCESSED / "images" / "test").glob("*.jpg"))),
            "classes": ["car", "motorcycle", "bus", "truck", "ambulance"],
        },
        "ambulance_cnn": {
            "train_amb": len(list((PROCESSED / "ambulance" / "train" / "ambulance").glob("*.jpg"))),
            "train_nonamb": len(list((PROCESSED / "ambulance" / "train" / "non_ambulance").glob("*.jpg"))),
            "val_amb": len(list((PROCESSED / "ambulance" / "val" / "ambulance").glob("*.jpg"))),
        },
        "congestion": {
            "csv_path": str(PROCESSED / "traffic_dataset.csv"),
            "features": 15,
            "classes": ["low", "medium", "high"],
        }
    }
    out = ROOT / "datasets" / "dataset_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"📋 Dataset summary saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
        choices=["all","yolo","ambulance","synthetic"])
    args = ap.parse_args()

    print("\n📦 Dataset Preparation Pipeline")
    print("=" * 45)

    if args.task in ("all", "yolo"):
        create_yolo_structure()

    if args.task in ("all", "ambulance"):
        create_ambulance_structure()

    if args.task in ("all", "synthetic"):
        try:
            create_synthetic_traffic_csv()
        except ImportError:
            print("⚠️  Skipping CSV generation — run from project root with: python -m datasets.prepare_dataset")

    if args.task == "all":
        try:
            create_dataset_summary()
        except Exception:
            pass

    print("\n✅ Dataset preparation complete!")
    print("   Next: python ml_models/congestion/train_models.py --generate-synthetic")
    print("   Then: python ml_models/yolo/train.py")
    print("   Then: python ml_models/cnn/train.py")
