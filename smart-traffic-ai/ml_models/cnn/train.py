"""
ml_models/cnn/train.py
=======================
Full CNN training pipeline for ambulance detection.

Usage:
    python train.py --epochs 50 --batch 32 --lr 0.001
"""

import argparse
import os
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from ml_models.cnn.architecture import AmbulanceCNN

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "datasets" / "processed" / "ambulance"
WEIGHTS_DIR = ROOT / "ml_models" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data Augmentation ──────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(data_dir: Path, batch_size: int):
    train_ds = datasets.ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORMS)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=VAL_TRANSFORMS)

    # Handle class imbalance with weighted sampling
    class_counts = np.bincount([s[1] for s in train_ds.samples])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[s[1]] for s in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    print(f"  Classes: {train_ds.classes}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  Class distribution: {dict(zip(train_ds.classes, class_counts))}")
    return train_dl, val_dl, train_ds.classes


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        probs = torch.softmax(logits, 1)[:, 1]
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels, all_probs


def plot_training(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Ambulance Classifier — Training", fontsize=13, fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train Loss", color="#E63946", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#457B9D", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", color="#E63946", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Acc", color="#457B9D", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=150)
    plt.close()
    print(f"📊 Training curves saved.")


def plot_confusion_matrix(labels, preds, classes, save_path: Path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Ambulance CNN")
    plt.tight_layout()
    plt.savefig(save_path / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved.")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚑 Ambulance CNN Training")
    print(f"   Device: {device} | Epochs: {args.epochs} | Batch: {args.batch}")

    if not DATA_DIR.exists():
        print(f"⚠️  Dataset not found at {DATA_DIR}")
        print("   Run: python datasets/prepare_dataset.py --task ambulance")
        return

    train_dl, val_dl, classes = get_dataloaders(DATA_DIR, args.batch)
    model = AmbulanceCNN(num_classes=len(classes)).to(device)

    # Focal-style class weighting
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 4.0]).to(device)  # ambulance gets 4x weight
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler)
        vl_loss, vl_acc, vl_preds, vl_labels, vl_probs = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f} | {elapsed:.1f}s")

        # Save best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            torch.save(model.state_dict(), WEIGHTS_DIR / "ambulance_cnn.pth")
            print(f"   ✅ Best model saved (val_acc={vl_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"   ⏹️  Early stopping at epoch {epoch}")
                break

    # ── Final evaluation ───────────────────────────────────────
    print("\n📊 FINAL METRICS")
    print(classification_report(vl_labels, vl_preds, target_names=classes))
    try:
        auc = roc_auc_score(vl_labels, vl_probs)
        print(f"  ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    # ── Plots ──────────────────────────────────────────────────
    plot_training(history, WEIGHTS_DIR.parent / "cnn")
    plot_confusion_matrix(vl_labels, vl_preds, classes, WEIGHTS_DIR.parent / "cnn")

    # Save training history JSON
    with open(WEIGHTS_DIR.parent / "cnn" / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ CNN Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()
    train(args)
