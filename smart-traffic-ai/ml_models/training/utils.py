"""
ml_models/training/utils.py
=============================
Shared utilities used across YOLOv8, CNN, and congestion training pipelines.

Includes:
    - EarlyStopping callback
    - AverageMeter for loss/accuracy tracking
    - set_seed for reproducibility
    - plot_learning_curves
    - export_metrics_json
"""

import json
import random
import time
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless-safe


# ── Reproducibility ────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set all RNG seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Random seed fixed to {seed}")


# ── Training bookkeeping ───────────────────────────────────────

class AverageMeter:
    """Computes and stores running average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: avg={self.avg:.4f}"


class EarlyStopping:
    """
    Stops training when a monitored metric has not improved for `patience` epochs.

    Usage::

        es = EarlyStopping(patience=10, mode='max', min_delta=0.001)
        for epoch in range(100):
            val_acc = ...
            if es(val_acc):
                print("Early stopping triggered")
                break
    """

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.triggered = False

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True

        return False

    def reset(self):
        self.best = None
        self.counter = 0
        self.triggered = False


class Timer:
    """Context manager + standalone timer."""

    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def elapsed_str(self) -> str:
        s = self.elapsed()
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        elif m:
            return f"{m}m {s}s"
        return f"{s}s"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        pass


# ── Visualisation ──────────────────────────────────────────────

def plot_learning_curves(
    train_vals: list,
    val_vals: list,
    metric_name: str = "Loss",
    save_path: str = "learning_curve.png",
    title: str = "Training Curve",
):
    """Plot train vs val metric over epochs and save to file."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_vals) + 1)
    ax.plot(epochs, train_vals, label=f"Train {metric_name}", color="#E63946", linewidth=2)
    ax.plot(epochs, val_vals,   label=f"Val {metric_name}",   color="#457B9D", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bar_comparison(
    names: list,
    values: list,
    ylabel: str = "Score",
    title: str = "Model Comparison",
    save_path: str = "comparison.png",
    color: str = "#E63946",
):
    """Simple bar chart comparing multiple model scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=color, edgecolor="white", linewidth=0.5, width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(values) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Metrics export ─────────────────────────────────────────────

def export_metrics_json(metrics: dict, path: str):
    """Write a metrics dict to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📋 Metrics saved → {path}")


def load_metrics_json(path: str) -> dict:
    """Load a metrics JSON file. Returns empty dict if not found."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ── Data helpers ───────────────────────────────────────────────

def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    """Simple stratified train/val split without sklearn dependency."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=val_ratio, stratify=y, random_state=seed)


def class_distribution(y: np.ndarray, class_names: list = None) -> dict:
    """Return a dict of class → count."""
    unique, counts = np.unique(y, return_counts=True)
    dist = {}
    for u, c in zip(unique, counts):
        key = class_names[int(u)] if class_names else int(u)
        dist[key] = int(c)
    return dist


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced class weights: n_samples / (n_classes * class_count)."""
    unique, counts = np.unique(y, return_counts=True)
    n = len(y)
    weights = n / (len(unique) * counts)
    # Return ordered by class index
    order = np.argsort(unique)
    return weights[order]
