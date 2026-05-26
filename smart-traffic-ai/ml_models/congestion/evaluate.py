"""
ml_models/congestion/evaluate.py
==================================
Load all trained congestion models and produce a full evaluation report
with confusion matrices, ROC curves, and comparison tables.

Usage:
    python evaluate.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = ROOT / "ml_models" / "weights"
DATA_DIR = ROOT / "datasets" / "processed"
PLOTS_DIR = ROOT / "ml_models" / "congestion" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["low", "medium", "high"]
LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
FEATURE_COLS = [
    "vehicle_count", "car_count", "bus_count", "truck_count", "motorcycle_count",
    "hour", "day_of_week", "is_weekend", "is_peak_hour",
    "rain_intensity", "visibility", "incident_nearby",
    "avg_speed_kmh", "hour_sin", "hour_cos",
]


def load_test_data():
    csv = DATA_DIR / "traffic_dataset.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Dataset not found: {csv}\nRun: python ml_models/congestion/train_models.py --generate-synthetic")
    df = pd.read_csv(csv)
    # Use last 20% as test set (same split as training)
    n = len(df)
    test_df = df.iloc[int(n * 0.8):]
    X = test_df[FEATURE_COLS].values
    y = test_df["density_class"].map(LABEL_MAP).values
    return X, y


def load_models():
    models = {}
    for name, fname in [
        ("Random Forest", "random_forest.pkl"),
        ("XGBoost", "xgboost.pkl"),
        ("Logistic Regression", "logistic_regression.pkl"),
        ("Voting Ensemble", "congestion_ensemble.pkl"),
    ]:
        path = WEIGHTS_DIR / fname
        if path.exists():
            models[name] = joblib.load(path)
        else:
            print(f"⚠️  {fname} not found — run train_models.py first")
    return models


def plot_roc_curves(models, X_te, y_te):
    y_bin = label_binarize(y_te, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        X_eval = X_te
        if hasattr(model, "_scaler"):
            X_eval = model._scaler.transform(X_te)
        try:
            proba = model.predict_proba(X_eval)
        except Exception:
            continue

        colors = ["#22C55E", "#F59E0B", "#EF4444"]
        for i, cls in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc = roc_auc_score(y_bin[:, i], proba[:, i])
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f"{cls} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title(f"{name}", fontsize=11)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("ROC Curves — Congestion Prediction Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 ROC curves saved.")


def full_evaluation():
    print("\n🔍 CONGESTION MODEL EVALUATION")
    print("=" * 55)

    X_te, y_te = load_test_data()
    models = load_models()

    if not models:
        print("❌ No trained models found. Run train_models.py first.")
        return

    results = {}
    for name, model in models.items():
        X_eval = X_te
        if hasattr(model, "_scaler"):
            X_eval = model._scaler.transform(X_te)

        preds = model.predict(X_eval)
        try:
            proba = model.predict_proba(X_eval)
            auc = roc_auc_score(label_binarize(y_te, classes=[0, 1, 2]), proba,
                                multi_class="ovr", average="weighted")
        except Exception:
            auc = None

        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="weighted")
        results[name] = {"accuracy": acc, "f1": f1, "auc": auc, "preds": preds}

        print(f"\n── {name} ──")
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}" + (f" | AUC: {auc:.4f}" if auc else ""))
        print(classification_report(y_te, preds, target_names=CLASSES, digits=4))

    # ── Comparison table ───────────────────────────────────────
    print("\n📊 COMPARISON TABLE")
    print("-" * 55)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 55)
    for name, r in results.items():
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "  N/A  "
        print(f"{name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {auc_str:>10}")
    print("-" * 55)

    # ── Save JSON ──────────────────────────────────────────────
    save_results = {k: {"accuracy": v["accuracy"], "f1": v["f1"], "auc": v["auc"]}
                    for k, v in results.items()}
    with open(WEIGHTS_DIR / "eval_report.json", "w") as f:
        json.dump(save_results, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────
    plot_roc_curves(models, X_te, y_te)

    # Confusion matrix for best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    cm = confusion_matrix(y_te, results[best_name]["preds"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Best Model: {best_name}\nConfusion Matrix", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "best_model_cm.png", dpi=150)
    plt.close()
    print(f"📊 Best model ({best_name}) confusion matrix saved.")
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    full_evaluation()
