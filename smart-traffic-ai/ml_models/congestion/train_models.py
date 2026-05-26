"""
ml_models/congestion/train_models.py
======================================
Trains ensemble ML models for traffic congestion prediction.
Models: Random Forest, XGBoost, Logistic Regression, Voting Ensemble.

Usage:
    python train_models.py
    python train_models.py --generate-synthetic  # create synthetic dataset if real data missing
"""

import argparse
import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "datasets" / "processed"
WEIGHTS_DIR = ROOT / "ml_models" / "weights"
PLOTS_DIR = ROOT / "ml_models" / "congestion" / "plots"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ["low", "medium", "high"]
FEATURE_COLS = [
    "vehicle_count", "car_count", "bus_count", "truck_count", "motorcycle_count",
    "hour", "day_of_week", "is_weekend", "is_peak_hour",
    "rain_intensity", "visibility", "incident_nearby",
    "avg_speed_kmh", "hour_sin", "hour_cos",
]


# ── Dataset Generation ─────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate realistic synthetic traffic dataset for training.
    Encodes domain knowledge about peak hours, weekdays, weather effects.
    """
    print(f"⚙️  Generating {n_samples} synthetic traffic records...")
    rng = np.random.default_rng(42)
    now = datetime.utcnow()

    records = []
    for _ in range(n_samples):
        hour = rng.integers(0, 24)
        dow = rng.integers(0, 7)
        is_weekend = int(dow >= 5)
        is_peak = int(hour in [7, 8, 9, 17, 18, 19, 20])
        rain = float(rng.beta(0.5, 5))
        visibility = float(np.clip(1.0 - rain * rng.uniform(0.2, 0.8), 0.2, 1.0))
        incident = int(rng.random() < 0.05)

        # Vehicle count logic
        base = {"peak_weekday": 120, "peak_weekend": 70, "offpeak": 30}
        if is_peak and not is_weekend:
            base_cnt = base["peak_weekday"]
        elif is_peak:
            base_cnt = base["peak_weekend"]
        else:
            base_cnt = base["offpeak"]

        noise = rng.normal(0, 15)
        rain_effect = rain * 20
        incident_effect = incident * 40
        vehicle_count = max(0, int(base_cnt + noise + rain_effect + incident_effect))

        car_count = int(vehicle_count * rng.uniform(0.45, 0.60))
        bus_count = int(vehicle_count * rng.uniform(0.08, 0.15))
        truck_count = int(vehicle_count * rng.uniform(0.05, 0.12))
        motorcycle_count = vehicle_count - car_count - bus_count - truck_count

        avg_speed = max(5, 60 - (vehicle_count * 0.3) - (rain * 15) - (incident * 20))

        # Label assignment
        if vehicle_count < 40:
            label = "low"
        elif vehicle_count < 90:
            label = "medium"
        else:
            label = "high"

        records.append({
            "vehicle_count": vehicle_count,
            "car_count": car_count,
            "bus_count": bus_count,
            "truck_count": truck_count,
            "motorcycle_count": motorcycle_count,
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak,
            "rain_intensity": round(rain, 4),
            "visibility": round(visibility, 4),
            "incident_nearby": incident,
            "avg_speed_kmh": round(avg_speed, 2),
            "hour_sin": round(np.sin(2 * np.pi * hour / 24), 6),
            "hour_cos": round(np.cos(2 * np.pi * hour / 24), 6),
            "density_class": label,
        })

    df = pd.DataFrame(records)
    out_path = DATA_DIR / "traffic_dataset.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset saved → {out_path}")
    print(df["density_class"].value_counts().to_string())
    return df


def load_dataset() -> pd.DataFrame:
    path = DATA_DIR / "traffic_dataset.csv"
    if not path.exists():
        print(f"⚠️  Dataset not found. Generating synthetic data...")
        return generate_synthetic_dataset()
    df = pd.read_csv(path)
    print(f"📂 Loaded dataset: {len(df)} rows")
    return df


def prepare_features(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df["density_class"].map(LABEL_MAP).values
    return X, y


# ── Model Training ─────────────────────────────────────────────

def train_random_forest(X_tr, y_tr) -> RandomForestClassifier:
    print("\n🌲 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    return model


def train_xgboost(X_tr, y_tr) -> XGBClassifier:
    print("⚡ Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr)], verbose=False)
    return model


def train_logistic_regression(X_tr, y_tr) -> LogisticRegression:
    print("📈 Training Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)
    model = LogisticRegression(
        C=1.0, max_iter=500, class_weight="balanced",
         solver="lbfgs", random_state=42
    )
    model.fit(X_scaled, y_tr)
    # Store scaler inside model for later use
    model._scaler = scaler
    return model


def build_ensemble(rf, xgb, lr) -> VotingClassifier:
    print("🗳️  Building Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],  # LR excluded (needs scaling)
        voting="soft",
        weights=[1, 1.5],  # XGBoost slightly higher weight
    )
    return ensemble


# ── Evaluation ─────────────────────────────────────────────────

def evaluate_model(name: str, model, X_te, y_te, is_lr: bool = False):
    if is_lr and hasattr(model, "_scaler"):
        X_te = model._scaler.transform(X_te)
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="weighted")
    auc = roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")

    print(f"\n{'=' * 45}")
    print(f"📊 {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_te, preds, target_names=CLASS_NAMES)}")

    return {"name": name, "accuracy": acc, "f1": f1, "auc": auc, "preds": preds, "proba": proba}


def plot_all_confusion_matrices(results, y_te):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_te, res["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{res['name']}\nAcc={res['accuracy']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.suptitle("Congestion Prediction — Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print("📊 Confusion matrices saved.")


def plot_feature_importance(rf, feature_names):
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[idx], color="#E63946")
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title("Random Forest — Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("📊 Feature importance saved.")


def plot_model_comparison(results):
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#E63946")
    ax.bar(x + width / 2, f1s, width, label="F1-Score", color="#457B9D")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    ax.set_title("Model Comparison — Congestion Prediction")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print("📊 Model comparison saved.")


# ── Main ───────────────────────────────────────────────────────

def main(generate_synthetic: bool = False):
    print("\n🚦 CONGESTION PREDICTION ML PIPELINE")
    print("=" * 50)

    if generate_synthetic:
        df = generate_synthetic_dataset()
    else:
        df = load_dataset()

    X, y = prepare_features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

    # Train models
    rf = train_random_forest(X_tr, y_tr)
    xgb = train_xgboost(X_tr, y_tr)
    lr = train_logistic_regression(X_tr, y_tr)
    ensemble = build_ensemble(rf, xgb, lr)
    ensemble.fit(X_tr, y_tr)

    # Evaluate
    all_results = [
        evaluate_model("Random Forest", rf, X_te, y_te),
        evaluate_model("XGBoost", xgb, X_te, y_te),
        evaluate_model("Logistic Regression", lr, X_te, y_te, is_lr=True),
        evaluate_model("Voting Ensemble", ensemble, X_te, y_te),
    ]

    # Save models
    joblib.dump(rf, WEIGHTS_DIR / "random_forest.pkl")
    joblib.dump(xgb, WEIGHTS_DIR / "xgboost.pkl")
    joblib.dump(lr, WEIGHTS_DIR / "logistic_regression.pkl")
    joblib.dump(ensemble, WEIGHTS_DIR / "congestion_ensemble.pkl")
    print(f"\n✅ Models saved to {WEIGHTS_DIR}")

    # Save metrics JSON
    metrics = {r["name"]: {"accuracy": r["accuracy"], "f1": r["f1"], "auc": r["auc"]}
               for r in all_results}
    with open(WEIGHTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_all_confusion_matrices(all_results, y_te)
    plot_feature_importance(rf, FEATURE_COLS)
    plot_model_comparison(all_results)

    best = max(all_results, key=lambda r: r["accuracy"])
    print(f"\n🏆 Best model: {best['name']} (acc={best['accuracy']:.4f})")
    print("✅ Training pipeline complete!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate-synthetic", action="store_true")
    args = ap.parse_args()
    main(generate_synthetic=args.generate_synthetic)
