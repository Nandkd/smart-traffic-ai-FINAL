"""
ml_models/congestion/predict.py
=================================
Real-time congestion prediction using the trained ensemble model.
Accepts raw feature dicts and returns class + probability breakdown.

Usage (standalone):
    python predict.py --vehicle-count 80 --hour 8 --day 0
    python predict.py --interactive
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).parent.parent.parent
WEIGHTS_DIR = ROOT / "ml_models" / "weights"

LABEL_NAMES = {0: "low", 1: "medium", 2: "high"}
LABEL_MAP   = {"low": 0, "medium": 1, "high": 2}

FEATURE_COLS = [
    "vehicle_count", "car_count", "bus_count", "truck_count", "motorcycle_count",
    "hour", "day_of_week", "is_weekend", "is_peak_hour",
    "rain_intensity", "visibility", "incident_nearby",
    "avg_speed_kmh", "hour_sin", "hour_cos",
]


class CongestionPredictor:
    """
    Wraps the saved Voting Ensemble (RF + XGBoost) for real-time predictions.

    Example::

        p = CongestionPredictor()
        result = p.predict({
            "vehicle_count": 95,
            "hour": 8,
            "day_of_week": 0,     # Monday
            "rain_intensity": 0.2,
        })
        print(result["predicted_class"])   # 'high'
    """

    def __init__(self, model_name: str = "ensemble"):
        fname_map = {
            "ensemble": "congestion_ensemble.pkl",
            "random_forest": "random_forest.pkl",
            "xgboost": "xgboost.pkl",
            "logistic_regression": "logistic_regression.pkl",
        }
        path = WEIGHTS_DIR / fname_map.get(model_name, "congestion_ensemble.pkl")
        if path.exists():
            self._model = joblib.load(path)
            self._available = True
            print(f"✅ Congestion model loaded: {path.name}")
        else:
            self._model = None
            self._available = False
            print(f"⚠️  Model not found at {path}. Run train_models.py first.")

        self._model_name = model_name

    # ── Feature engineering ───────────────────────────────────

    @staticmethod
    def build_features(data: dict) -> np.ndarray:
        """Convert raw input dict to the 15-feature vector expected by the model."""
        hour = int(data.get("hour", 12))
        dow  = int(data.get("day_of_week", 0))
        vc   = float(data.get("vehicle_count", 0))
        cc   = float(data.get("car_count", vc * 0.5))
        bc   = float(data.get("bus_count",  vc * 0.1))
        tc   = float(data.get("truck_count", vc * 0.08))
        mc   = float(data.get("motorcycle_count", vc - cc - bc - tc))

        return np.array([[
            vc, cc, bc, tc, mc,
            hour, dow,
            1.0 if dow >= 5 else 0.0,                                # is_weekend
            1.0 if hour in (7, 8, 9, 17, 18, 19, 20) else 0.0,      # is_peak_hour
            float(data.get("rain_intensity", 0.0)),
            float(data.get("visibility", 1.0)),
            float(data.get("incident_nearby", 0)),
            float(data.get("avg_speed_kmh", max(5.0, 60 - vc * 0.3))),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
        ]])

    # ── Prediction ────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        """
        Predict congestion class from a raw feature dict.

        Returns a dict with keys:
            predicted_class, probabilities, confidence,
            recommendation, inference_ms, model_name, available
        """
        t0 = time.time()
        features = self.build_features(data)

        if self._available and self._model is not None:
            # Real model inference
            scaler = getattr(self._model, "_scaler", None)
            X = scaler.transform(features) if scaler else features
            pred_idx = int(self._model.predict(X)[0])
            proba = self._model.predict_proba(X)[0]
            predicted_class = LABEL_NAMES[pred_idx]
            probabilities = {
                "low": round(float(proba[0]), 4),
                "medium": round(float(proba[1]), 4),
                "high": round(float(proba[2]), 4),
            }
            confidence = round(float(proba[pred_idx]), 4)
            mode = "model"
        else:
            # Rule-based fallback
            predicted_class, probabilities = self._rule_based(features[0])
            confidence = probabilities[predicted_class]
            mode = "rule_based_fallback"

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        return {
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "confidence": confidence,
            "recommendation": self._recommendation(predicted_class),
            "suggested_green_extension_sec": {"low": 0, "medium": 20, "high": 40}[predicted_class],
            "inference_ms": elapsed_ms,
            "model_name": self._model_name,
            "mode": mode,
        }

    def predict_batch(self, data_list: list) -> list:
        """Batch prediction for a list of feature dicts."""
        return [self.predict(d) for d in data_list]

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _rule_based(features: np.ndarray):
        """Simple rule-based fallback when no trained model is available."""
        import random
        vc   = features[0]
        hour = int(features[5])
        is_peak = hour in (7, 8, 9, 17, 18, 19, 20)

        if vc > 80 or (is_peak and vc > 50):
            cls = "high"
            probs = [0.05 + random.uniform(0, 0.03),
                     0.12 + random.uniform(0, 0.03),
                     0.83 - random.uniform(0, 0.03)]
        elif vc > 35:
            cls = "medium"
            probs = [0.12 + random.uniform(0, 0.04),
                     0.73 + random.uniform(0, 0.04),
                     0.15 - random.uniform(0, 0.04)]
        else:
            cls = "low"
            probs = [0.81 + random.uniform(0, 0.04),
                     0.14 - random.uniform(0, 0.03),
                     0.05 - random.uniform(0, 0.02)]

        total = sum(probs)
        probs = [round(max(0.01, p / total), 4) for p in probs]
        return cls, {"low": probs[0], "medium": probs[1], "high": probs[2]}

    @staticmethod
    def _recommendation(cls: str) -> str:
        return {
            "low":    "Normal operation. No signal adjustment needed.",
            "medium": "Apply +20s green extension on busiest lane.",
            "high":   "High congestion: +40s extension, consider route diversion.",
        }[cls]


# ── CLI ────────────────────────────────────────────────────────

def interactive_mode(predictor: CongestionPredictor):
    print("\n🚦 Interactive Congestion Predictor")
    print("   Enter feature values (press Enter for defaults)\n")
    while True:
        try:
            data = {
                "vehicle_count": float(input("  Vehicle count [50]: ") or 50),
                "hour":          int(input("  Hour (0-23) [12]: ")    or 12),
                "day_of_week":   int(input("  Day of week (0=Mon) [0]: ") or 0),
                "rain_intensity":float(input("  Rain intensity (0-1) [0]: ") or 0),
                "incident_nearby": int(input("  Incident nearby? (0/1) [0]: ") or 0),
            }
            result = predictor.predict(data)
            print(f"\n  → Class      : {result['predicted_class'].upper()}")
            print(f"    Confidence : {result['confidence']*100:.1f}%")
            print(f"    Probs      : {result['probabilities']}")
            print(f"    Advice     : {result['recommendation']}")
            print(f"    Latency    : {result['inference_ms']} ms\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle-count", type=float, default=50)
    ap.add_argument("--hour",          type=int,   default=12)
    ap.add_argument("--day",           type=int,   default=0)
    ap.add_argument("--rain",          type=float, default=0.0)
    ap.add_argument("--incident",      type=int,   default=0)
    ap.add_argument("--model",         default="ensemble",
                    choices=["ensemble","random_forest","xgboost","logistic_regression"])
    ap.add_argument("--interactive",   action="store_true")
    args = ap.parse_args()

    predictor = CongestionPredictor(model_name=args.model)

    if args.interactive:
        interactive_mode(predictor)
    else:
        result = predictor.predict({
            "vehicle_count": args.vehicle_count,
            "hour": args.hour,
            "day_of_week": args.day,
            "rain_intensity": args.rain,
            "incident_nearby": args.incident,
        })
        print("\n" + "=" * 45)
        print("📊 CONGESTION PREDICTION")
        print("=" * 45)
        print(json.dumps(result, indent=2))
