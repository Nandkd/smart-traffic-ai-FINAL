"""
ml_models/congestion/feature_engineering.py
=============================================
Standalone feature engineering pipeline for congestion prediction.

Transforms raw sensor/detection data → 15-column feature matrix
ready for sklearn / XGBoost training and inference.

Also generates derived features, handles missing values,
and produces a feature importance summary.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ── Constants ──────────────────────────────────────────────────

PEAK_HOURS   = {7, 8, 9, 17, 18, 19, 20}
FEATURE_COLS = [
    "vehicle_count",    # 0  total vehicles in frame
    "car_count",        # 1
    "bus_count",        # 2
    "truck_count",      # 3
    "motorcycle_count", # 4
    "hour",             # 5  0–23
    "day_of_week",      # 6  0=Mon … 6=Sun
    "is_weekend",       # 7  binary
    "is_peak_hour",     # 8  binary
    "rain_intensity",   # 9  0.0–1.0
    "visibility",       # 10 0.0–1.0
    "incident_nearby",  # 11 binary
    "avg_speed_kmh",    # 12 estimated from tracking
    "hour_sin",         # 13 cyclic encoding
    "hour_cos",         # 14 cyclic encoding
]

LABEL_MAP     = {"low": 0, "medium": 1, "high": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ── Feature builder ────────────────────────────────────────────

def engineer_features(raw: dict) -> np.ndarray:
    """
    Convert a raw input dict into a (1, 15) feature array.

    Handles:
    - Missing fields (sensible defaults)
    - Derived vehicle sub-counts
    - Cyclic time encoding
    - Speed estimation from vehicle count

    Parameters
    ----------
    raw : dict  — e.g. from a detection event or API request body

    Returns
    -------
    np.ndarray of shape (1, 15), dtype float64
    """
    vc  = float(raw.get("vehicle_count", 0))
    cc  = float(raw.get("car_count",        vc * 0.50))
    bc  = float(raw.get("bus_count",        vc * 0.10))
    tc  = float(raw.get("truck_count",      vc * 0.08))
    mc  = float(raw.get("motorcycle_count", max(0, vc - cc - bc - tc)))

    now  = datetime.utcnow()
    hour = int(raw.get("hour", now.hour))
    dow  = int(raw.get("day_of_week", now.weekday()))

    rain  = float(raw.get("rain_intensity", 0.0))
    vis   = float(raw.get("visibility",     1.0))
    inc   = float(raw.get("incident_nearby", 0))
    speed = float(raw.get("avg_speed_kmh",
                           max(5.0, 60.0 - vc * 0.30 - rain * 15.0 - inc * 20.0)))

    return np.array([[
        vc, cc, bc, tc, mc,
        hour, dow,
        1.0 if dow >= 5 else 0.0,
        1.0 if hour in PEAK_HOURS else 0.0,
        rain, vis, inc, speed,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
    ]])


def engineer_features_batch(records: list) -> np.ndarray:
    """
    Vectorised feature engineering for a list of raw dicts.
    Returns ndarray of shape (N, 15).
    """
    return np.vstack([engineer_features(r) for r in records])


def engineer_features_df(df: pd.DataFrame) -> np.ndarray:
    """
    Engineer features from a pandas DataFrame.
    Expects columns matching the raw dict keys.
    Returns ndarray of shape (N, 15).
    """
    records = df.to_dict(orient="records")
    return engineer_features_batch(records)


# ── Label helpers ──────────────────────────────────────────────

def encode_labels(labels: list) -> np.ndarray:
    """Convert list of string labels → integer array."""
    return np.array([LABEL_MAP.get(str(l).lower(), 0) for l in labels])


def decode_labels(y: np.ndarray) -> list:
    """Convert integer array → list of string labels."""
    return [LABEL_MAP_INV.get(int(i), "low") for i in y]


def make_density_label(vehicle_count: float) -> str:
    """Simple threshold-based density labelling."""
    if vehicle_count >= 90:
        return "high"
    if vehicle_count >= 40:
        return "medium"
    return "low"


# ── Feature statistics ─────────────────────────────────────────

def feature_summary(X: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame with mean / std / min / max per feature."""
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    summary = df.describe().T[["mean", "std", "min", "max"]]
    summary.index.name = "feature"
    return summary


def detect_outliers(X: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    """
    Return boolean mask of rows that contain at least one outlier
    (Z-score > z_thresh) in any feature column.
    """
    z = np.abs((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
    return z.max(axis=1) > z_thresh


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values with column medians for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


# ── Pipeline class ─────────────────────────────────────────────

class FeaturePipeline:
    """
    Stateful feature pipeline: fits normalization stats on training
    data and applies them at inference time.

    Example::

        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(raw_train_records)
        X_test  = pipeline.transform(raw_test_records)
    """

    def __init__(self):
        self._mean = None
        self._std  = None
        self._fitted = False

    def fit(self, records: list):
        X = engineer_features_batch(records)
        self._mean = X.mean(axis=0)
        self._std  = X.std(axis=0) + 1e-8
        self._fitted = True
        return self

    def transform(self, records: list) -> np.ndarray:
        X = engineer_features_batch(records)
        if self._fitted:
            return (X - self._mean) / self._std
        return X

    def fit_transform(self, records: list) -> np.ndarray:
        return self.fit(records).transform(records)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._fitted:
            return X * self._std + self._mean
        return X

    def save(self, path: str):
        import joblib
        joblib.dump({"mean": self._mean, "std": self._std}, path)

    def load(self, path: str):
        import joblib
        state = joblib.load(path)
        self._mean   = state["mean"]
        self._std    = state["std"]
        self._fitted = True
        return self


# ── Quick test ─────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "vehicle_count": 75,
        "car_count": 40, "bus_count": 8, "truck_count": 6, "motorcycle_count": 21,
        "hour": 8, "day_of_week": 0,
        "rain_intensity": 0.1, "visibility": 0.9, "incident_nearby": 0,
    }
    X = engineer_features(sample)
    print("Feature vector shape:", X.shape)
    print(pd.Series(X[0], index=FEATURE_COLS).to_string())
    print("\nDensity label:", make_density_label(sample["vehicle_count"]))
