"""
tests/test_ml_models.py
========================
Unit tests for ML model components — CNN architecture,
feature engineering, congestion predictor, and training utilities.

Run: pytest tests/test_ml_models.py -v
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── CNN Architecture ───────────────────────────────────────────

class TestAmbulanceCNN:
    def test_import(self):
        from ml_models.cnn.architecture import AmbulanceCNN
        model = AmbulanceCNN()
        assert model is not None

    def test_output_shape(self):
        import torch
        from ml_models.cnn.architecture import AmbulanceCNN
        model = AmbulanceCNN(num_classes=2)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"

    def test_predict_proba_shape(self):
        import torch
        from ml_models.cnn.architecture import AmbulanceCNN
        model = AmbulanceCNN()
        x = torch.randn(2, 3, 224, 224)
        proba = model.predict_proba(x)
        assert proba.shape == (2, 2)

    def test_softmax_sums_to_one(self):
        import torch
        from ml_models.cnn.architecture import AmbulanceCNN
        model = AmbulanceCNN()
        x = torch.randn(3, 3, 224, 224)
        proba = model.predict_proba(x)
        sums = proba.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, np.ones(3), atol=1e-5)

    def test_parameter_count(self):
        from ml_models.cnn.architecture import AmbulanceCNN
        model = AmbulanceCNN()
        n = sum(p.numel() for p in model.parameters())
        # Should be roughly 4M parameters
        assert 1_000_000 < n < 20_000_000, f"Unexpected param count: {n:,}"


# ── Feature Engineering ────────────────────────────────────────

class TestFeatureEngineering:
    def test_build_features_shape(self):
        from ml_models.congestion.predict import CongestionPredictor
        feats = CongestionPredictor.build_features({
            "vehicle_count": 50,
            "hour": 8,
            "day_of_week": 0,
        })
        assert feats.shape == (1, 15), f"Expected (1,15), got {feats.shape}"

    def test_feature_values_range(self):
        from ml_models.congestion.predict import CongestionPredictor
        feats = CongestionPredictor.build_features({
            "vehicle_count": 100,
            "car_count": 60,
            "bus_count": 10,
            "truck_count": 8,
            "motorcycle_count": 22,
            "hour": 18,
            "day_of_week": 1,
            "rain_intensity": 0.3,
            "visibility": 0.8,
            "incident_nearby": 1,
        })
        # hour_sin and hour_cos should be in [-1, 1]
        assert -1.01 <= feats[0, 13] <= 1.01  # hour_sin
        assert -1.01 <= feats[0, 14] <= 1.01  # hour_cos

    def test_weekend_flag(self):
        from ml_models.congestion.predict import CongestionPredictor
        # Saturday = 5
        feats = CongestionPredictor.build_features({"hour": 12, "day_of_week": 5})
        assert feats[0, 7] == 1.0  # is_weekend flag

    def test_weekday_flag(self):
        from ml_models.congestion.predict import CongestionPredictor
        # Monday = 0
        feats = CongestionPredictor.build_features({"hour": 12, "day_of_week": 0})
        assert feats[0, 7] == 0.0  # is_weekend flag

    def test_peak_hour_flag(self):
        from ml_models.congestion.predict import CongestionPredictor
        for peak_h in [7, 8, 9, 17, 18, 19, 20]:
            feats = CongestionPredictor.build_features({"hour": peak_h, "day_of_week": 0})
            assert feats[0, 8] == 1.0, f"Hour {peak_h} should be peak"

    def test_offpeak_flag(self):
        from ml_models.congestion.predict import CongestionPredictor
        feats = CongestionPredictor.build_features({"hour": 3, "day_of_week": 0})
        assert feats[0, 8] == 0.0  # not peak


# ── Congestion Predictor (rule-based fallback) ─────────────────

class TestCongestionPredictor:
    @pytest.fixture(autouse=True)
    def predictor(self):
        from ml_models.congestion.predict import CongestionPredictor
        # Intentionally use a non-existent weights path → rule-based fallback
        self.p = CongestionPredictor.__new__(CongestionPredictor)
        self.p._model = None
        self.p._available = False
        self.p._model_name = "test_fallback"

    def test_predict_returns_valid_class(self):
        result = self.p.predict({"vehicle_count": 60, "hour": 8, "day_of_week": 0})
        assert result["predicted_class"] in ("low", "medium", "high")

    def test_predict_probabilities_sum_to_one(self):
        result = self.p.predict({"vehicle_count": 40, "hour": 12, "day_of_week": 2})
        probs = result["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}"

    def test_high_traffic_prediction(self):
        result = self.p.predict({"vehicle_count": 150, "hour": 8, "day_of_week": 0})
        assert result["predicted_class"] == "high"

    def test_low_traffic_prediction(self):
        result = self.p.predict({"vehicle_count": 5, "hour": 3, "day_of_week": 0})
        assert result["predicted_class"] == "low"

    def test_predict_has_recommendation(self):
        result = self.p.predict({"vehicle_count": 50})
        assert "recommendation" in result
        assert isinstance(result["recommendation"], str)

    def test_predict_has_inference_ms(self):
        result = self.p.predict({"vehicle_count": 30})
        assert "inference_ms" in result
        assert result["inference_ms"] >= 0

    def test_batch_prediction_length(self):
        inputs = [{"vehicle_count": v, "hour": 12} for v in [10, 50, 100]]
        results = self.p.predict_batch(inputs)
        assert len(results) == 3

    def test_confidence_in_range(self):
        result = self.p.predict({"vehicle_count": 70, "hour": 17, "day_of_week": 1})
        assert 0.0 <= result["confidence"] <= 1.0


# ── Training Utilities ─────────────────────────────────────────

class TestTrainingUtils:
    def test_average_meter(self):
        from ml_models.training.utils import AverageMeter
        m = AverageMeter("loss")
        m.update(1.0)
        m.update(3.0)
        assert m.avg == pytest.approx(2.0)
        assert m.count == 2

    def test_average_meter_weighted(self):
        from ml_models.training.utils import AverageMeter
        m = AverageMeter()
        m.update(2.0, n=4)  # sum=8
        m.update(6.0, n=2)  # sum=12
        assert m.avg == pytest.approx(12.0 / 6)

    def test_early_stopping_max_mode(self):
        from ml_models.training.utils import EarlyStopping
        es = EarlyStopping(patience=3, mode="max")
        assert not es(0.80)
        assert not es(0.85)   # improved
        assert not es(0.84)   # counter=1
        assert not es(0.83)   # counter=2
        assert es(0.82)       # counter=3 → triggered

    def test_early_stopping_resets_on_improvement(self):
        from ml_models.training.utils import EarlyStopping
        es = EarlyStopping(patience=3, mode="max")
        es(0.80)
        es(0.78)   # counter=1
        es(0.90)   # improved → counter resets
        assert es.counter == 0

    def test_set_seed_numpy(self):
        from ml_models.training.utils import set_seed
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_class_distribution(self):
        from ml_models.training.utils import class_distribution
        y = np.array([0, 0, 1, 1, 1, 2])
        dist = class_distribution(y, ["low", "medium", "high"])
        assert dist["low"] == 2
        assert dist["medium"] == 3
        assert dist["high"] == 1

    def test_compute_class_weights_balanced(self):
        from ml_models.training.utils import compute_class_weights
        # Equal classes → equal weights
        y = np.array([0, 0, 1, 1, 2, 2])
        w = compute_class_weights(y)
        np.testing.assert_allclose(w[0], w[1], rtol=1e-5)
        np.testing.assert_allclose(w[1], w[2], rtol=1e-5)

    def test_compute_class_weights_imbalanced(self):
        from ml_models.training.utils import compute_class_weights
        # Rare class should get higher weight
        y = np.array([0] * 90 + [1] * 10)
        w = compute_class_weights(y)
        assert w[1] > w[0], "Rare class should have higher weight"

    def test_timer_elapsed(self):
        import time
        from ml_models.training.utils import Timer
        t = Timer()
        t.start()
        time.sleep(0.05)
        assert t.elapsed() >= 0.04

    def test_timer_context_manager(self):
        import time
        from ml_models.training.utils import Timer
        with Timer() as t:
            time.sleep(0.02)
        assert t.elapsed() >= 0.01


# ── Augmentation ───────────────────────────────────────────────

class TestAugmentation:
    def test_horizontal_flip_shape(self):
        from ml_models.training.augmentation import random_horizontal_flip
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = random_horizontal_flip(img, p=1.0)
        assert out.shape == img.shape

    def test_horizontal_flip_content(self):
        from ml_models.training.augmentation import random_horizontal_flip
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[:, 0, :] = 255  # left column white
        out = random_horizontal_flip(img, p=1.0)
        assert out[:, -1, 0].mean() == 255  # right column should now be white

    def test_brightness_contrast_range(self):
        from ml_models.training.augmentation import random_brightness_contrast
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        out = random_brightness_contrast(img)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_augment_numpy_pipeline(self):
        from ml_models.training.augmentation import augment_image_numpy
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = augment_image_numpy(img)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.uint8

    def test_cnn_val_transform(self):
        from ml_models.training.augmentation import get_cnn_val_transforms
        from PIL import Image
        tf = get_cnn_val_transforms(224)
        pil = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = tf(pil)
        assert tensor.shape == (3, 224, 224)

    def test_mixup_output_shape(self):
        from ml_models.training.augmentation import mixup
        x1 = np.random.rand(224, 224, 3).astype(np.float32)
        x2 = np.random.rand(224, 224, 3).astype(np.float32)
        mixed_x, mixed_y = mixup(x1, 0, x2, 1, num_classes=2)
        assert mixed_x.shape == (224, 224, 3)
        assert mixed_y.shape == (2,)
        assert abs(mixed_y.sum() - 1.0) < 1e-5
