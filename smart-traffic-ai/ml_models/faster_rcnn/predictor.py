"""
ml_models/faster_rcnn/predictor.py
====================================
Secondary ambulance verification using the fine-tuned Faster R-CNN model.

Role in the pipeline
--------------------
Stage 1  — YOLOv8 (best.pt) detects all vehicles and flags potential ambulances.
Stage 2  — THIS MODULE re-examines every YOLO-flagged crop.
           Only a CLASS-2 (Ambulance) detection with score >= threshold confirms.
           Class-1 (Vehicle) detections are explicitly ignored — they never
           trigger the emergency override.

Trained class mapping  (FIXED — do not change without retraining)
------------------------------------------------------------------
    Index 0  →  Background   (never triggers anything)
    Index 1  →  Vehicle      (suppressed — must NOT trigger override)
    Index 2  →  Ambulance    (ONLY class that can confirm emergency)

Architecture
------------
    torchvision.models.detection.fasterrcnn_resnet50_fpn
    num_classes = 3  (background + vehicle + ambulance)

    FastRCNNPredictor head sizes that must match the checkpoint:
        cls_score  weight : [3, in_features]   (3 rows, one per class)
        bbox_pred  weight : [12, in_features]  (4 coords × 3 classes)

Standalone smoke-test
---------------------
    python -m ml_models.faster_rcnn.predictor --image path/to/crop.jpg
    python -m ml_models.faster_rcnn.predictor --image crop.jpg --show-all-detections
"""

import os
import time
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Class mapping — hardcoded to match training, never read from env vars
# ---------------------------------------------------------------------------

# All three classes the model was trained on.
CLASS_MAP = {
    0: "background",
    1: "vehicle",
    2: "ambulance",
}

# The ONLY label index that may confirm an ambulance and fire the override.
AMBULANCE_LABEL = 2

# Label index for generic vehicles — must never trigger emergency logic.
VEHICLE_LABEL = 1

# Total output classes (background + vehicle + ambulance).
NUM_CLASSES = 3

# ---------------------------------------------------------------------------
# Tunable constants (safe to override via environment variables)
# ---------------------------------------------------------------------------

# Minimum Faster R-CNN score to accept as a confirmed ambulance.
DEFAULT_SCORE_THRESHOLD = float(os.getenv("FASTER_RCNN_SCORE_THRESHOLD", "0.50"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent
DEFAULT_WEIGHTS = ROOT / "ml_models" / "weights" / "faster_rcnn_ambulance_vehicle.pth"

# Torchvision inference transform — PIL RGB → [0, 1] float tensor.
_TO_TENSOR = T.ToTensor()

# ---------------------------------------------------------------------------
# Module-level lazy singleton — one instance shared by all Flask threads
# ---------------------------------------------------------------------------
_predictor_instance = None
_predictor_lock = threading.Lock()


def get_faster_rcnn(weights_path: str = None, score_threshold: float = None):
    """
    Return the shared FasterRCNNPredictor, loading it on first call.

    score_threshold overrides DEFAULT_SCORE_THRESHOLD when provided.
    Returns None (not an exception) when the weight file is absent so that
    callers fall back to YOLO-only detection instead of crashing.
    """
    global _predictor_instance
    if _predictor_instance is None:
        with _predictor_lock:
            if _predictor_instance is None:   # double-checked locking
                _predictor_instance = _load(weights_path, score_threshold)
    return _predictor_instance if _predictor_instance != "unavailable" else None


# ---------------------------------------------------------------------------
# Internal loader
# ---------------------------------------------------------------------------

def _load(weights_path: str = None, score_threshold: float = None):
    w = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if not w.exists():
        _log(
            f"[frcnn] Weight file not found: {w}\n"
            f"        Place faster_rcnn_ambulance_vehicle.pth in "
            f"ml_models/weights/ to enable Faster R-CNN verification.\n"
            f"        Until then, YOLO detections are trusted without secondary check."
        )
        return "unavailable"
    effective_threshold = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD
    try:
        p = FasterRCNNPredictor(str(w), score_threshold=effective_threshold)
        _log(
            f"[frcnn] Loaded  : {w.name}\n"
            f"        Device  : {p.device}\n"
            f"        Classes : {NUM_CLASSES}  {CLASS_MAP}\n"
            f"        Trigger : label=={AMBULANCE_LABEL} ({CLASS_MAP[AMBULANCE_LABEL]}) "
            f"with score >= {p.score_threshold}\n"
            f"        Suppressed label=={VEHICLE_LABEL} ({CLASS_MAP[VEHICLE_LABEL]}) "
            f"— never fires override"
        )
        return p
    except Exception as exc:
        _log(f"[frcnn] Load failed: {exc}")
        return "unavailable"


def _log(msg: str):
    """Logger that works both inside and outside a Flask app context."""
    try:
        from flask import current_app
        current_app.logger.info(msg)
    except RuntimeError:
        print(msg)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class FasterRCNNPredictor:
    """
    Wrapper around the fine-tuned Faster R-CNN used for ambulance verification.

    Parameters
    ----------
    weights : str
        Absolute path to the .pth checkpoint.
    score_threshold : float
        Minimum score for a class-2 detection to be accepted (default 0.50).
    device : str | None
        'cuda' / 'cpu', or None for auto-detect.
    """

    def __init__(
        self,
        weights: str,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        device: str = None,
    ):
        self.score_threshold = score_threshold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _build_and_load(str(weights), self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_numpy(self, bgr_crop: np.ndarray) -> dict:
        """
        Verify whether a BGR numpy array (OpenCV crop) contains an ambulance.

        Parameters
        ----------
        bgr_crop : np.ndarray  shape (H, W, 3)  dtype uint8

        Returns
        -------
        dict
            ambulance_confirmed : bool   True only when label==2 and score>=threshold
            confidence          : float  highest class-2 score (0 if none)
            inference_ms        : float
            model               : str    identifier
            class_map           : dict   {0: "background", 1: "vehicle", 2: "ambulance"}
            ambulance_label     : int    always 2
        """
        pil = Image.fromarray(bgr_crop[:, :, ::-1].astype(np.uint8))  # BGR → RGB
        return self._infer(pil)

    def verify_pil(self, pil_image: Image.Image) -> dict:
        """Verify a PIL RGB image."""
        return self._infer(pil_image.convert("RGB"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _infer(self, pil_image: Image.Image) -> dict:
        t0 = time.time()

        tensor = _TO_TENSOR(pil_image).to(self.device)   # [3, H, W]

        with torch.no_grad():
            outputs = self.model([tensor])               # list of 1 pred dict

        preds  = outputs[0]
        labels = preds.get("labels", torch.tensor([])).cpu().tolist()
        scores = preds.get("scores", torch.tensor([])).cpu().tolist()

        # ── Diagnostic: log every raw prediction ──────────────────
        if labels:
            for lbl, sc in zip(labels, scores):
                name = CLASS_MAP.get(lbl, f"unknown({lbl})")
                if lbl == AMBULANCE_LABEL and sc >= self.score_threshold:
                    tag = " <-- AMBULANCE CONFIRMED"
                elif lbl == AMBULANCE_LABEL:
                    tag = f" <-- ambulance class but score {sc:.4f} < threshold {self.score_threshold} [SCENARIO B]"
                elif lbl == VEHICLE_LABEL:
                    tag = " (vehicle class — suppressed) [SCENARIO A if only this appears]"
                else:
                    tag = ""
                _log(f"[frcnn] label={lbl} ({name}) score={sc:.4f}{tag}")
        else:
            _log("[frcnn] no detections returned from this crop")
        # ──────────────────────────────────────────────────────────

        best_ambulance_score = 0.0

        for lbl, sc in zip(labels, scores):
            if lbl == AMBULANCE_LABEL and sc >= self.score_threshold:
                # Class 2 — Ambulance: eligible to confirm emergency
                best_ambulance_score = max(best_ambulance_score, sc)
            # Class 1 — Vehicle: explicitly skipped, no action taken
            # Class 0 — Background: explicitly skipped, no action taken

        confirmed = best_ambulance_score > 0.0
        elapsed   = round((time.time() - t0) * 1000, 1)

        return {
            "ambulance_confirmed": confirmed,
            "confidence":          round(best_ambulance_score, 4),
            "inference_ms":        elapsed,
            "model":               "faster_rcnn_ambulance_vehicle",
            "class_map":           CLASS_MAP,
            "ambulance_label":     AMBULANCE_LABEL,
            "raw_labels":          labels,
            "raw_scores":          [round(s, 4) for s in scores],
        }


# ---------------------------------------------------------------------------
# Model builder with pre-load validation
# ---------------------------------------------------------------------------

def _build_and_load(weights_path: str, device: torch.device):
    """
    1. Build fasterrcnn_resnet50_fpn with NUM_CLASSES=3.
    2. Validate the checkpoint's head dimensions match before loading.
    3. Load with strict=True so any key mismatch raises immediately.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # ── Build architecture with correct class count ─────────────
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    # After replacement:
    #   cls_score.weight  shape: [3, in_features]   — rows 0/1/2 = bg/vehicle/ambulance
    #   bbox_pred.weight  shape: [12, in_features]  — 4 coords * 3 classes

    # ── Load checkpoint ─────────────────────────────────────────
    raw = torch.load(weights_path, map_location=device)

    # Unwrap common checkpoint wrappers
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    else:
        state = raw

    # ── Pre-load dimension check ─────────────────────────────────
    # Read the saved head weight shapes directly from the state dict so we
    # can emit a human-readable error before PyTorch raises a cryptic one.
    cls_key  = "roi_heads.box_predictor.cls_score.weight"
    bbox_key = "roi_heads.box_predictor.bbox_pred.weight"

    if cls_key in state:
        saved_n_cls = state[cls_key].shape[0]          # expected 3
        if saved_n_cls != NUM_CLASSES:
            raise ValueError(
                f"[frcnn] Class count mismatch.\n"
                f"  Checkpoint cls_score rows : {saved_n_cls}\n"
                f"  predictor.py NUM_CLASSES  : {NUM_CLASSES}\n"
                f"  Fix: set NUM_CLASSES = {saved_n_cls} in predictor.py "
                f"and update CLASS_MAP / AMBULANCE_LABEL accordingly."
            )

    if bbox_key in state:
        saved_bbox_rows = state[bbox_key].shape[0]      # expected 12  (4 * 3)
        expected_bbox   = 4 * NUM_CLASSES
        if saved_bbox_rows != expected_bbox:
            raise ValueError(
                f"[frcnn] bbox_pred shape mismatch.\n"
                f"  Checkpoint bbox_pred rows : {saved_bbox_rows}  "
                f"(implies {saved_bbox_rows // 4} classes)\n"
                f"  predictor.py NUM_CLASSES  : {NUM_CLASSES}  "
                f"(expects {expected_bbox} rows)\n"
                f"  Adjust NUM_CLASSES to match the checkpoint."
            )

    # ── Load and check for missing / unexpected keys ────────────
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"[frcnn] Missing keys in checkpoint:\n  "
            + "\n  ".join(incompatible.missing_keys)
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"[frcnn] Unexpected keys in checkpoint:\n  "
            + "\n  ".join(incompatible.unexpected_keys)
        )

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="Faster R-CNN ambulance verifier")
    ap.add_argument("--image",   required=True,  help="Path to JPEG/PNG image or crop")
    ap.add_argument("--weights", default=None,   help="Override default weights path")
    ap.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    ap.add_argument("--show-all-detections", action="store_true",
                    help="Print every detection (label + score), not just ambulances")
    args = ap.parse_args()

    print("\nClass mapping for this model:")
    for idx, name in CLASS_MAP.items():
        marker = " <-- AMBULANCE TRIGGER" if idx == AMBULANCE_LABEL else \
                 " <-- suppressed (never triggers override)" if idx == VEHICLE_LABEL else ""
        print(f"  {idx} = {name}{marker}")
    print(f"\nScore threshold : {args.threshold}")
    print(f"NUM_CLASSES     : {NUM_CLASSES}\n")

    predictor = FasterRCNNPredictor(
        weights=args.weights or str(DEFAULT_WEIGHTS),
        score_threshold=args.threshold,
    )

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    if args.show_all_detections:
        # Run raw inference and show every detection for debugging
        pil = Image.fromarray(img[:, :, ::-1].astype("uint8"))
        tensor = _TO_TENSOR(pil).to(predictor.device)
        with torch.no_grad():
            raw_preds = predictor.model([tensor])[0]
        labels = raw_preds["labels"].cpu().tolist()
        scores = raw_preds["scores"].cpu().tolist()
        print("All detections (label, class_name, score):")
        for lbl, sc in zip(labels, scores):
            name = CLASS_MAP.get(lbl, f"unknown({lbl})")
            print(f"  label={lbl}  {name:<12}  score={sc:.4f}")
        print()

    result = predictor.verify_numpy(img)

    print("=" * 52)
    print("  Faster R-CNN Ambulance Verification Result")
    print("=" * 52)
    print(f"  Ambulance confirmed : {'YES — OVERRIDE WILL FIRE' if result['ambulance_confirmed'] else 'NO  — override suppressed'}")
    print(f"  Confidence (cls=2)  : {result['confidence'] * 100:.2f}%")
    print(f"  Inference           : {result['inference_ms']} ms")
    print(f"  Trigger label       : {result['ambulance_label']} "
          f"({CLASS_MAP[result['ambulance_label']]})")
    print("=" * 52)
