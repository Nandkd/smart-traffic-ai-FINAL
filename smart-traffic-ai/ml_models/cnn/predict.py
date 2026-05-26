"""
ml_models/cnn/predict.py
=========================
Real-time inference module for the trained CNN ambulance classifier.
Supports single image, batch, base64, and numpy array inputs.

Usage (standalone):
    python predict.py --image path/to/car.jpg
    python predict.py --image path/to/ambulance.jpg --visualize
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml_models.cnn.architecture import AmbulanceCNN

ROOT = Path(__file__).parent.parent.parent
WEIGHTS_PATH = ROOT / "ml_models" / "weights" / "ambulance_cnn.pth"

# ── Pre-processing (must match validation transform in train.py) ──
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_NAMES = ["non_ambulance", "ambulance"]


class AmbulancePredictor:
    """
    Singleton-style predictor.  Load once, call many times.

    Example::

        predictor = AmbulancePredictor()
        result = predictor.predict_path("ambulance.jpg")
        print(result)
        # {'class': 'ambulance', 'confidence': 0.9821, 'ambulance_detected': True, 'inference_ms': 6.4}
    """

    def __init__(self, weights: str = None, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = AmbulanceCNN(num_classes=2)

        w = weights or str(WEIGHTS_PATH)
        if Path(w).exists():
            self.model.load_state_dict(
                torch.load(w, map_location=self.device)
            )
            print(f"✅ CNN weights loaded from {w}")
        else:
            print(f"⚠️  Weights not found at {w}. Using random weights (train first).")

        self.model.to(self.device)
        self.model.eval()

    # ── Public API ─────────────────────────────────────────────

    def predict_path(self, image_path: str) -> dict:
        """Run inference on an image file path."""
        img = Image.open(image_path).convert("RGB")
        return self._infer(img)

    def predict_pil(self, pil_image: Image.Image) -> dict:
        """Run inference on a PIL Image."""
        return self._infer(pil_image.convert("RGB"))

    def predict_numpy(self, bgr_array: np.ndarray) -> dict:
        """Run inference on a BGR numpy array (OpenCV format)."""
        rgb = bgr_array[:, :, ::-1]  # BGR → RGB
        pil = Image.fromarray(rgb.astype(np.uint8))
        return self._infer(pil)

    def predict_base64(self, b64_str: str) -> dict:
        """Run inference on a base64-encoded image string."""
        import base64, io
        header, _, data = b64_str.partition(",")
        raw = base64.b64decode(data if data else b64_str)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        return self._infer(pil)

    def predict_batch(self, images: list) -> list:
        """
        Batch inference on a list of PIL Images.
        Returns a list of result dicts.
        """
        t0 = time.time()
        tensors = torch.stack([INFERENCE_TRANSFORM(img) for img in images]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensors)
            probs = F.softmax(logits, dim=1)
        elapsed_ms = (time.time() - t0) * 1000 / len(images)

        results = []
        for prob in probs:
            cls_idx = int(prob.argmax())
            conf = float(prob[cls_idx])
            results.append({
                "class": CLASS_NAMES[cls_idx],
                "confidence": round(conf, 4),
                "ambulance_detected": cls_idx == 1,
                "probabilities": {
                    "non_ambulance": round(float(prob[0]), 4),
                    "ambulance": round(float(prob[1]), 4),
                },
                "inference_ms": round(elapsed_ms, 1),
            })
        return results

    # ── Internal ───────────────────────────────────────────────

    def _infer(self, pil_image: Image.Image) -> dict:
        t0 = time.time()
        tensor = INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        cls_idx = int(probs.argmax())
        conf = float(probs[cls_idx])
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        return {
            "class": CLASS_NAMES[cls_idx],
            "confidence": round(conf, 4),
            "ambulance_detected": cls_idx == 1,
            "action": "EMERGENCY_OVERRIDE" if cls_idx == 1 else "NORMAL",
            "probabilities": {
                "non_ambulance": round(float(probs[0]), 4),
                "ambulance": round(float(probs[1]), 4),
            },
            "inference_ms": elapsed_ms,
            "model": "ambulance_cnn_v1",
        }


# ── CLI ────────────────────────────────────────────────────────

def visualize_result(image_path: str, result: dict):
    """Draw result on image and display."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image for visualization.")
        return

    is_amb = result["ambulance_detected"]
    color = (0, 0, 255) if is_amb else (0, 200, 0)
    label = f"{'AMBULANCE' if is_amb else 'NON-AMBULANCE'} {result['confidence']*100:.1f}%"

    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, h), color, 4)
    cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2)
    cv2.putText(img, f"{result['inference_ms']}ms", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    win = "Ambulance CNN Result — Press any key to close"
    cv2.imshow(win, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image file")
    ap.add_argument("--visualize", action="store_true", help="Show annotated image")
    ap.add_argument("--weights", default=None, help="Custom weights path")
    args = ap.parse_args()

    predictor = AmbulancePredictor(weights=args.weights)
    result = predictor.predict_path(args.image)

    print("\n" + "=" * 45)
    print("🚑 AMBULANCE CNN PREDICTION")
    print("=" * 45)
    print(f"  Class         : {result['class'].upper()}")
    print(f"  Ambulance?    : {'✅ YES' if result['ambulance_detected'] else '❌ NO'}")
    print(f"  Confidence    : {result['confidence']*100:.2f}%")
    print(f"  Probabilities : amb={result['probabilities']['ambulance']:.4f} | "
          f"non={result['probabilities']['non_ambulance']:.4f}")
    print(f"  Inference     : {result['inference_ms']} ms")
    print(f"  Action        : {result['action']}")
    print("=" * 45)

    if args.visualize:
        visualize_result(args.image, result)
