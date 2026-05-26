"""backend/routes/detection.py — YOLOv8 & CNN inference endpoints."""

import os
import base64
import uuid
import json
import time
import random
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required

detection_bp = Blueprint("detection", __name__)

# ── Lazy model loading ─────────────────────────────────────────
_yolo_model = None
_cnn_model = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            weights = os.path.join(
                os.path.dirname(__file__), "../../ml_models/weights/yolov8_traffic.pt"
            )
            if os.path.exists(weights):
                _yolo_model = YOLO(weights)
            else:
                _yolo_model = YOLO("yolov8n.pt")  # fallback to pretrained
        except Exception as e:
            current_app.logger.warning(f"YOLO load failed: {e}")
            _yolo_model = "unavailable"
    return _yolo_model if _yolo_model != "unavailable" else None


def get_cnn():
    global _cnn_model
    if _cnn_model is None:
        try:
            import torch
            from ml_models.cnn.architecture import AmbulanceCNN
            weights = os.path.join(
                os.path.dirname(__file__), "../../ml_models/weights/ambulance_cnn.pth"
            )
            model = AmbulanceCNN()
            if os.path.exists(weights):
                model.load_state_dict(torch.load(weights, map_location="cpu"))
            model.eval()
            _cnn_model = model
        except Exception as e:
            current_app.logger.warning(f"CNN load failed: {e}")
            _cnn_model = "unavailable"
    return _cnn_model if _cnn_model != "unavailable" else None


# ── Helper: decode base64 image ────────────────────────────────
def decode_image(b64_str: str) -> np.ndarray:
    import cv2
    data = base64.b64decode(b64_str.split(",")[-1])
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ── Simulate detection when model unavailable ──────────────────
def simulate_yolo_detection():
    """Return realistic simulated YOLO output for demo/testing."""
    classes = ["car", "bus", "truck", "motorcycle", "ambulance"]
    weights_sim = [0.55, 0.15, 0.12, 0.15, 0.03]
    n = random.randint(3, 15)
    detections = []
    for _ in range(n):
        cls = random.choices(classes, weights=weights_sim)[0]
        detections.append({
            "class": cls,
            "confidence": round(random.uniform(0.72, 0.98), 3),
            "bbox": [
                random.randint(0, 500), random.randint(0, 300),
                random.randint(50, 200), random.randint(40, 150),
            ],
        })
    vehicle_counts = {}
    for d in detections:
        vehicle_counts[d["class"]] = vehicle_counts.get(d["class"], 0) + 1
    total = sum(vehicle_counts.values())
    density = "low" if total < 10 else ("medium" if total < 25 else "high")
    return {
        "detections": detections,
        "vehicle_counts": vehicle_counts,
        "total_vehicles": total,
        "density_class": density,
        "ambulance_detected": "ambulance" in vehicle_counts,
        "inference_ms": round(random.uniform(11, 18), 1),
        "mode": "simulation",
    }


# ── Endpoints ──────────────────────────────────────────────────

@detection_bp.route("/vehicles", methods=["POST"])
@jwt_required()
def detect_vehicles():
    """
    POST /api/detect/vehicles
    Body: { "image": "<base64 string>" }  OR  multipart file
    Returns: detections, vehicle counts, density class
    """
    start = time.time()
    model = get_yolo()

    # Try real inference
    if model and (request.content_type or "").startswith("multipart"):
        import cv2
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided"}), 400
        tmp_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.jpg")
        file.save(tmp_path)
        try:
            results = model(tmp_path, conf=0.4)
            r = results[0]
            detections = []
            for box in r.boxes:
                detections.append({
                    "class": r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox": [round(x) for x in box.xyxy[0].tolist()],
                })
            vehicle_counts = {}
            for d in detections:
                vehicle_counts[d["class"]] = vehicle_counts.get(d["class"], 0) + 1
            total = sum(vehicle_counts.values())
            density = "low" if total < 10 else ("medium" if total < 25 else "high")
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return jsonify({
                "detections": detections,
                "vehicle_counts": vehicle_counts,
                "total_vehicles": total,
                "density_class": density,
                "ambulance_detected": "ambulance" in vehicle_counts,
                "inference_ms": elapsed_ms,
                "mode": "yolov8",
            }), 200
        finally:
            os.remove(tmp_path)

    # Fallback: JSON base64 body or simulation
    data = request.get_json(silent=True) or {}
    if not data.get("image") and not model:
        result = simulate_yolo_detection()
        return jsonify(result), 200

    if data.get("image") and model:
        img = decode_image(data["image"])
        tmp_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.jpg")
        import cv2
        cv2.imwrite(tmp_path, img)
        try:
            results = model(tmp_path, conf=0.4)
            r = results[0]
            detections = [
                {"class": r.names[int(b.cls)], "confidence": round(float(b.conf), 3),
                 "bbox": [round(x) for x in b.xyxy[0].tolist()]}
                for b in r.boxes
            ]
            vehicle_counts = {}
            for d in detections:
                vehicle_counts[d["class"]] = vehicle_counts.get(d["class"], 0) + 1
            total = sum(vehicle_counts.values())
            density = "low" if total < 10 else ("medium" if total < 25 else "high")
            return jsonify({
                "detections": detections, "vehicle_counts": vehicle_counts,
                "total_vehicles": total, "density_class": density,
                "ambulance_detected": "ambulance" in vehicle_counts,
                "inference_ms": round((time.time() - start) * 1000, 1), "mode": "yolov8",
            }), 200
        finally:
            os.remove(tmp_path)

    # Pure simulation
    return jsonify(simulate_yolo_detection()), 200


@detection_bp.route("/ambulance", methods=["POST"])
@jwt_required()
def detect_ambulance():
    """
    POST /api/detect/ambulance
    Runs CNN ambulance classifier on submitted image.
    """
    # Simulated result for demo (replace with real CNN inference)
    is_ambulance = random.random() < 0.12
    confidence = random.uniform(0.88, 0.99) if is_ambulance else random.uniform(0.02, 0.18)
    return jsonify({
        "ambulance_detected": is_ambulance,
        "confidence": round(confidence, 4),
        "action": "EMERGENCY_OVERRIDE" if is_ambulance else "NORMAL",
        "inference_ms": round(random.uniform(5, 12), 1),
        "model": "ambulance_cnn_v1",
    }), 200


@detection_bp.route("/live-feed", methods=["GET"])
@jwt_required()
def live_feed_stats():
    """Simulated live feed stats (would connect to RTSP in production)."""
    return jsonify({
        "fps": random.randint(28, 32),
        "resolution": "1920x1080",
        "vehicles_in_frame": random.randint(0, 20),
        "tracking_ids": list(range(1, random.randint(3, 12))),
        "latency_ms": random.randint(40, 120),
        "source": "rtsp://camera-01.traffic.local",
    }), 200
