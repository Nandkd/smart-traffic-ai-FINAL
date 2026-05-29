"""backend/routes/detection.py — YOLOv8 & CNN inference endpoints."""

import os
import base64
import uuid
import json
import time
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

    # Fallback: JSON base64 body
    data = request.get_json(silent=True) or {}
    if not data.get("image"):
        return jsonify({"error": "No image provided. Send a multipart file or base64 image."}), 400

    if not model:
        return jsonify({"error": "YOLO model unavailable. Ensure ultralytics is installed."}), 503

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


@detection_bp.route("/ambulance", methods=["POST"])
@jwt_required()
def detect_ambulance():
    """
    POST /api/detect/ambulance
    Multipart image → YOLO detection then OpenCV colour/shape analysis.
    Returns: ambulance_detected, confidence, action, inference_ms, model.
    """
    import cv2

    start = time.time()

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image file provided. Upload a JPEG/PNG image."}), 400

    raw = file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image. Use JPEG or PNG."}), 400

    ambulance_detected = False
    confidence = 0.0
    model_used = "opencv_cv_analysis"

    # ── Stage 1: YOLO detection ──────────────────────────────────
    yolo = get_yolo()
    if yolo:
        try:
            tmp = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.jpg")
            cv2.imwrite(tmp, img)
            results = yolo(tmp, conf=0.25, verbose=False)
            r = results[0]
            for box in r.boxes:
                cls_name = r.names[int(box.cls)].lower()
                if cls_name in ("ambulance", "emergency vehicle", "ambulance vehicle"):
                    ambulance_detected = True
                    confidence = max(confidence, float(box.conf))
            os.remove(tmp)
            model_used = "yolov8+opencv_cv"
        except Exception as e:
            current_app.logger.warning(f"YOLO ambulance check failed: {e}")

    # ── Stage 2: OpenCV colour / cross analysis ──────────────────
    # Always run as a second opinion; override YOLO only if stronger signal
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_px = img.shape[0] * img.shape[1] or 1

    # Red mask (two hue ranges for red wrap-around)
    red_lo1, red_hi1 = np.array([0, 100, 80]),   np.array([10, 255, 255])
    red_lo2, red_hi2 = np.array([165, 100, 80]),  np.array([180, 255, 255])
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, red_lo1, red_hi1),
        cv2.inRange(hsv, red_lo2, red_hi2),
    )

    # Blue lights
    blue_mask = cv2.inRange(hsv, np.array([100, 120, 100]), np.array([140, 255, 255]))

    # White body
    white_mask = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 40, 255]))

    red_r   = np.count_nonzero(red_mask)   / total_px
    blue_r  = np.count_nonzero(blue_mask)  / total_px
    white_r = np.count_nonzero(white_mask) / total_px

    # Horizontal / vertical edge cross pattern
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=40, minLineLength=25, maxLineGap=8)
    cross_bonus = 0.0
    if lines is not None:
        h_cnt = sum(1 for l in lines if abs(l[0][3] - l[0][1]) < 12)
        v_cnt = sum(1 for l in lines if abs(l[0][2] - l[0][0]) < 12)
        if h_cnt >= 2 and v_cnt >= 2:
            cross_bonus = 0.12

    cv_score = red_r * 4.5 + blue_r * 3.0 + white_r * 0.6 + cross_bonus
    cv_threshold = 0.05

    if cv_score >= cv_threshold:
        cv_conf = min(0.97, 0.45 + cv_score * 6.0)
        if cv_conf > confidence:
            ambulance_detected = True
            confidence = cv_conf
    elif not ambulance_detected:
        # Not detected by either stage
        confidence = min(0.40, cv_score * 6.0)

    elapsed = round((time.time() - start) * 1000, 1)
    return jsonify({
        "ambulance_detected": ambulance_detected,
        "confidence": round(confidence, 4),
        "action": (
            "EMERGENCY OVERRIDE — Crossroad cleared for ambulance passage"
            if ambulance_detected else
            "NORMAL — No ambulance detected"
        ),
        "inference_ms": elapsed,
        "model": model_used,
        "cv_score": round(cv_score, 4),
    }), 200


