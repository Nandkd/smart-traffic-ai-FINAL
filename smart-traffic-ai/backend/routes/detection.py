"""backend/routes/detection.py — YOLOv8 inference endpoints using best.pt."""

import os
import base64
import uuid
import time
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required

detection_bp = Blueprint("detection", __name__)

# Trained classes in best.pt (from data.yaml)
TRAINED_CLASSES = {"car", "motorcycle", "bus", "truck", "ambulance"}

# ── Lazy model loading — YOLO ──────────────────────────────────
_yolo_model = None

_WEIGHTS_REL = os.path.join(
    os.path.dirname(__file__), "../../ml_models/weights/best.pt"
)


# ── Lazy model loading — Faster R-CNN ─────────────────────────
_frcnn_model = None


def get_faster_rcnn():
    """Return the shared FasterRCNNPredictor, or None if unavailable."""
    global _frcnn_model
    if _frcnn_model is None:
        try:
            weights = current_app.config.get("FASTER_RCNN_WEIGHTS")
        except RuntimeError:
            weights = None
        from ml_models.faster_rcnn.predictor import get_faster_rcnn as _load
        _frcnn_model = _load(weights) or "unavailable"
    return _frcnn_model if _frcnn_model != "unavailable" else None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # App config takes precedence (set in settings.py); fall back to relative path
            try:
                weights = current_app.config.get("YOLO_WEIGHTS", _WEIGHTS_REL)
            except RuntimeError:
                weights = _WEIGHTS_REL
            if os.path.exists(weights):
                _yolo_model = YOLO(weights)
                try:
                    current_app.logger.info(f"[detection] YOLO loaded: {weights}")
                except RuntimeError:
                    print(f"[detection] YOLO loaded: {weights}")
            else:
                msg = f"[detection] best.pt not found at {weights} — place weights file there and restart"
                try:
                    current_app.logger.warning(msg)
                except RuntimeError:
                    print(msg)
                _yolo_model = "unavailable"
        except Exception as e:
            try:
                current_app.logger.warning(f"[detection] YOLO load failed: {e}")
            except RuntimeError:
                print(f"[detection] YOLO load failed: {e}")
            _yolo_model = "unavailable"
    return _yolo_model if _yolo_model != "unavailable" else None


# ── Helper: decode base64 image ────────────────────────────────
def decode_image(b64_str: str) -> np.ndarray:
    import cv2
    data = base64.b64decode(b64_str.split(",")[-1])
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _run_inference(model, img_path: str) -> tuple[list, dict]:
    """
    Run best.pt inference on img_path.
    Returns (detections, vehicle_counts).
    Ambulance uses a lower confidence threshold since it is safety-critical.
    """
    results = model(img_path, conf=0.25, verbose=False)
    r = results[0]
    detections = []
    vehicle_counts = {}
    for box in r.boxes:
        cls_name = r.names[int(box.cls)].lower()
        conf_val = float(box.conf)
        # Only keep classes the model was actually trained on
        if cls_name not in TRAINED_CLASSES:
            continue
        # Per-class confidence gate: ambulance is safety-critical → keep lower threshold
        min_conf = 0.30 if cls_name == "ambulance" else 0.40
        if conf_val < min_conf:
            continue
        vehicle_counts[cls_name] = vehicle_counts.get(cls_name, 0) + 1
        detections.append({
            "class":      cls_name,
            "confidence": round(conf_val, 3),
            "bbox":       [round(x) for x in box.xyxy[0].tolist()],
        })
    return detections, vehicle_counts


# ── Endpoints ──────────────────────────────────────────────────

@detection_bp.route("/vehicles", methods=["POST"])
@jwt_required()
def detect_vehicles():
    """
    POST /api/detect/vehicles
    Body: multipart image file  OR  { "image": "<base64 string>" }
    Returns: detections, vehicle_counts, density_class, ambulance_detected
    """
    start = time.time()
    model = get_yolo()

    if model is None:
        return jsonify({"error": "YOLO model unavailable — place best.pt in ml_models/weights/ and restart"}), 503

    upload_folder = current_app.config["UPLOAD_FOLDER"]

    # ── Multipart file ─────────────────────────────────────────
    if (request.content_type or "").startswith("multipart"):
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided"}), 400
        tmp_path = os.path.join(upload_folder, f"{uuid.uuid4()}.jpg")
        file.save(tmp_path)
        try:
            detections, vehicle_counts = _run_inference(model, tmp_path)
        finally:
            os.remove(tmp_path)
    else:
        # ── Base64 JSON body ───────────────────────────────────
        import cv2
        data = request.get_json(silent=True) or {}
        if not data.get("image"):
            return jsonify({"error": "No image provided. Send a multipart file or base64 image."}), 400
        img = decode_image(data["image"])
        tmp_path = os.path.join(upload_folder, f"{uuid.uuid4()}.jpg")
        cv2.imwrite(tmp_path, img)
        try:
            detections, vehicle_counts = _run_inference(model, tmp_path)
        finally:
            os.remove(tmp_path)

    total   = sum(vehicle_counts.values())
    density = "high" if total > 50 else ("medium" if total > 20 else "low")

    return jsonify({
        "detections":         detections,
        "vehicle_counts":     vehicle_counts,
        "total_vehicles":     total,
        "density_class":      density,
        "ambulance_detected": "ambulance" in vehicle_counts,
        "inference_ms":       round((time.time() - start) * 1000, 1),
        "model":              "best.pt (custom Indian traffic)",
    }), 200


@detection_bp.route("/ambulance", methods=["POST"])
@jwt_required()
def detect_ambulance():
    """
    POST /api/detect/ambulance
    Multipart image → YOLO best.pt ambulance class detection,
    with OpenCV colour/cross analysis as a secondary confirmation.
    Returns: ambulance_detected, confidence, action, inference_ms, model
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
    confidence  = 0.0
    model_used  = "opencv_cv_analysis"
    frcnn_used  = False   # set True once Faster R-CNN runs
    yolo_bbox   = None    # ambulance bbox from YOLO, used to crop for FRCNN

    # ── Stage 1: YOLO best.pt ambulance class ───────────────────
    yolo = get_yolo()
    if yolo:
        try:
            tmp = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.jpg")
            cv2.imwrite(tmp, img)
            results = yolo(tmp, conf=0.25, verbose=False)
            r = results[0]
            best_yolo_conf = 0.0
            for box in r.boxes:
                if r.names[int(box.cls)].lower() == "ambulance":
                    c = float(box.conf)
                    if c > best_yolo_conf:
                        best_yolo_conf = c
                        yolo_bbox = [round(x) for x in box.xyxy[0].tolist()]
            os.remove(tmp)
            if yolo_bbox:
                ambulance_detected = True
                confidence = best_yolo_conf
                model_used = "best.pt"
        except Exception as e:
            current_app.logger.warning(f"YOLO ambulance check failed: {e}")

    # ── Stage 2: Faster R-CNN verification (runs only when YOLO fires) ──
    #
    # When YOLO detected an ambulance and produced a bounding box, crop
    # that region and pass it to Faster R-CNN.  The result is authoritative:
    #   • FRCNN confirms  → keep ambulance_detected = True
    #   • FRCNN rejects   → override to False (suppress false positive)
    #
    # When YOLO found nothing, skip FRCNN and fall through to the OpenCV
    # colour/cross fallback (Stage 3) so legacy behaviour is preserved.
    if yolo_bbox is not None:
        frcnn = get_faster_rcnn()
        if frcnn is not None:
            try:
                x1, y1, x2, y2 = yolo_bbox
                # Guard against degenerate boxes
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    frcnn_result = frcnn.verify_numpy(crop)
                    frcnn_used = True
                    if frcnn_result["ambulance_confirmed"]:
                        # FRCNN agrees — use its confidence if it's higher
                        confidence = max(confidence, frcnn_result["confidence"])
                        model_used = "best.pt+faster_rcnn"
                    else:
                        # FRCNN rejects the YOLO detection — suppress override
                        ambulance_detected = False
                        confidence = frcnn_result["confidence"]
                        model_used = "best.pt+faster_rcnn(rejected)"
            except Exception as e:
                current_app.logger.warning(f"Faster R-CNN verification failed: {e}")
                # On FRCNN error, trust YOLO — do not suppress
        # FRCNN unavailable (weights not placed yet) — trust YOLO as-is

    # ── Stage 3: OpenCV colour / cross fallback ──────────────────
    #
    # Only runs when YOLO detected nothing (yolo_bbox is None).
    # Preserves legacy behaviour: colour heuristic can catch ambulances
    # that YOLO missed before the FRCNN weights are available.
    hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_px  = img.shape[0] * img.shape[1] or 1

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   100, 80]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([165, 100, 80]), np.array([180, 255, 255])),
    )
    blue_mask  = cv2.inRange(hsv, np.array([100, 120, 100]), np.array([140, 255, 255]))
    white_mask = cv2.inRange(hsv, np.array([0, 0, 190]),     np.array([180, 40,  255]))

    red_r   = np.count_nonzero(red_mask)   / total_px
    blue_r  = np.count_nonzero(blue_mask)  / total_px
    white_r = np.count_nonzero(white_mask) / total_px

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

    if yolo_bbox is None:
        # YOLO found nothing — apply OpenCV fallback
        if cv_score >= 0.05:
            cv_conf = min(0.97, 0.45 + cv_score * 6.0)
            if cv_conf > confidence:
                ambulance_detected = True
                confidence = cv_conf
                model_used = "opencv_cv_analysis"
        else:
            confidence = min(0.40, cv_score * 6.0)

    elapsed = round((time.time() - start) * 1000, 1)
    return jsonify({
        "ambulance_detected": ambulance_detected,
        "confidence":  round(confidence, 4),
        "action": (
            "EMERGENCY OVERRIDE — Crossroad cleared for ambulance passage"
            if ambulance_detected else
            "NORMAL — No ambulance detected"
        ),
        "inference_ms": elapsed,
        "model":        model_used,
        "cv_score":     round(cv_score, 4),
        "frcnn_used":   frcnn_used,
    }), 200
