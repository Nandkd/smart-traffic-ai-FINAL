"""
backend/routes/crossroad.py
============================
Single Indian crossroad controller.

One intersection — 4 roads (North, South, East, West).
Upload one video per road.
Real YOLO detects Indian traffic on each road.
Signal controller gives green to most congested road.
Ambulance detected → immediate priority clearance.

Indian vehicle classes:
  car, motorcycle, auto_rickshaw, bus, truck,
  bicycle, pedestrian, ambulance
"""

import os
import time
import uuid
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required

crossroad_bp = Blueprint("crossroad", __name__)

# ── Indian traffic vehicle classes ─────────────────────────────
INDIAN_VEHICLES = [
    "car", "motorcycle", "auto_rickshaw",
    "bus", "truck", "bicycle", "pedestrian", "ambulance"
]

# Map COCO class names → Indian traffic classes
COCO_TO_INDIAN = {
    "car":          "car",
    "motorcycle":   "motorcycle",
    "bicycle":      "bicycle",
    "bus":          "bus",
    "truck":        "truck",
    "person":       "pedestrian",
    "rickshaw":     "auto_rickshaw",
    "auto":         "auto_rickshaw",
    "ambulance":    "ambulance",
    "van":          "car",
    "motorbike":    "motorcycle",
    "three-wheeler":"auto_rickshaw",
}

VEHICLE_WEIGHT = {
    "car":          1.0,
    "motorcycle":   0.5,
    "auto_rickshaw":0.8,
    "bus":          2.5,
    "truck":        2.0,
    "bicycle":      0.3,
    "pedestrian":   0.2,
    "ambulance":    10.0,   # highest priority weight
}

ROADS = ["north", "south", "east", "west"]
ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}

YOLO_MODELS = {
    "yolov8n":  {"name": "YOLOv8n",  "speed": "fastest", "map": "37.3", "rec": False},
    "yolov8s":  {"name": "YOLOv8s",  "speed": "fast",    "map": "44.9", "rec": True},
    "yolov8m":  {"name": "YOLOv8m",  "speed": "medium",  "map": "50.2", "rec": False},
    "yolov8l":  {"name": "YOLOv8l",  "speed": "slow",    "map": "52.9", "rec": False},
    "yolov9c":  {"name": "YOLOv9c",  "speed": "medium",  "map": "53.0", "rec": False},
    "yolov10s": {"name": "YOLOv10s", "speed": "fast",    "map": "46.3", "rec": False},
    "yolov11n": {"name": "YOLOv11n", "speed": "fastest", "map": "39.5", "rec": False},
    "yolov11s": {"name": "YOLOv11s", "speed": "fast",    "map": "47.0", "rec": False},
    "yolov11m": {"name": "YOLOv11m", "speed": "medium",  "map": "51.5", "rec": False},
}

# ── Signal timing formula for Indian traffic ───────────────────
def green_duration(vehicle_count: int, congestion_score: float) -> int:
    """
    Indian traffic needs longer greens due to chaotic merging.
    Based on PCU (Passenger Car Unit) equivalent counts.
    """
    if vehicle_count == 0: return 0
    if vehicle_count <= 10:  return 15
    if vehicle_count <= 25:  return 25
    if vehicle_count <= 50:  return 40
    if vehicle_count <= 80:  return 55
    if vehicle_count <= 120: return 70
    return 90   # max green for very heavy Indian traffic


def compute_pcu(vehicle_counts: dict) -> float:
    """
    Passenger Car Unit — Indian Roads Congress standard.
    Gives weighted count for different vehicle types.
    """
    pcu = 0.0
    for vtype, count in vehicle_counts.items():
        pcu += count * VEHICLE_WEIGHT.get(vtype, 1.0)
    return round(pcu, 2)


def decide_active_road(road_states: dict) -> str:
    """
    Choose which road gets green signal.
    Priority: ambulance > highest PCU > longest waiting.
    """
    # Check ambulance on any road
    for road, state in road_states.items():
        if state.get("ambulance_detected") and state.get("status") == "completed":
            return road

    # Pick road with highest PCU from completed roads
    scores = {}
    for road, state in road_states.items():
        if state.get("status") in ("completed", "processing") and state.get("total_vehicles", 0) > 0:
            scores[road] = state.get("pcu_count", 0)

    if scores:
        return max(scores, key=scores.get)
    return None


# ── Per-road state ─────────────────────────────────────────────
def blank_road(road: str) -> dict:
    return {
        "road":             road,
        "direction":        road.capitalize(),
        "status":           "idle",        # idle|queued|processing|completed|failed
        "progress":         0,
        "filename":         None,
        "model_used":       "yolov8s",
        "real_inference":   False,
        "video_info":       {},
        # Detection results
        "vehicle_counts":   {},
        "total_vehicles":   0,
        "pcu_count":        0.0,
        "density_class":    "unknown",
        "congestion_score": 0.0,
        "detections":       [],
        "frame_results":    [],
        "frames_done":      0,
        "current_frame":    None,
        # Ambulance
        "ambulance_detected":   False,
        "ambulance_timestamps": [],
        # Signal (set by controller)
        "signal":   "red",    # red | green | yellow
        "green_duration": 0,
        # Metrics
        "avg_vehicles":  0,
        "peak_vehicles": 0,
        "peak_frame":    None,
        "updated_at":    datetime.utcnow().isoformat(),
    }


# ── Global crossroad state ─────────────────────────────────────
_crossroad = {
    "name":           "Indian Crossroad",
    "location":       "Enter your location",
    "active_road":    None,         # which road has green right now
    "signal_mode":    "idle",       # idle | auto | emergency | manual
    "emergency_road": None,
    "cycle_count":    0,
    "roads":          {r: blank_road(r) for r in ROADS},
    "updated_at":     datetime.utcnow().isoformat(),
}
_lock = threading.Lock()


def update_signals():
    """
    Recompute which road gets green based on current detection data.
    Called after every frame detection update.
    """
    global _crossroad
    road_states = _crossroad["roads"]

    # Emergency override
    for road, state in road_states.items():
        if state["ambulance_detected"] and state["status"] in ("processing", "completed"):
            _crossroad["active_road"]    = road
            _crossroad["signal_mode"]    = "emergency"
            _crossroad["emergency_road"] = road
            for r in ROADS:
                _crossroad["roads"][r]["signal"] = "green" if r == road else "red"
                _crossroad["roads"][r]["green_duration"] = 90 if r == road else 0
            return

    # Normal: highest PCU gets green
    best = decide_active_road(road_states)
    if best:
        _crossroad["active_road"] = best
        if _crossroad["signal_mode"] != "manual":
            _crossroad["signal_mode"] = "auto"
        for r in ROADS:
            if r == best:
                vc = road_states[r].get("total_vehicles", 0)
                cs = road_states[r].get("congestion_score", 0.5)
                dur = green_duration(vc, cs)
                _crossroad["roads"][r]["signal"]         = "green"
                _crossroad["roads"][r]["green_duration"] = dur
            else:
                _crossroad["roads"][r]["signal"]         = "red"
                _crossroad["roads"][r]["green_duration"] = 0
    _crossroad["updated_at"] = datetime.utcnow().isoformat()


# ── YOLO / OpenCV processing ───────────────────────────────────
ULTRALYTICS_MAP = {
    "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt", "yolov8l": "yolov8l.pt",
    "yolov11n": "yolo11n.pt","yolov11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
}


def process_road_video(road: str, video_path: str, model_key: str, job_id: str):
    """Background thread — processes one road's video."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with _lock:
            _crossroad["roads"][road]["status"] = "failed"
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with _lock:
        _crossroad["roads"][road]["video_info"] = {
            "fps":          round(fps, 1),
            "total_frames": total_frames,
            "duration_sec": round(total_frames / max(fps, 1), 1),
            "resolution":   f"{width}x{height}",
        }
        _crossroad["roads"][road]["status"] = "processing"

    # Load YOLO if available
    yolo_model = None
    try:
        from ultralytics import YOLO
        mfile = ULTRALYTICS_MAP.get(model_key)
        if mfile:
            yolo_model = YOLO(mfile)
            with _lock:
                _crossroad["roads"][road]["real_inference"] = True
    except Exception as e:
        print(f"[{road}] YOLO load failed: {e} — using OpenCV")

    frame_results     = []
    frame_num         = 0
    sample_every      = max(1, int(fps))
    max_frames        = 100
    ambulance_times   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_every == 0 and len(frame_results) < max_frames:
            t0 = time.time()

            if yolo_model:
                # ── Real YOLO ──────────────────────────────────
                try:
                    results = yolo_model(frame, verbose=False, conf=0.30)
                    r = results[0]
                    vehicle_counts = {}
                    detections     = []
                    for box in r.boxes:
                        raw = r.names[int(box.cls)].lower()
                        mapped = COCO_TO_INDIAN.get(raw, raw)
                        if mapped in INDIAN_VEHICLES:
                            vehicle_counts[mapped] = vehicle_counts.get(mapped, 0) + 1
                            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                            detections.append({
                                "class":      mapped,
                                "confidence": round(float(box.conf), 3),
                                "bbox":       [x1, y1, x2, y2],
                            })
                except Exception:
                    vehicle_counts = {}
                    detections     = []

            else:
                # ── OpenCV blob analysis ───────────────────────
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur  = cv2.GaussianBlur(gray, (15, 15), 0)
                _, th = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
                kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                th    = cv2.dilate(th, kern, iterations=2)
                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                vehicle_counts = {}
                detections     = []
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < 500:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    aspect = w / max(h, 1)
                    # Classify by size — Indian road typical sizes
                    if area > 25000:
                        cls = "bus"
                    elif area > 15000:
                        cls = "truck"
                    elif area > 5000:
                        cls = "car"
                    elif area > 2000:
                        cls = "auto_rickshaw"
                    elif aspect > 2.5:
                        cls = "motorcycle"
                    elif area < 1000:
                        cls = "bicycle"
                    else:
                        cls = "motorcycle"
                    vehicle_counts[cls] = vehicle_counts.get(cls, 0) + 1
                    conf = round(min(0.55 + area / 40000, 0.92), 3)
                    detections.append({
                        "class":      cls,
                        "confidence": conf,
                        "bbox":       [x, y, x + w, y + h],
                    })

            elapsed_ms   = round((time.time() - t0) * 1000, 1)
            total        = sum(vehicle_counts.values())
            ts_sec       = round(frame_num / max(fps, 1), 2)
            pcu          = compute_pcu(vehicle_counts)
            ambulance    = "ambulance" in vehicle_counts

            if ambulance:
                ambulance_times.append(ts_sec)

            # Density thresholds for Indian traffic (higher than global)
            density = "high" if total > 50 else ("medium" if total > 20 else "low")
            cong_sc = round(min(total / 150, 1.0), 3)

            frame_data = {
                "frame_number":      frame_num,
                "timestamp_sec":     ts_sec,
                "vehicle_counts":    vehicle_counts,
                "detections":        detections,
                "total_vehicles":    total,
                "pcu_count":         pcu,
                "density_class":     density,
                "congestion_score":  cong_sc,
                "ambulance_detected": ambulance,
                "inference_ms":      elapsed_ms,
            }
            frame_results.append(frame_data)

            progress = min(95, int((len(frame_results) / max_frames) * 90) + 5)

            with _lock:
                if _crossroad["roads"][road].get("job_id") != job_id:
                    break   # cancelled
                _crossroad["roads"][road].update({
                    "current_frame":     frame_data,
                    "vehicle_counts":    vehicle_counts,
                    "total_vehicles":    total,
                    "pcu_count":         pcu,
                    "density_class":     density,
                    "congestion_score":  cong_sc,
                    "detections":        detections[:20],
                    "ambulance_detected": ambulance,
                    "ambulance_timestamps": ambulance_times[:],
                    "frames_done":       len(frame_results),
                    "progress":          progress,
                    "updated_at":        datetime.utcnow().isoformat(),
                })
                update_signals()

        frame_num += 1

    cap.release()

    # Final summary
    if frame_results:
        all_v    = [f["total_vehicles"] for f in frame_results]
        avg_v    = round(sum(all_v) / len(all_v), 1)
        peak_v   = max(all_v)
        peak_frm = max(frame_results, key=lambda x: x["total_vehicles"])

        # Aggregate vehicle type totals
        totals = {}
        for f in frame_results:
            for cls, cnt in f["vehicle_counts"].items():
                totals[cls] = totals.get(cls, 0) + cnt

        final_pcu = compute_pcu({k: v / len(frame_results) for k, v in totals.items()})

        with _lock:
            _crossroad["roads"][road].update({
                "status":            "completed",
                "progress":          100,
                "frame_results":     frame_results,
                "avg_vehicles":      avg_v,
                "peak_vehicles":     peak_v,
                "peak_frame":        peak_frm,
                "vehicle_counts":    {k: round(v / len(frame_results), 1) for k, v in totals.items()},
                "total_vehicles":    round(avg_v),
                "pcu_count":         final_pcu,
                "ambulance_timestamps": ambulance_times,
                "updated_at":        datetime.utcnow().isoformat(),
            })
            _crossroad["cycle_count"] += 1
            update_signals()

    else:
        with _lock:
            _crossroad["roads"][road]["status"] = "failed"

    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass


# ── API Endpoints ──────────────────────────────────────────────

@crossroad_bp.route("/state", methods=["GET"])
@jwt_required()
def get_state():
    """Full crossroad state."""
    with _lock:
        state = {
            "name":          _crossroad["name"],
            "location":      _crossroad["location"],
            "active_road":   _crossroad["active_road"],
            "signal_mode":   _crossroad["signal_mode"],
            "emergency_road":_crossroad["emergency_road"],
            "cycle_count":   _crossroad["cycle_count"],
            "updated_at":    _crossroad["updated_at"],
            "roads": {
                r: {k: v for k, v in s.items() if k != "frame_results"}
                for r, s in _crossroad["roads"].items()
            },
        }
    return jsonify(state), 200


@crossroad_bp.route("/models", methods=["GET"])
@jwt_required()
def list_models():
    return jsonify({"models": YOLO_MODELS}), 200


@crossroad_bp.route("/upload/<road>", methods=["POST"])
@jwt_required()
def upload_road_video(road):
    """Upload video for one road (north/south/east/west)."""
    if road not in ROADS:
        return jsonify({"error": f"Invalid road. Use: {ROADS}"}), 400

    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file      = request.files["video"]
    model_key = request.form.get("model", "yolov8s")
    location  = request.form.get("location", "")

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXTS)}"}), 400

    if model_key not in YOLO_MODELS:
        model_key = "yolov8s"

    upload_folder = current_app.config.get("UPLOAD_FOLDER", "/tmp")
    filename      = f"road_{road}_{uuid.uuid4().hex}{ext}"
    filepath      = os.path.join(upload_folder, filename)
    file.save(filepath)

    job_id = uuid.uuid4().hex[:10]

    with _lock:
        _crossroad["roads"][road] = blank_road(road)
        _crossroad["roads"][road].update({
            "job_id":   job_id,
            "status":   "queued",
            "filename": file.filename,
            "model_used": model_key,
        })
        if location:
            _crossroad["location"] = location

    thread = threading.Thread(
        target=process_road_video,
        args=(road, filepath, model_key, job_id),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "job_id":   job_id,
        "road":     road,
        "message":  f"{road.upper()} road video processing started",
        "model":    YOLO_MODELS[model_key]["name"],
    }), 202


@crossroad_bp.route("/frames/<road>", methods=["GET"])
@jwt_required()
def get_frames(road):
    with _lock:
        frames = _crossroad["roads"].get(road, {}).get("frame_results", [])
    return jsonify({"frames": frames, "count": len(frames)}), 200


@crossroad_bp.route("/ambulance/<road>", methods=["POST"])
@jwt_required()
def trigger_ambulance(road):
    if road not in ROADS:
        return jsonify({"error": "Invalid road"}), 400
    with _lock:
        _crossroad["roads"][road]["ambulance_detected"] = True
        _crossroad["signal_mode"]    = "emergency"
        _crossroad["emergency_road"] = road
        for r in ROADS:
            _crossroad["roads"][r]["signal"]         = "green" if r == road else "red"
            _crossroad["roads"][r]["green_duration"] = 90 if r == road else 0
        _crossroad["active_road"] = road
        _crossroad["updated_at"]  = datetime.utcnow().isoformat()
    return jsonify({
        "message": f"Emergency: {road.upper()} road cleared — 90s green",
        "road": road,
    }), 200


@crossroad_bp.route("/ambulance/<road>/clear", methods=["POST"])
@jwt_required()
def clear_ambulance(road):
    with _lock:
        _crossroad["roads"][road]["ambulance_detected"] = False
        _crossroad["emergency_road"] = None
        update_signals()
    return jsonify({"message": "Emergency cleared"}), 200


@crossroad_bp.route("/signal/<road>/override", methods=["POST"])
@jwt_required()
def override_signal(road):
    if road not in ROADS:
        return jsonify({"error": "Invalid road"}), 400
    data     = request.get_json(silent=True) or {}
    duration = int(data.get("duration", 30))
    with _lock:
        _crossroad["signal_mode"] = "manual"
        _crossroad["active_road"] = road
        for r in ROADS:
            _crossroad["roads"][r]["signal"]         = "green" if r == road else "red"
            _crossroad["roads"][r]["green_duration"] = duration if r == road else 0
        _crossroad["updated_at"] = datetime.utcnow().isoformat()
    return jsonify({"message": f"{road.upper()} green for {duration}s"}), 200


@crossroad_bp.route("/settings", methods=["POST"])
@jwt_required()
def update_settings():
    data = request.get_json(silent=True) or {}
    with _lock:
        if "name"     in data: _crossroad["name"]     = data["name"]
        if "location" in data: _crossroad["location"] = data["location"]
    return jsonify({"message": "Settings updated"}), 200


@crossroad_bp.route("/reset/<road>", methods=["POST"])
@jwt_required()
def reset_road(road):
    if road not in ROADS:
        return jsonify({"error": "Invalid road"}), 400
    with _lock:
        _crossroad["roads"][road] = blank_road(road)
        update_signals()
    return jsonify({"message": f"{road.upper()} road reset"}), 200


@crossroad_bp.route("/reset/all", methods=["POST"])
@jwt_required()
def reset_all():
    with _lock:
        for r in ROADS:
            _crossroad["roads"][r] = blank_road(r)
        _crossroad["active_road"]    = None
        _crossroad["signal_mode"]    = "idle"
        _crossroad["emergency_road"] = None
        _crossroad["updated_at"]     = datetime.utcnow().isoformat()
    return jsonify({"message": "All roads reset"}), 200
