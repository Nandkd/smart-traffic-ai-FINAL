"""
backend/routes/crossroad.py
============================
Single Indian crossroad controller — 4-Phase signal system.

One intersection — 4 roads (North, South, East, West).
Upload one video per road.
Real YOLO detects Indian traffic on each road.

4-Phase signal control (IRC standard):
  Phase A — North + South Straight (simultaneous)
  Phase B — North + South Right Turns (dedicated green arrow)
  Phase C — East + West Straight (simultaneous)
  Phase D — East + West Right Turns (dedicated green arrow)

ROI-based lane counting:
  Left 65% of frame width  → straight lane
  Right 35% of frame width → right-turn lane

Phase selection: highest combined PCU score of active roads wins.
Ambulance: immediate 90s priority override, all phases suspended.
"""

import os
import json
import copy
import time
import uuid
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
from flask_jwt_extended import jwt_required

crossroad_bp = Blueprint("crossroad", __name__)

# ── Vehicle classes that best.pt was trained on ─────────────────
# Matches data.yaml: car(0) motorcycle(1) bus(2) truck(3) ambulance(4)
INDIAN_VEHICLES = ["car", "motorcycle", "bus", "truck", "ambulance"]

# Normalise any raw model output name variants to canonical names
COCO_TO_INDIAN = {
    "car":        "car",
    "van":        "car",
    "motorcycle": "motorcycle",
    "motorbike":  "motorcycle",
    "bus":        "bus",
    "truck":      "truck",
    "ambulance":  "ambulance",
}

VEHICLE_WEIGHT = {
    "car":        1.0,
    "motorcycle": 0.5,
    "bus":        2.5,
    "truck":      2.0,
    "ambulance":  10.0,  # absolute priority — triggers emergency override
}

ROADS = ["north", "south", "east", "west"]
ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}

# ── Faster R-CNN lazy loader (module-level singleton) ──────────
_frcnn_model = None
_frcnn_lock  = threading.Lock()


def _get_faster_rcnn():
    """
    Return the shared FasterRCNNPredictor for ambulance verification.
    Safe to call from multiple road-processing threads simultaneously.
    Returns None if weights are missing so detection degrades gracefully.
    """
    global _frcnn_model
    if _frcnn_model is None:
        with _frcnn_lock:
            if _frcnn_model is None:
                try:
                    from flask import current_app
                    weights = current_app.config.get("FASTER_RCNN_WEIGHTS")
                except RuntimeError:
                    weights = None
                from ml_models.faster_rcnn.predictor import get_faster_rcnn as _load
                _frcnn_model = _load(weights) or "unavailable"
    return _frcnn_model if _frcnn_model != "unavailable" else None

# ── JSON Event Schema ──────────────────────────────────────────
#
# TELEMETRY_UPDATE — published by the orchestrator every 1 s
# {
#   "event": "TELEMETRY_UPDATE",
#   "intersection_state": {
#     "<road>": {
#       "lane":   "North",
#       "counts": {"car": 2, "motorcycle": 10, "auto_rickshaw": 5, "bus": 1, ...},
#       "total":  18,
#       "alert":  null            # or "Ambulance in North Lane"
#     }, ...
#   },
#   "emergency":      false,
#   "emergency_road": null,
#   "signal_mode":    "auto",
#   "timestamp":      "ISO-8601"
# }
#
# EMERGENCY_DETECTED — same shape but event = "EMERGENCY_DETECTED" and
#   top-level "alert": "Ambulance in <Road> Lane"
#
EVENT_TYPES = {
    "TELEMETRY_UPDATE":    "TELEMETRY_UPDATE",
    "EMERGENCY_DETECTED":  "EMERGENCY_DETECTED",
    "SIGNAL_CHANGE":       "SIGNAL_CHANGE",
}

# ── 4-Phase signal definitions ─────────────────────────────────
PHASES = {
    "A": {
        "label":    "Phase A — N+S Straight",
        "desc":     "North & South straight lanes move simultaneously. East & West stopped.",
        "roads":    ["north", "south"],
        "movement": "straight",
        "arrows":   "↑↓",
    },
    "B": {
        "label":    "Phase B — N+S Right Turns",
        "desc":     "North & South dedicated right-turn arrow. Prevents head-on collisions.",
        "roads":    ["north", "south"],
        "movement": "right_turn",
        "arrows":   "↗↙",
    },
    "C": {
        "label":    "Phase C — E+W Straight",
        "desc":     "East & West straight lanes move simultaneously. North & South stopped.",
        "roads":    ["east", "west"],
        "movement": "straight",
        "arrows":   "→←",
    },
    "D": {
        "label":    "Phase D — E+W Right Turns",
        "desc":     "East & West dedicated right-turn arrow. Prevents head-on collisions.",
        "roads":    ["east", "west"],
        "movement": "right_turn",
        "arrows":   "↘↖",
    },
}

YOLO_MODELS = {
    # ── Custom trained on Indian traffic (5 classes: car/motorcycle/bus/truck/ambulance)
    "custom":   {"name": "Custom Indian Traffic", "speed": "fast", "map": "trained", "rec": True},
    # ── YOLOv11 (Latest) ──────────────────────────────────────
    "yolov11n": {"name": "YOLOv11n", "speed": "fastest", "map": "39.5", "rec": False},
    "yolov11s": {"name": "YOLOv11s", "speed": "fast",    "map": "47.0", "rec": False},
    "yolov11m": {"name": "YOLOv11m", "speed": "medium",  "map": "51.5", "rec": False},
    # ── YOLOv8 (stable) ───────────────────────────────────────
    "yolov8n":  {"name": "YOLOv8n",  "speed": "fastest", "map": "37.3", "rec": False},
    "yolov8s":  {"name": "YOLOv8s",  "speed": "fast",    "map": "44.9", "rec": False},
    "yolov8m":  {"name": "YOLOv8m",  "speed": "medium",  "map": "50.2", "rec": False},
    "yolov8l":  {"name": "YOLOv8l",  "speed": "slow",    "map": "52.9", "rec": False},
    # ── YOLOv9 / v10 ─────────────────────────────────────────
    "yolov9c":  {"name": "YOLOv9c",  "speed": "medium",  "map": "53.0", "rec": False},
    "yolov10s": {"name": "YOLOv10s", "speed": "fast",    "map": "46.3", "rec": False},
}


# ── Dynamic signal timing — Webster's formula (Indian traffic) ─
_MIN_GREEN  = 15   # IRC minimum pedestrian crossing time
_MAX_GREEN  = 90   # practical cap
_LOST_TIME  = 4 * 4  # 4 phases × 4 s lost per phase change (amber + all-red)


def _webster_cycle(total_pcu: float) -> int:
    """
    Webster optimal cycle: C = (1.5L + 5) / (1 - Y)
    L = total lost time, Y = saturation ratio (capped at 0.85)
    Adapted for Indian 4-phase intersection PCU loads.
    """
    y = min(0.85, total_pcu / 150.0)
    c = (1.5 * _LOST_TIME + 5) / max(0.05, 1.0 - y)
    return max(60, min(180, round(c)))


def green_duration(phase_pcu: float, total_pcu: float) -> int:
    """
    Proportional green allocation within a Webster cycle.
    A phase with higher PCU share gets proportionally more green time.
    """
    if total_pcu <= 0 or phase_pcu <= 0:
        return _MIN_GREEN
    cycle      = _webster_cycle(total_pcu)
    effective  = cycle - _LOST_TIME          # usable green seconds in cycle
    proportion = min(1.0, phase_pcu / total_pcu)
    return max(_MIN_GREEN, min(_MAX_GREEN, round(proportion * effective)))


def compute_pcu(vehicle_counts: dict) -> float:
    pcu = 0.0
    for vtype, count in vehicle_counts.items():
        pcu += count * VEHICLE_WEIGHT.get(vtype, 1.0)
    return round(pcu, 2)


def _iou(b1, b2):
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def compute_roi_counts(detections: list, frame_width: int) -> dict:
    """
    Split detections into straight lane (left 65%) and right-turn lane (right 35%).
    Decision boundary: bbox center-x vs 65% of frame width.
    """
    split_x  = frame_width * 0.65
    straight = {}
    right    = {}
    for det in detections:
        bbox   = det.get("bbox", [0, 0, 0, 0])
        cx     = (bbox[0] + bbox[2]) / 2
        cls    = det.get("class", "car")
        target = straight if cx < split_x else right
        target[cls] = target.get(cls, 0) + 1

    return {
        "straight_count": sum(straight.values()),
        "straight_pcu":   round(compute_pcu(straight), 2),
        "right_count":    sum(right.values()),
        "right_pcu":      round(compute_pcu(right), 2),
    }


# ── Per-road state ─────────────────────────────────────────────
def blank_road(road: str) -> dict:
    return {
        "road":             road,
        "direction":        road.capitalize(),
        "status":           "idle",
        "progress":         0,
        "filename":         None,
        "model_used":       "",
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
        # ROI lane counts (auto 65/35 split)
        "straight_count":   0,
        "straight_pcu":     0.0,
        "right_count":      0,
        "right_pcu":        0.0,
        # Ambulance
        "ambulance_detected":   False,
        "ambulance_timestamps": [],
        # Signal (set by controller)
        "signal":           "red",
        "green_duration":   0,
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
    "active_road":    None,
    "signal_mode":    "idle",
    "emergency_road": None,
    "cycle_count":    0,
    "current_phase":  None,
    "phase_scores":   {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
    "roads":          {r: blank_road(r) for r in ROADS},
    "updated_at":     datetime.utcnow().isoformat(),
    # Rolling telemetry state — updated on every frame, broadcast to Dashboard subscribers
    "intersection_state": {
        r: {"lane": r.capitalize(), "counts": {}, "total": 0, "alert": None}
        for r in ROADS
    },
    "last_telemetry_at": None,
}
_lock = threading.Lock()


def _update_intersection_telemetry(road: str, vehicle_counts: dict, ambulance_detected: bool = False):
    """
    Update the rolling intersection_state for one lane.
    Must be called while holding _lock.

    Publishes a TELEMETRY_UPDATE (or EMERGENCY_DETECTED) event that the
    /telemetry/stream SSE endpoint will forward to all Dashboard subscribers.
    """
    counts = {k: int(v) for k, v in vehicle_counts.items() if isinstance(v, (int, float))}
    total  = sum(counts.values())
    alert  = f"Ambulance in {road.capitalize()} Lane" if ambulance_detected else None
    _crossroad["intersection_state"][road] = {
        "lane":   road.capitalize(),
        "counts": counts,
        "total":  total,
        "alert":  alert,
    }
    _crossroad["last_telemetry_at"] = datetime.utcnow().isoformat()


def update_signals():
    """
    4-Phase signal controller.
    Computes PCU scores per phase from ROI data and grants green
    to both roads in the winning phase simultaneously.
    Emergency override takes absolute priority.
    """
    global _crossroad
    road_states = _crossroad["roads"]

    # ── Emergency override — ambulance on any road ─────────────
    for road, state in road_states.items():
        if state["ambulance_detected"] and state["status"] in ("processing", "completed"):
            _crossroad["active_road"]    = road
            _crossroad["signal_mode"]    = "emergency"
            _crossroad["emergency_road"] = road
            _crossroad["current_phase"]  = None
            for r in ROADS:
                _crossroad["roads"][r]["signal"]         = "green" if r == road else "red"
                _crossroad["roads"][r]["green_duration"] = 90 if r == road else 0
            _crossroad["updated_at"] = datetime.utcnow().isoformat()
            return

    # ── Compute phase scores from ROI PCU data ─────────────────
    def road_pcu_for(road, movement):
        s = road_states[road]
        if s.get("status") not in ("completed", "processing"):
            return 0.0
        if movement == "straight":
            return s.get("straight_pcu", 0.0)
        else:
            return s.get("right_pcu", 0.0)

    phase_scores = {
        "A": road_pcu_for("north", "straight")   + road_pcu_for("south", "straight"),
        "B": road_pcu_for("north", "right_turn")  + road_pcu_for("south", "right_turn"),
        "C": road_pcu_for("east",  "straight")   + road_pcu_for("west",  "straight"),
        "D": road_pcu_for("east",  "right_turn")  + road_pcu_for("west",  "right_turn"),
    }

    # Fall back to total PCU when ROI data is all zeros (OpenCV mode — no bbox zones)
    any_roi = any(v > 0.0 for v in phase_scores.values())
    if not any_roi:
        ns = (road_states["north"].get("pcu_count", 0.0) +
              road_states["south"].get("pcu_count", 0.0))
        ew = (road_states["east"].get("pcu_count",  0.0) +
              road_states["west"].get("pcu_count",  0.0))
        phase_scores = {"A": ns, "B": 0.0, "C": ew, "D": 0.0}

    _crossroad["phase_scores"] = {k: round(v, 2) for k, v in phase_scores.items()}

    best_phase = max(phase_scores, key=phase_scores.get)
    best_score = phase_scores[best_phase]

    if best_score == 0.0:
        _crossroad["updated_at"] = datetime.utcnow().isoformat()
        return

    _crossroad["current_phase"] = best_phase
    active_roads = PHASES[best_phase]["roads"]

    if _crossroad["signal_mode"] != "manual":
        _crossroad["signal_mode"] = "auto"

    # Primary road for backward compat (first road in phase pair)
    _crossroad["active_road"] = active_roads[0]

    # Proportional green: winning phase gets time relative to its PCU share
    total_score = sum(v for v in phase_scores.values())
    dur = green_duration(best_score, total_score)

    for r in ROADS:
        if r in active_roads:
            _crossroad["roads"][r]["signal"]         = "green"
            _crossroad["roads"][r]["green_duration"] = dur
        else:
            _crossroad["roads"][r]["signal"]         = "red"
            _crossroad["roads"][r]["green_duration"] = 0

    _crossroad["updated_at"] = datetime.utcnow().isoformat()


# ── YOLO / OpenCV processing ───────────────────────────────────
_CUSTOM_YOLO_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "../../ml_models/weights/best.pt"
)

ULTRALYTICS_MAP = {
    "custom":   _CUSTOM_YOLO_WEIGHTS,   # trained on Indian traffic dataset
    "yolov11n": "yolo11n.pt", "yolov11s": "yolo11s.pt", "yolov11m": "yolo11m.pt",
    "yolov8n":  "yolov8n.pt", "yolov8s":  "yolov8s.pt",
    "yolov8m":  "yolov8m.pt", "yolov8l":  "yolov8l.pt",
    "yolov9c":  "yolov9c.pt", "yolov10s": "yolov10s.pt",
}

_yolo_cache: dict = {}
_yolo_lock  = threading.Lock()


def _get_yolo(model_key: str):
    with _yolo_lock:
        if model_key not in _yolo_cache:
            try:
                from ultralytics import YOLO
                mfile = ULTRALYTICS_MAP.get(model_key)
                if not mfile:
                    _yolo_cache[model_key] = None
                elif model_key == "custom" and not os.path.exists(mfile):
                    print(f"[YOLO] Custom weights not found at {mfile} — place best.pt there and restart")
                    _yolo_cache[model_key] = None
                else:
                    _yolo_cache[model_key] = YOLO(mfile)
                    print(f"[YOLO] loaded {model_key}: {mfile}")
            except Exception as e:
                print(f"[YOLO] load failed for {model_key}: {e}")
                _yolo_cache[model_key] = None
        return _yolo_cache[model_key]


def process_road_video(road: str, video_path: str, model_key: str, job_id: str, flask_app=None):
    """Background thread — processes one road's video with ROI-based lane counting."""
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

    yolo_model = _get_yolo(model_key)
    if yolo_model is not None:
        with _lock:
            _crossroad["roads"][road]["real_inference"] = True

    frame_results   = []
    frame_num       = 0
    sample_every    = max(1, int(fps))
    max_frames      = 100
    ambulance_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_every == 0 and len(frame_results) < max_frames:
            t0 = time.time()

            if yolo_model:
                # ── Real YOLO inference ────────────────────────
                try:
                    # Run at low conf so ambulance detections aren't silently dropped;
                    # per-class filtering below applies the real thresholds.
                    results = yolo_model(frame, verbose=False, conf=0.25)
                    r = results[0]
                    vehicle_counts = {}
                    detections     = []
                    for box in r.boxes:
                        raw      = r.names[int(box.cls)].lower()
                        mapped   = COCO_TO_INDIAN.get(raw, raw)
                        conf_val = float(box.conf)
                        # Ambulance is safety-critical — keep lower threshold
                        min_conf = 0.30 if mapped == "ambulance" else 0.40
                        if conf_val < min_conf or mapped not in INDIAN_VEHICLES:
                            continue
                        vehicle_counts[mapped] = vehicle_counts.get(mapped, 0) + 1
                        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                        detections.append({
                            "class":      mapped,
                            "confidence": round(conf_val, 3),
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

            elapsed_ms = round((time.time() - t0) * 1000, 1)
            total      = sum(vehicle_counts.values())
            ts_sec     = round(frame_num / max(fps, 1), 2)
            pcu        = compute_pcu(vehicle_counts)
            ambulance  = "ambulance" in vehicle_counts

            # ── Faster R-CNN secondary verification ───────────────
            # When YOLO flags an ambulance in this frame, crop its
            # bounding box and ask Faster R-CNN to confirm.  Only a
            # confirmed ambulance appends to ambulance_times and
            # eventually triggers the 90-second emergency override.
            frcnn_verified = False
            if ambulance:
                frcnn = _get_faster_rcnn()
                if frcnn is not None:
                    try:
                        # Find the highest-confidence ambulance detection
                        amb_boxes = [
                            d for d in detections if d["class"] == "ambulance"
                        ]
                        if amb_boxes:
                            best_box = max(amb_boxes, key=lambda d: d["confidence"])
                            x1, y1, x2, y2 = best_box["bbox"]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                crop = frame[y1:y2, x1:x2]
                                frcnn_result = frcnn.verify_numpy(crop)
                                if frcnn_result["ambulance_confirmed"]:
                                    frcnn_verified = True
                                else:
                                    # FRCNN rejected — suppress emergency override.
                                    # Ambulance bbox is KEPT in detections[] so the
                                    # CCTV overlay renders it; ambulance_detected stays
                                    # False so the 90-second green is never triggered.
                                    ambulance = False
                                    amb_bboxes = [d["bbox"] for d in amb_boxes]
                                    kept = []
                                    for _d in detections:
                                        if _d["class"] in ("car", "truck", "bus"):
                                            overlaps = any(
                                                _iou(_d["bbox"], ab) > 0.40
                                                for ab in amb_bboxes
                                            )
                                            if overlaps:
                                                # Cross-class duplicate of the ambulance —
                                                # drop to avoid double-boxing same vehicle.
                                                cnt = vehicle_counts.get(_d["class"], 1)
                                                if cnt <= 1:
                                                    vehicle_counts.pop(_d["class"], None)
                                                else:
                                                    vehicle_counts[_d["class"]] = cnt - 1
                                                continue
                                        kept.append(_d)
                                    detections = kept
                    except Exception as _frcnn_exc:
                        # On error, trust YOLO so we never silently miss a real
                        # ambulance due to a transient inference failure.
                        frcnn_verified = True
                        print(f"[{road}] FRCNN verify error (trusting YOLO): {_frcnn_exc}")
                # Weights not yet placed — trust YOLO as-is

            if ambulance:
                ambulance_times.append(ts_sec)

            # ROI lane split for this frame
            roi = compute_roi_counts(detections, max(width, 1))

            density = "high" if total > 50 else ("medium" if total > 20 else "low")
            cong_sc = round(min(total / 150, 1.0), 3)

            frame_data = {
                "frame_number":       frame_num,
                "timestamp_sec":      ts_sec,
                "vehicle_counts":     vehicle_counts,
                "detections":         detections,
                "total_vehicles":     total,
                "pcu_count":          pcu,
                "density_class":      density,
                "congestion_score":   cong_sc,
                "ambulance_detected": ambulance,
                "frcnn_verified":     frcnn_verified,
                "inference_ms":       elapsed_ms,
                "straight_count":     roi["straight_count"],
                "straight_pcu":       roi["straight_pcu"],
                "right_count":        roi["right_count"],
                "right_pcu":          roi["right_pcu"],
            }
            frame_results.append(frame_data)

            progress = min(95, int((len(frame_results) / max_frames) * 90) + 5)

            with _lock:
                if _crossroad["roads"][road].get("job_id") != job_id:
                    break
                _crossroad["roads"][road].update({
                    "current_frame":      frame_data,
                    "vehicle_counts":     vehicle_counts,
                    "total_vehicles":     total,
                    "pcu_count":          pcu,
                    "density_class":      density,
                    "congestion_score":   cong_sc,
                    "detections":         detections[:20],
                    "ambulance_detected": ambulance,
                    "ambulance_timestamps": ambulance_times[:],
                    "frames_done":        len(frame_results),
                    "progress":           progress,
                    "straight_count":     roi["straight_count"],
                    "straight_pcu":       roi["straight_pcu"],
                    "right_count":        roi["right_count"],
                    "right_pcu":          roi["right_pcu"],
                    "updated_at":         datetime.utcnow().isoformat(),
                })
                _update_intersection_telemetry(road, vehicle_counts, ambulance)
                update_signals()

        frame_num += 1

    cap.release()

    # ── Final summary ──────────────────────────────────────────
    if frame_results:
        all_v    = [f["total_vehicles"] for f in frame_results]
        avg_v    = round(sum(all_v) / len(all_v), 1)
        peak_v   = max(all_v)
        peak_frm = max(frame_results, key=lambda x: x["total_vehicles"])

        totals = {}
        for f in frame_results:
            for cls, cnt in f["vehicle_counts"].items():
                totals[cls] = totals.get(cls, 0) + cnt

        final_pcu   = compute_pcu({k: v / len(frame_results) for k, v in totals.items()})
        avg_counts  = {k: round(v / len(frame_results), 1) for k, v in totals.items()}
        amb_detected = len(ambulance_times) > 0
        density     = "high" if round(avg_v) > 50 else ("medium" if round(avg_v) > 20 else "low")
        cong_score  = round(min(round(avg_v) / 150, 1.0), 3)

        # Average ROI counts across all frames
        avg_straight_count = round(sum(f["straight_count"] for f in frame_results) / len(frame_results), 1)
        avg_straight_pcu   = round(sum(f["straight_pcu"]   for f in frame_results) / len(frame_results), 2)
        avg_right_count    = round(sum(f["right_count"]     for f in frame_results) / len(frame_results), 1)
        avg_right_pcu      = round(sum(f["right_pcu"]       for f in frame_results) / len(frame_results), 2)

        with _lock:
            _crossroad["roads"][road].update({
                "status":             "completed",
                "progress":           100,
                "frame_results":      frame_results,
                "avg_vehicles":       avg_v,
                "peak_vehicles":      peak_v,
                "peak_frame":         peak_frm,
                "vehicle_counts":     avg_counts,
                "total_vehicles":     round(avg_v),
                "pcu_count":          final_pcu,
                "density_class":      density,
                "congestion_score":   cong_score,
                "ambulance_detected": amb_detected,
                "ambulance_timestamps": ambulance_times,
                "straight_count":     avg_straight_count,
                "straight_pcu":       avg_straight_pcu,
                "right_count":        avg_right_count,
                "right_pcu":          avg_right_pcu,
                "updated_at":         datetime.utcnow().isoformat(),
            })
            _update_intersection_telemetry(road, avg_counts, amb_detected)
            _crossroad["cycle_count"] += 1
            update_signals()

        # Persist to database
        try:
            from backend.database import db
            from backend.models.traffic_record import TrafficRecord

            dominant_type = max(avg_counts, key=avg_counts.get) if avg_counts else "car"
            ctx = flask_app.app_context() if flask_app else None
            if ctx:
                ctx.push()
            try:
                rec = TrafficRecord(
                    timestamp=datetime.utcnow(),
                    intersection_id=1,
                    lane=road,
                    vehicle_type=dominant_type,
                    vehicle_count=round(avg_v),
                    density_class=density,
                    congestion_score=cong_score,
                    ambulance_detected=amb_detected,
                )
                db.session.add(rec)
                db.session.commit()
                print(f"[{road}] DB record saved: {round(avg_v)} vehicles, {density}")
            finally:
                if ctx:
                    ctx.pop()
        except Exception as e:
            print(f"[{road}] DB write failed: {e}")

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
    with _lock:
        state = {
            "name":              _crossroad["name"],
            "location":          _crossroad["location"],
            "active_road":       _crossroad["active_road"],
            "signal_mode":       _crossroad["signal_mode"],
            "emergency_road":    _crossroad["emergency_road"],
            "cycle_count":       _crossroad["cycle_count"],
            "current_phase":     _crossroad["current_phase"],
            "phase_scores":      _crossroad["phase_scores"],
            "updated_at":        _crossroad["updated_at"],
            # Live telemetry snapshot — powers the Dashboard table
            "intersection_state":  copy.deepcopy(_crossroad["intersection_state"]),
            "last_telemetry_at":   _crossroad["last_telemetry_at"],
            "roads": {
                r: {k: v for k, v in s.items() if k != "frame_results"}
                for r, s in _crossroad["roads"].items()
            },
        }
    return jsonify(state), 200


@crossroad_bp.route("/telemetry", methods=["GET"])
@jwt_required()
def get_telemetry():
    """REST snapshot of the current intersection telemetry — same data as the SSE stream."""
    with _lock:
        sig_mode   = _crossroad["signal_mode"]
        emerg_road = _crossroad.get("emergency_road")
        payload = {
            "event":              EVENT_TYPES["EMERGENCY_DETECTED"] if sig_mode == "emergency"
                                  else EVENT_TYPES["TELEMETRY_UPDATE"],
            "intersection_state": copy.deepcopy(_crossroad["intersection_state"]),
            "emergency":          sig_mode == "emergency",
            "emergency_road":     emerg_road,
            "signal_mode":        sig_mode,
            "timestamp":          datetime.utcnow().isoformat(),
        }
        if sig_mode == "emergency" and emerg_road:
            payload["alert"] = f"Ambulance in {emerg_road.capitalize()} Lane"
    return jsonify(payload), 200


@crossroad_bp.route("/telemetry/stream", methods=["GET"])
def telemetry_stream():
    """
    Server-Sent Events endpoint — broadcasts intersection_state every 1 second.

    Authentication: pass the JWT as ?jwt=<token>  (query-string location).

    Frontend usage (EventSource):
        const es = new EventSource(`/api/crossroad/telemetry/stream?jwt=${token}`)
        es.onmessage = e => setTelemetry(JSON.parse(e.data))

    Python async usage (aiohttp):
        async with session.get(url) as resp:
            async for line in resp.content:
                if line.startswith(b"data: "):
                    payload = json.loads(line[6:])
    """
    from flask_jwt_extended import verify_jwt_in_request
    try:
        verify_jwt_in_request(locations=["headers", "query_string"])
    except Exception:
        return jsonify({"error": "Missing or invalid token"}), 401

    def generate():
        while True:
            try:
                with _lock:
                    sig_mode   = _crossroad["signal_mode"]
                    emerg_road = _crossroad.get("emergency_road")
                    state      = copy.deepcopy(_crossroad["intersection_state"])

                payload = {
                    "event":              EVENT_TYPES["EMERGENCY_DETECTED"] if sig_mode == "emergency"
                                          else EVENT_TYPES["TELEMETRY_UPDATE"],
                    "intersection_state": state,
                    "emergency":          sig_mode == "emergency",
                    "emergency_road":     emerg_road,
                    "signal_mode":        sig_mode,
                    "timestamp":          datetime.utcnow().isoformat(),
                }
                if sig_mode == "emergency" and emerg_road:
                    payload["alert"] = f"Ambulance in {emerg_road.capitalize()} Lane"

                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(1)
            except GeneratorExit:
                break
            except Exception:
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@crossroad_bp.route("/models", methods=["GET"])
@jwt_required()
def list_models():
    return jsonify({"models": YOLO_MODELS}), 200


@crossroad_bp.route("/phases", methods=["GET"])
@jwt_required()
def list_phases():
    return jsonify({"phases": PHASES}), 200


@crossroad_bp.route("/upload/<road>", methods=["POST"])
@jwt_required()
def upload_road_video(road):
    if road not in ROADS:
        return jsonify({"error": f"Invalid road. Use: {ROADS}"}), 400

    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file      = request.files["video"]
    model_key = request.form.get("model", "custom")
    location  = request.form.get("location", "")

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXTS)}"}), 400

    if model_key not in YOLO_MODELS:
        model_key = "custom"

    upload_folder = current_app.config.get("UPLOAD_FOLDER", "/tmp")
    filename      = f"road_{road}_{uuid.uuid4().hex}{ext}"
    filepath      = os.path.join(upload_folder, filename)
    file.save(filepath)

    job_id = uuid.uuid4().hex[:10]

    with _lock:
        _crossroad["roads"][road] = blank_road(road)
        _crossroad["roads"][road].update({
            "job_id":     job_id,
            "status":     "queued",
            "filename":   file.filename,
            "model_used": model_key,
        })
        if location:
            _crossroad["location"] = location

    flask_app = current_app._get_current_object()
    thread = threading.Thread(
        target=process_road_video,
        args=(road, filepath, model_key, job_id, flask_app),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "job_id":  job_id,
        "road":    road,
        "message": f"{road.upper()} road video processing started",
        "model":   YOLO_MODELS[model_key]["name"],
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
        _crossroad["current_phase"]  = None
        for r in ROADS:
            _crossroad["roads"][r]["signal"]         = "green" if r == road else "red"
            _crossroad["roads"][r]["green_duration"] = 90 if r == road else 0
        _crossroad["active_road"] = road
        _crossroad["updated_at"]  = datetime.utcnow().isoformat()
        # Stamp EMERGENCY_DETECTED alert into telemetry so the Dashboard flashes a warning
        _update_intersection_telemetry(
            road,
            _crossroad["roads"][road].get("vehicle_counts", {}),
            ambulance_detected=True,
        )
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
        _crossroad["signal_mode"]    = "auto"
        # Clear the alert flag from telemetry
        _update_intersection_telemetry(
            road,
            _crossroad["roads"][road].get("vehicle_counts", {}),
            ambulance_detected=False,
        )
        update_signals()
    return jsonify({"message": "Emergency cleared — resumed auto mode", "signal_mode": "auto"}), 200


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


@crossroad_bp.route("/signal/timings", methods=["POST"])
@jwt_required()
def set_all_timings():
    data = request.get_json(silent=True) or {}
    timings = {}
    for r in ROADS:
        val = data.get(r, 30)
        timings[r] = max(0, min(120, int(val)))
    active_road = data.get("active_road")
    if active_road not in ROADS:
        active_road = max(timings, key=timings.get)
    with _lock:
        _crossroad["signal_mode"] = "manual"
        _crossroad["active_road"] = active_road
        for r in ROADS:
            _crossroad["roads"][r]["signal"]         = "green" if r == active_road else "red"
            _crossroad["roads"][r]["green_duration"] = timings[r]
        _crossroad["updated_at"] = datetime.utcnow().isoformat()
    return jsonify({
        "message":     f"Signal timings updated — {active_road.upper()} active",
        "timings":     timings,
        "active_road": active_road,
        "signal_mode": "manual",
    }), 200


@crossroad_bp.route("/signal/auto", methods=["POST"])
@jwt_required()
def resume_auto():
    with _lock:
        _crossroad["signal_mode"] = "auto"
        update_signals()
    return jsonify({"message": "Switched to AI auto mode", "signal_mode": "auto"}), 200


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
        _crossroad["current_phase"]  = None
        _crossroad["phase_scores"]   = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        _crossroad["updated_at"]     = datetime.utcnow().isoformat()
    return jsonify({"message": "All roads reset"}), 200
