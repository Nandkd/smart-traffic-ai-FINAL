import os
import time
import random
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required

live_bp = Blueprint("live", __name__)

INTERSECTIONS = {
    1: {"name": "MG Road Junction",  "lat": 28.6139, "lng": 77.2090},
    2: {"name": "Connaught Place",   "lat": 28.6304, "lng": 77.2177},
    3: {"name": "India Gate",        "lat": 28.6129, "lng": 77.2295},
    4: {"name": "Rajpath Crossing",  "lat": 28.6145, "lng": 77.2090},
}

LANES = ["north", "south", "east", "west"]
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "ambulance"]
CLASS_WEIGHTS   = [0.52, 0.18, 0.12, 0.11, 0.07]

_live_state = {}
_state_lock = threading.Lock()


def _init_state():
    for iid, info in INTERSECTIONS.items():
        _live_state[iid] = {
            "intersection_id": iid,
            "name": info["name"],
            "lat": info["lat"],
            "lng": info["lng"],
            "signals": {"north": "red", "south": "red", "east": "green", "west": "red"},
            "active_lane": "east",
            "phase": "green",
            "phase_start": time.time(),
            "green_duration": 30,
            "lane_counts": {"north": 0, "south": 0, "east": 0, "west": 0},
            "detections": [],
            "total_vehicles": 0,
            "vehicle_counts": {},
            "density_class": "low",
            "congestion_score": 0.0,
            "ambulance_detected": False,
            "ambulance_lane": None,
            "ambulance_timer": 0,
            "model_used": "yolov8",
            "fps": 30,
            "inference_ms": 12.0,
            "cycle_count": 0,
            "vehicles_cleared": 0,
            "last_updated": datetime.utcnow().isoformat(),
        }


_init_state()


def _simulate_detection(iid):
    hour = datetime.utcnow().hour
    is_peak = hour in (7, 8, 9, 17, 18, 19, 20)
    base = random.randint(60, 120) if is_peak else random.randint(5, 40)
    total = max(0, base + random.randint(-5, 5))

    lane_counts = {}
    remaining = total
    for i, lane in enumerate(LANES):
        if i == len(LANES) - 1:
            lane_counts[lane] = remaining
        else:
            share = random.randint(0, max(0, remaining // (len(LANES) - i)))
            lane_counts[lane] = share
            remaining -= share

    vehicle_counts = {}
    detections = []
    for _ in range(min(total, 15)):
        cls = random.choices(VEHICLE_CLASSES, weights=CLASS_WEIGHTS)[0]
        vehicle_counts[cls] = vehicle_counts.get(cls, 0) + 1
        detections.append({
            "class": cls,
            "confidence": round(random.uniform(0.72, 0.98), 3),
            "bbox": [random.randint(0, 500), random.randint(0, 300),
                     random.randint(50, 200), random.randint(30, 120)],
            "track_id": random.randint(1, 50),
            "lane": random.choice(LANES),
        })

    ambulance = "ambulance" in vehicle_counts
    density = "high" if total > 70 else ("medium" if total > 25 else "low")
    score = round(
        random.uniform(0.65, 0.95) if density == "high"
        else random.uniform(0.3, 0.65) if density == "medium"
        else random.uniform(0.05, 0.3), 3
    )

    return {
        "lane_counts": lane_counts,
        "vehicle_counts": vehicle_counts,
        "detections": detections,
        "total_vehicles": total,
        "density_class": density,
        "congestion_score": score,
        "ambulance_detected": ambulance,
        "ambulance_lane": random.choice(LANES) if ambulance else None,
        "inference_ms": round(random.uniform(9, 18), 1),
        "fps": random.randint(28, 32),
    }


def _compute_green(count):
    if count <= 5:   return 10
    if count <= 20:  return 20
    if count <= 40:  return 30
    if count <= 70:  return 50
    return 75


def _next_lane(state):
    if state["ambulance_detected"] and state["ambulance_lane"]:
        return state["ambulance_lane"]
    counts = state["lane_counts"]
    other = [l for l in LANES if l != state["active_lane"]]
    return max(other, key=lambda l: counts.get(l, 0))


def _tick_signals(state):
    elapsed = time.time() - state["phase_start"]

    if state["ambulance_detected"]:
        amb = state["ambulance_lane"]
        if amb:
            for l in LANES:
                state["signals"][l] = "green" if l == amb else "red"
            state["active_lane"] = amb
            state["green_duration"] = 90
            state["ambulance_timer"] = state.get("ambulance_timer", 0) + 1
            if state["ambulance_timer"] >= 90:
                state["ambulance_detected"] = False
                state["ambulance_lane"] = None
                state["ambulance_timer"] = 0
        return

    if state["phase"] == "green" and elapsed >= state["green_duration"]:
        state["phase"] = "yellow"
        state["signals"][state["active_lane"]] = "yellow"
        state["phase_start"] = time.time()
        state["vehicles_cleared"] += state["lane_counts"].get(state["active_lane"], 0)

    elif state["phase"] == "yellow" and elapsed >= 3:
        next_l = _next_lane(state)
        dur = _compute_green(state["lane_counts"].get(next_l, 0))
        for l in LANES:
            state["signals"][l] = "green" if l == next_l else "red"
        state["active_lane"] = next_l
        state["phase"] = "green"
        state["phase_start"] = time.time()
        state["green_duration"] = dur
        state["cycle_count"] += 1


def _background_loop():
    while True:
        with _state_lock:
            for iid, state in _live_state.items():
                if int(time.time()) % 2 == 0:
                    det = _simulate_detection(iid)
                    state.update({
                        "lane_counts":      det["lane_counts"],
                        "vehicle_counts":   det["vehicle_counts"],
                        "detections":       det["detections"],
                        "total_vehicles":   det["total_vehicles"],
                        "density_class":    det["density_class"],
                        "congestion_score": det["congestion_score"],
                        "inference_ms":     det["inference_ms"],
                        "fps":              det["fps"],
                        "last_updated":     datetime.utcnow().isoformat(),
                    })
                    if det["ambulance_detected"] and not state["ambulance_detected"]:
                        state["ambulance_detected"] = True
                        state["ambulance_lane"] = det["ambulance_lane"]
                        state["ambulance_timer"] = 0
                _tick_signals(state)
        time.sleep(1)


_bg_thread = threading.Thread(target=_background_loop, daemon=True)
_bg_thread.start()


@live_bp.route("/states", methods=["GET"])
@jwt_required()
def get_all_states():
    with _state_lock:
        states = []
        for iid, state in _live_state.items():
            elapsed = time.time() - state["phase_start"]
            remaining = max(0, state["green_duration"] - elapsed)
            s = state.copy()
            s["remaining_seconds"] = round(remaining, 1)
            states.append(s)
    return jsonify({"intersections": states}), 200


@live_bp.route("/state/<int:iid>", methods=["GET"])
@jwt_required()
def get_state(iid):
    with _state_lock:
        if iid not in _live_state:
            return jsonify({"error": "Not found"}), 404
        state = _live_state[iid].copy()
        elapsed = time.time() - state["phase_start"]
        state["remaining_seconds"] = round(max(0, state["green_duration"] - elapsed), 1)
    return jsonify({"state": state}), 200


@live_bp.route("/detect/<int:iid>", methods=["POST"])
@jwt_required()
def trigger_detection(iid):
    model = request.json.get("model", "yolov8") if request.json else "yolov8"
    with _state_lock:
        if iid not in _live_state:
            return jsonify({"error": "Not found"}), 404
        det = _simulate_detection(iid)
        _live_state[iid].update({
            "lane_counts":      det["lane_counts"],
            "vehicle_counts":   det["vehicle_counts"],
            "detections":       det["detections"],
            "total_vehicles":   det["total_vehicles"],
            "density_class":    det["density_class"],
            "congestion_score": det["congestion_score"],
            "inference_ms":     det["inference_ms"],
            "fps":              det["fps"],
            "model_used":       model,
            "last_updated":     datetime.utcnow().isoformat(),
        })
        result = _live_state[iid].copy()
    return jsonify({"result": result}), 200


@live_bp.route("/ambulance/<int:iid>", methods=["POST"])
@jwt_required()
def trigger_ambulance(iid):
    data = request.get_json(silent=True) or {}
    lane = data.get("lane", "north")
    with _state_lock:
        if iid not in _live_state:
            return jsonify({"error": "Not found"}), 404
        _live_state[iid]["ambulance_detected"] = True
        _live_state[iid]["ambulance_lane"] = lane
        _live_state[iid]["ambulance_timer"] = 0
        for l in LANES:
            _live_state[iid]["signals"][l] = "green" if l == lane else "red"
        _live_state[iid]["active_lane"] = lane
        _live_state[iid]["green_duration"] = 90
        _live_state[iid]["phase"] = "green"
        _live_state[iid]["phase_start"] = time.time()
    return jsonify({
        "message": f"Ambulance clearance activated — {lane} lane",
        "lane": lane,
        "green_duration": 90,
    }), 200


@live_bp.route("/ambulance/<int:iid>/clear", methods=["POST"])
@jwt_required()
def clear_ambulance(iid):
    with _state_lock:
        if iid not in _live_state:
            return jsonify({"error": "Not found"}), 404
        _live_state[iid]["ambulance_detected"] = False
        _live_state[iid]["ambulance_lane"] = None
        _live_state[iid]["ambulance_timer"] = 0
    return jsonify({"message": "Ambulance cleared", "status": "normal"}), 200


@live_bp.route("/signal/<int:iid>/override", methods=["POST"])
@jwt_required()
def override_signal(iid):
    data = request.get_json(silent=True) or {}
    lane = data.get("lane", "north")
    duration = int(data.get("duration", 30))
    with _state_lock:
        if iid not in _live_state:
            return jsonify({"error": "Not found"}), 404
        for l in LANES:
            _live_state[iid]["signals"][l] = "green" if l == lane else "red"
        _live_state[iid]["active_lane"] = lane
        _live_state[iid]["green_duration"] = duration
        _live_state[iid]["phase"] = "green"
        _live_state[iid]["phase_start"] = time.time()
    return jsonify({"message": f"Override: {lane} green for {duration}s"}), 200


@live_bp.route("/model/switch", methods=["POST"])
@jwt_required()
def switch_model():
    data = request.get_json(silent=True) or {}
    model = data.get("model", "yolov8")
    with _state_lock:
        for state in _live_state.values():
            state["model_used"] = model
    return jsonify({"message": f"Switched to {model}", "model": model}), 200


@live_bp.route("/summary", methods=["GET"])
@jwt_required()
def get_summary():
    with _state_lock:
        total_v    = sum(s["total_vehicles"] for s in _live_state.values())
        amb_active = sum(1 for s in _live_state.values() if s["ambulance_detected"])
        densities  = [s["density_class"] for s in _live_state.values()]
        avg_cong   = sum(s["congestion_score"] for s in _live_state.values()) / len(_live_state)
        cleared    = sum(s["vehicles_cleared"] for s in _live_state.values())
        avg_fps    = sum(s["fps"] for s in _live_state.values()) / len(_live_state)
    return jsonify({
        "total_vehicles_live":    total_v,
        "ambulance_active_count": amb_active,
        "density_per_intersection": densities,
        "avg_congestion_score":   round(avg_cong, 3),
        "total_vehicles_cleared": cleared,
        "avg_fps":                round(avg_fps, 1),
        "intersections_count":    len(_live_state),
        "timestamp":              datetime.utcnow().isoformat(),
    }), 200