"""
backend/routes/video_detection.py
===================================
Video upload + frame-by-frame YOLO detection using the custom-trained best.pt.

Trained classes: car, motorcycle, bus, truck, ambulance

Endpoints:
  POST /api/video/upload          — upload + analyse video
  GET  /api/video/models          — list available models
  GET  /api/video/result/<job_id> — poll job status / result
  GET  /api/video/jobs            — list recent jobs
"""

import os
import uuid
import time
import threading
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

video_bp = Blueprint("video", __name__)

# ── In-memory job store ────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ── Custom model path ──────────────────────────────────────────
_BEST_PT = os.path.join(
    os.path.dirname(__file__), "../../ml_models/weights/best.pt"
)

# ── Model registry: single custom entry ───────────────────────
YOLO_MODELS = {
    "custom": {
        "name": "Custom Indian Traffic Model",
        "desc": "Trained on Indian traffic — car, motorcycle, bus, truck, ambulance",
        "speed": "fast",
        "map":   "trained",
    }
}

# Trained classes exactly as they appear in best.pt / data.yaml
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "ambulance"]

# Raw → canonical mapping in case model outputs slight name variants
_CLASS_MAP = {
    "car":        "car",
    "motorcycle": "motorcycle",
    "motorbike":  "motorcycle",
    "bus":        "bus",
    "truck":      "truck",
    "ambulance":  "ambulance",
}

ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}

# ── Singleton model cache ─────────────────────────────────────
_yolo_model = None
_yolo_lock  = threading.Lock()


def _get_model():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            try:
                from ultralytics import YOLO
                if os.path.exists(_BEST_PT):
                    _yolo_model = YOLO(_BEST_PT)
                    print(f"[video_detection] Loaded {_BEST_PT}")
                else:
                    print(f"[video_detection] best.pt not found at {_BEST_PT} — "
                          "place file there and restart")
                    _yolo_model = "unavailable"
            except Exception as e:
                print(f"[video_detection] YOLO load failed: {e}")
                _yolo_model = "unavailable"
    return _yolo_model if _yolo_model != "unavailable" else None


def allowed_video(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTS


# ── Core inference logic ───────────────────────────────────────

def _infer_frame(model, frame) -> tuple[list, dict, float]:
    """
    Run best.pt on one frame. Returns (detections, vehicle_counts, inference_ms).
    Ambulance uses a lower per-class confidence threshold (safety-critical).
    """
    t0 = time.time()
    results = model(frame, verbose=False, conf=0.25)
    r = results[0]
    detections     = []
    vehicle_counts = {}
    for box in r.boxes:
        raw      = r.names[int(box.cls)].lower()
        mapped   = _CLASS_MAP.get(raw, raw)
        conf_val = float(box.conf)
        min_conf = 0.30 if mapped == "ambulance" else 0.40
        if conf_val < min_conf or mapped not in VEHICLE_CLASSES:
            continue
        vehicle_counts[mapped] = vehicle_counts.get(mapped, 0) + 1
        detections.append({
            "class":      mapped,
            "confidence": round(conf_val, 3),
            "bbox":       [round(x) for x in box.xyxy[0].tolist()],
            "track_id":   -1,
        })
    return detections, vehicle_counts, round((time.time() - t0) * 1000, 1)


def run_video_job(job_id: str, video_path: str, model_key: str,
                  fps: float, total_frames: int, duration: float):
    """Background thread — processes video with best.pt and updates job state."""
    import cv2

    try:
        with _jobs_lock:
            _jobs[job_id]["status"]   = "processing"
            _jobs[job_id]["progress"] = 5

        model = _get_model()
        if model is None:
            with _jobs_lock:
                _jobs[job_id].update({
                    "status": "failed",
                    "error":  "YOLO model unavailable — place best.pt in ml_models/weights/ and restart",
                })
            return

        cap = cv2.VideoCapture(video_path)
        sample_every  = max(1, int(fps))
        frame_results = []
        frame_num     = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_every == 0:
                detections, vehicle_counts, elapsed_ms = _infer_frame(model, frame)

                total   = sum(vehicle_counts.values())
                density = "high" if total > 50 else ("medium" if total > 20 else "low")

                frame_results.append({
                    "frame_number":       frame_num,
                    "timestamp_sec":      round(frame_num / max(fps, 1), 2),
                    "detections":         detections,
                    "vehicle_counts":     vehicle_counts,
                    "total_vehicles":     total,
                    "density_class":      density,
                    "ambulance_detected": "ambulance" in vehicle_counts,
                    "inference_ms":       elapsed_ms,
                })

                progress = min(95, int((frame_num / max(total_frames, 1)) * 100))
                with _jobs_lock:
                    if job_id in _jobs:
                        _jobs[job_id]["progress"]    = progress
                        _jobs[job_id]["frames_done"] = len(frame_results)

                if len(frame_results) >= 120:
                    break

            frame_num += 1

        cap.release()

        # ── Build summary ──────────────────────────────────────
        if frame_results:
            all_vehicles   = [r["total_vehicles"] for r in frame_results]
            avg_veh        = round(sum(all_vehicles) / len(all_vehicles), 1)
            max_veh        = max(all_vehicles)
            min_veh        = min(all_vehicles)

            density_counts = {"low": 0, "medium": 0, "high": 0}
            vehicle_totals = {}
            ambulance_frames = []

            for r in frame_results:
                d = r.get("density_class", "low")
                if d in density_counts:
                    density_counts[d] += 1
                for cls, cnt in r.get("vehicle_counts", {}).items():
                    vehicle_totals[cls] = vehicle_totals.get(cls, 0) + cnt
                if r.get("ambulance_detected"):
                    ambulance_frames.append(r["timestamp_sec"])

            peak_frame      = max(frame_results, key=lambda x: x["total_vehicles"])
            overall_density = max(density_counts, key=density_counts.get)
            avg_ms = round(
                sum(r.get("inference_ms", 0) for r in frame_results) / len(frame_results), 1
            )

            summary = {
                "total_frames_analyzed":  len(frame_results),
                "avg_vehicles_per_frame": avg_veh,
                "max_vehicles_in_frame":  max_veh,
                "min_vehicles_in_frame":  min_veh,
                "overall_density":        overall_density,
                "density_breakdown":      density_counts,
                "vehicle_type_totals":    vehicle_totals,
                "ambulance_detected":     len(ambulance_frames) > 0,
                "ambulance_timestamps":   ambulance_frames[:10],
                "peak_frame": {
                    "frame_number":   peak_frame["frame_number"],
                    "timestamp_sec":  peak_frame["timestamp_sec"],
                    "total_vehicles": peak_frame["total_vehicles"],
                },
                "avg_inference_ms": avg_ms,
            }
        else:
            summary = {}

        with _jobs_lock:
            _jobs[job_id].update({
                "status":         "completed",
                "progress":       100,
                "frame_results":  frame_results,
                "summary":        summary,
                "real_inference": True,
                "completed_at":   datetime.utcnow().isoformat(),
            })

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({"status": "failed", "error": str(e)})
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass


# ── Endpoints ──────────────────────────────────────────────────

@video_bp.route("/models", methods=["GET"])
@jwt_required()
def list_models():
    return jsonify({"models": YOLO_MODELS}), 200


@video_bp.route("/upload", methods=["POST"])
@jwt_required()
def upload_video():
    """
    POST /api/video/upload
    Form fields:
      video  — video file
      model  — model key (only "custom" is supported)
    Returns job_id for polling.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file      = request.files["video"]
    model_key = request.form.get("model", "custom")
    if model_key not in YOLO_MODELS:
        model_key = "custom"

    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_video(file.filename):
        return jsonify({
            "error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTS)}"
        }), 400

    # Save file
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "/tmp")
    ext      = os.path.splitext(file.filename)[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Video metadata
    try:
        import cv2
        cap          = cv2.VideoCapture(filepath)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = round(total_frames / fps, 1) if fps > 0 else 0
    except Exception:
        fps = 30; total_frames = 900; width = 1280; height = 720; duration = 30

    job_id  = uuid.uuid4().hex[:12]
    user_id = get_jwt_identity()

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id":        job_id,
            "user_id":       user_id,
            "status":        "queued",
            "progress":      0,
            "frames_done":   0,
            "model_key":     model_key,
            "model_name":    YOLO_MODELS[model_key]["name"],
            "filename":      file.filename,
            "video_info": {
                "fps":          round(fps, 1),
                "total_frames": total_frames,
                "duration_sec": duration,
                "resolution":   f"{width}x{height}",
            },
            "frame_results":  [],
            "summary":        {},
            "real_inference": False,
            "created_at":     datetime.utcnow().isoformat(),
            "completed_at":   None,
        }

    threading.Thread(
        target=run_video_job,
        args=(job_id, filepath, model_key, fps, total_frames, duration),
        daemon=True,
    ).start()

    return jsonify({
        "job_id":     job_id,
        "status":     "queued",
        "model":      YOLO_MODELS[model_key]["name"],
        "video_info": _jobs[job_id]["video_info"],
        "message":    "Video processing started. Poll /api/video/result/<job_id>",
    }), 202


@video_bp.route("/result/<job_id>", methods=["GET"])
@jwt_required()
def get_result(job_id):
    with _jobs_lock:
        if job_id not in _jobs:
            return jsonify({"error": "Job not found"}), 404
        job = _jobs[job_id].copy()

    if job["status"] != "completed":
        job.pop("frame_results", None)

    return jsonify(job), 200


@video_bp.route("/jobs", methods=["GET"])
@jwt_required()
def list_jobs():
    user_id = get_jwt_identity()
    with _jobs_lock:
        user_jobs = [
            {k: v for k, v in j.items() if k != "frame_results"}
            for j in _jobs.values()
            if j.get("user_id") == user_id
        ]
    user_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"jobs": user_jobs[:20]}), 200


@video_bp.route("/jobs/<job_id>", methods=["DELETE"])
@jwt_required()
def delete_job(job_id):
    with _jobs_lock:
        if job_id in _jobs:
            del _jobs[job_id]
    return jsonify({"message": "Job deleted"}), 200
