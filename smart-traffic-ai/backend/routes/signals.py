"""backend/routes/signals.py — Traffic signal control endpoints."""

from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from backend.database import db
from backend.models.traffic_record import TrafficSignal
from backend.models.user import User

signals_bp = Blueprint("signals", __name__)


@signals_bp.route("/", methods=["GET"])
@jwt_required()
def list_signals():
    signals = TrafficSignal.query.all()
    return jsonify({"signals": [s.to_dict() for s in signals]}), 200


@signals_bp.route("/<int:signal_id>", methods=["GET"])
@jwt_required()
def get_signal(signal_id):
    s = TrafficSignal.query.get_or_404(signal_id)
    return jsonify({"signal": s.to_dict()}), 200


@signals_bp.route("/<int:signal_id>/update", methods=["PATCH"])
@jwt_required()
def update_signal(signal_id):
    """Update signal timings (operator/admin only)."""
    caller = User.query.get(int(get_jwt_identity()))
    if caller.role not in ("admin", "operator"):
        return jsonify({"error": "Insufficient permissions"}), 403

    data = request.get_json(silent=True) or {}
    s = TrafficSignal.query.get_or_404(signal_id)

    timings = data.get("timings", {})
    if "north" in timings:
        s.north_green = int(timings["north"])
    if "south" in timings:
        s.south_green = int(timings["south"])
    if "east" in timings:
        s.east_green = int(timings["east"])
    if "west" in timings:
        s.west_green = int(timings["west"])

    if "status" in data:
        s.status = data["status"]
    if "emergency_lane" in data:
        s.emergency_lane = data["emergency_lane"]

    s.last_optimized = datetime.utcnow()
    db.session.commit()
    return jsonify({"signal": s.to_dict()}), 200


@signals_bp.route("/<int:signal_id>/emergency", methods=["POST"])
@jwt_required()
def trigger_emergency(signal_id):
    """Activate emergency override for ambulance passage."""
    data = request.get_json(silent=True) or {}
    lane = data.get("lane", "north")

    s = TrafficSignal.query.get_or_404(signal_id)
    s.status = "emergency"
    s.emergency_lane = lane
    # Give 90s green to emergency lane
    for attr in ["north_green", "south_green", "east_green", "west_green"]:
        setattr(s, attr, 5)
    setattr(s, f"{lane}_green", 90)
    s.last_optimized = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "signal": s.to_dict(),
        "message": f"Emergency override activated — {lane} lane priority",
    }), 200


@signals_bp.route("/<int:signal_id>/reset", methods=["POST"])
@jwt_required()
def reset_signal(signal_id):
    """Reset signal to normal ML-optimized state."""
    s = TrafficSignal.query.get_or_404(signal_id)
    s.status = "active"
    s.emergency_lane = None
    s.north_green = 30
    s.south_green = 30
    s.east_green = 25
    s.west_green = 25
    s.last_optimized = datetime.utcnow()
    db.session.commit()
    return jsonify({"signal": s.to_dict(), "message": "Signal reset to normal operation"}), 200
