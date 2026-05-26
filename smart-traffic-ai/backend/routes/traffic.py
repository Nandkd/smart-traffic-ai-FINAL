"""backend/routes/traffic.py — Traffic data CRUD & stats endpoints."""

import random
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from sqlalchemy import func, desc

from backend.app import db
from backend.models.traffic_record import TrafficRecord, TrafficSignal

traffic_bp = Blueprint("traffic", __name__)


@traffic_bp.route("/density", methods=["GET"])
@jwt_required()
def get_density():
    """Return current live traffic density for all intersections."""
    signals = TrafficSignal.query.all()
    result = []
    for s in signals:
        # Latest record for this intersection
        rec = (TrafficRecord.query
               .filter_by(intersection_id=s.id)
               .order_by(desc(TrafficRecord.timestamp))
               .first())
        result.append({
            "intersection": s.to_dict(),
            "current": rec.to_dict() if rec else None,
        })
    return jsonify({"data": result}), 200


@traffic_bp.route("/history", methods=["GET"])
@jwt_required()
def get_history():
    """Return historical traffic records with optional filters."""
    intersection_id = request.args.get("intersection_id", type=int)
    hours = request.args.get("hours", 24, type=int)
    lane = request.args.get("lane")
    since = datetime.utcnow() - timedelta(hours=hours)

    q = TrafficRecord.query.filter(TrafficRecord.timestamp >= since)
    if intersection_id:
        q = q.filter_by(intersection_id=intersection_id)
    if lane:
        q = q.filter_by(lane=lane)

    records = q.order_by(desc(TrafficRecord.timestamp)).limit(500).all()
    return jsonify({"records": [r.to_dict() for r in records], "count": len(records)}), 200


@traffic_bp.route("/stats", methods=["GET"])
@jwt_required()
def get_stats():
    """Dashboard KPI statistics."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total_vehicles = db.session.query(func.sum(TrafficRecord.vehicle_count)).scalar() or 0
    today_vehicles = (db.session.query(func.sum(TrafficRecord.vehicle_count))
                      .filter(TrafficRecord.timestamp >= today_start).scalar() or 0)
    ambulance_count = TrafficRecord.query.filter_by(ambulance_detected=True).count()
    active_signals = TrafficSignal.query.filter_by(status="active").count()
    emergency_signals = TrafficSignal.query.filter_by(status="emergency").count()

    # Density breakdown (last 24h)
    since_24h = now - timedelta(hours=24)
    density_q = (db.session.query(TrafficRecord.density_class, func.count())
                 .filter(TrafficRecord.timestamp >= since_24h)
                 .group_by(TrafficRecord.density_class).all())
    density_dist = {row[0]: row[1] for row in density_q}

    # Avg congestion score
    avg_cong = (db.session.query(func.avg(TrafficRecord.congestion_score))
                .filter(TrafficRecord.timestamp >= since_24h).scalar() or 0.0)

    return jsonify({
        "total_vehicles": int(total_vehicles),
        "today_vehicles": int(today_vehicles),
        "ambulance_events": ambulance_count,
        "active_signals": active_signals,
        "emergency_signals": emergency_signals,
        "density_distribution": density_dist,
        "avg_congestion_score": round(float(avg_cong), 3),
        "system_uptime_pct": 99.4,
    }), 200


@traffic_bp.route("/hourly", methods=["GET"])
@jwt_required()
def get_hourly():
    """Vehicle counts grouped by hour for the last 24 hours (for line chart)."""
    since = datetime.utcnow() - timedelta(hours=24)
    rows = (db.session.query(
                func.strftime("%H", TrafficRecord.timestamp).label("hour"),
                func.sum(TrafficRecord.vehicle_count).label("total")
            )
            .filter(TrafficRecord.timestamp >= since)
            .group_by("hour")
            .all())
    data = [{"hour": int(r.hour), "vehicles": int(r.total or 0)} for r in rows]
    data.sort(key=lambda x: x["hour"])
    return jsonify({"hourly": data}), 200


@traffic_bp.route("/record", methods=["POST"])
@jwt_required()
def add_record():
    """Ingest a new traffic detection event."""
    data = request.get_json(silent=True) or {}
    try:
        rec = TrafficRecord(
            intersection_id=data["intersection_id"],
            lane=data.get("lane", "north"),
            vehicle_type=data.get("vehicle_type", "car"),
            vehicle_count=int(data.get("vehicle_count", 1)),
            density_class=data.get("density_class", "low"),
            congestion_score=float(data.get("congestion_score", 0.0)),
            ambulance_detected=bool(data.get("ambulance_detected", False)),
            confidence=data.get("confidence"),
        )
        db.session.add(rec)
        db.session.commit()
        return jsonify({"record": rec.to_dict()}), 201
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
