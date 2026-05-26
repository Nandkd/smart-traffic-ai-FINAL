from datetime import datetime
from backend.database import db


class TrafficSignal(db.Model):
    __tablename__ = "traffic_signals"

    id = db.Column(db.Integer, primary_key=True)
    location_name = db.Column(db.String(120), nullable=False)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    north_green = db.Column(db.Integer, default=30)
    south_green = db.Column(db.Integer, default=30)
    east_green = db.Column(db.Integer, default=25)
    west_green = db.Column(db.Integer, default=25)
    status = db.Column(db.String(20), default="active")
    emergency_lane = db.Column(db.String(10))
    last_optimized = db.Column(db.DateTime, default=datetime.utcnow)
    records = db.relationship("TrafficRecord", backref="intersection", lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timings": {
                "north": self.north_green,
                "south": self.south_green,
                "east": self.east_green,
                "west": self.west_green,
            },
            "status": self.status,
            "emergency_lane": self.emergency_lane,
            "last_optimized": self.last_optimized.isoformat(),
        }


class TrafficRecord(db.Model):
    __tablename__ = "traffic_records"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    intersection_id = db.Column(db.Integer, db.ForeignKey("traffic_signals.id"), nullable=False)
    lane = db.Column(db.String(10))
    vehicle_type = db.Column(db.String(20))
    vehicle_count = db.Column(db.Integer, default=0)
    density_class = db.Column(db.String(10))
    congestion_score = db.Column(db.Float, default=0.0)
    ambulance_detected = db.Column(db.Boolean, default=False)
    frame_id = db.Column(db.String(64))
    confidence = db.Column(db.Float)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "intersection_id": self.intersection_id,
            "lane": self.lane,
            "vehicle_type": self.vehicle_type,
            "vehicle_count": self.vehicle_count,
            "density_class": self.density_class,
            "congestion_score": round(self.congestion_score, 3),
            "ambulance_detected": self.ambulance_detected,
            "confidence": self.confidence,
        }


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    model_name = db.Column(db.String(40))
    input_features = db.Column(db.Text)
    predicted_class = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    actual_class = db.Column(db.String(20))
    intersection_id = db.Column(db.Integer)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
        }