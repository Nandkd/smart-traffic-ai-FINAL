# backend/app.py — CROSSROAD FINAL VERSION
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify
from flask_cors import CORS
from backend.database import db, jwt, migrate


def create_app():
    app = Flask(__name__)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    app.config["SECRET_KEY"]               = "dev-secret-key"
    app.config["JWT_SECRET_KEY"]           = "jwt-secret-key"
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 86400
    app.config["SQLALCHEMY_DATABASE_URI"]  = (
        "sqlite:///" + os.path.join(BASE_DIR, "traffic.db")
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"]       = 500 * 1024 * 1024  # 500 MB

    upload_folder = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = upload_folder

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)

    # Blueprints
    from backend.routes.auth          import auth_bp
    from backend.routes.traffic       import traffic_bp
    from backend.routes.detection     import detection_bp
    from backend.routes.prediction    import prediction_bp
    from backend.routes.analytics     import analytics_bp
    from backend.routes.signals       import signals_bp
    from backend.routes.crossroad     import crossroad_bp   # ← NEW

    app.register_blueprint(auth_bp,       url_prefix="/api/auth")
    app.register_blueprint(traffic_bp,    url_prefix="/api/traffic")
    app.register_blueprint(detection_bp,  url_prefix="/api/detect")
    app.register_blueprint(prediction_bp, url_prefix="/api/predict")
    app.register_blueprint(analytics_bp,  url_prefix="/api/analytics")
    app.register_blueprint(signals_bp,    url_prefix="/api/signals")
    app.register_blueprint(crossroad_bp,  url_prefix="/api/crossroad")  # ← NEW

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Max 500MB"}), 413

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Server error"}), 500

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "smart-traffic-ai-crossroad"})

    with app.app_context():
        db.create_all()
        seed_data()

    return app


def seed_data():
    from backend.models.user import User
    from backend.models.traffic_record import TrafficSignal

    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", email="admin@trafficai.io", role="admin")
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("Admin created: admin / admin123")

    if TrafficSignal.query.count() == 0:
        for name, lat, lng in [
            ("MG Road Junction", 28.6139, 77.2090),
            ("Connaught Place",  28.6304, 77.2177),
            ("India Gate",       28.6129, 77.2295),
            ("Rajpath Crossing", 28.6145, 77.2090),
        ]:
            db.session.add(TrafficSignal(
                location_name=name, latitude=lat, longitude=lng,
                north_green=30, south_green=30,
                east_green=25, west_green=25, status="active",
            ))
        db.session.commit()

    # No fake traffic records — charts show real data only after video uploads


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
