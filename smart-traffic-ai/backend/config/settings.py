"""
backend/config/settings.py
============================
Flask configuration classes for development, testing, and production.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-not-for-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-secret-not-for-production")
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv("JWT_EXPIRES", 86400))

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024          # 50 MB
    UPLOAD_FOLDER = str(BASE_DIR / "uploads")

    # ML model paths
    YOLO_WEIGHTS = os.getenv(
        "YOLO_WEIGHTS_PATH",
        str(BASE_DIR.parent / "ml_models" / "weights" / "best.pt"),
    )
    CNN_WEIGHTS = os.getenv(
        "CNN_WEIGHTS_PATH",
        str(BASE_DIR.parent / "ml_models" / "weights" / "ambulance_cnn.pth"),
    )
    ENSEMBLE_PATH = os.getenv(
        "ENSEMBLE_PATH",
        str(BASE_DIR.parent / "ml_models" / "weights" / "congestion_ensemble.pkl"),
    )

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'traffic_system.db'}"


class TestingConfig(BaseConfig):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False


class ProductionConfig(BaseConfig):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{BASE_DIR / 'traffic_system_prod.db'}",
    )
    # In production, set a real SECRET_KEY via env var
    assert os.getenv("SECRET_KEY"), "SECRET_KEY env var must be set in production!"


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str = None) -> type:
    name = name or os.getenv("FLASK_ENV", "development")
    return CONFIG_MAP.get(name, DevelopmentConfig)
