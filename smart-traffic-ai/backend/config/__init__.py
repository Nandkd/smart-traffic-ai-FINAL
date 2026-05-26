# backend/config/__init__.py
from backend.config.settings import get_config, DevelopmentConfig, ProductionConfig, TestingConfig

__all__ = ["get_config", "DevelopmentConfig", "ProductionConfig", "TestingConfig"]
