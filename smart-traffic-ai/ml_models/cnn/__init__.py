# ml_models/cnn/__init__.py
from ml_models.cnn.architecture import AmbulanceCNN
from ml_models.cnn.predict import AmbulancePredictor

__all__ = ["AmbulanceCNN", "AmbulancePredictor"]
