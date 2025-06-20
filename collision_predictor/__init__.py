"""Collision Predictor package entry point."""
from .model import CollisionPredictor, ConvLSTMClassifier  # noqa: F401
__all__ = [
    "CollisionPredictor",
    "ConvLSTMClassifier",
]