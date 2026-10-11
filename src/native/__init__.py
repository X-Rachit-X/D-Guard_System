"""Native mode: the Kaggle-trained MILK10k evidential model, served for one case at a time."""
from src.native.engine import NativeEngine, NativeResult

__all__ = ["NativeEngine", "NativeResult"]
