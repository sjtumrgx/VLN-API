"""Scene analysis module."""

from .analyzer import (
    BoundingBox,
    DetectedObject,
    Obstacle,
    SceneAnalysisResult,
    SceneAnalyzer,
    TraversableRegion,
)
from .schema import validate_scene_analysis

__all__ = [
    "BoundingBox",
    "DetectedObject",
    "Obstacle",
    "SceneAnalysisResult",
    "SceneAnalyzer",
    "TraversableRegion",
    "validate_scene_analysis",
]
