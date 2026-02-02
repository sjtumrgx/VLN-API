"""Unit tests for scene analysis module."""

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from embodied_nav.scene_analysis.analyzer import (
    BoundingBox,
    DetectedObject,
    Obstacle,
    SceneAnalysisResult,
    SceneAnalyzer,
    TraversableRegion,
)
from embodied_nav.scene_analysis.schema import validate_scene_analysis


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_creation(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50


class TestSceneAnalysisResult:
    """Tests for SceneAnalysisResult class."""

    def test_default_values(self):
        """Test default values."""
        result = SceneAnalysisResult()
        assert result.objects == []
        assert result.obstacles == []
        assert result.traversable_regions == []
        assert result.summary == ""


class TestSceneAnalyzer:
    """Tests for SceneAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        mock_client = MagicMock()
        analyzer = SceneAnalyzer(mock_client)
        assert analyzer.llm_client == mock_client

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        mock_client = MagicMock()
        analyzer = SceneAnalyzer(mock_client)

        text = '''Here is the analysis:
```json
{"objects": [], "obstacles": [], "traversable_regions": [], "summary": "test"}
```
'''
        result = analyzer._extract_json(text)
        data = json.loads(result)
        assert data["summary"] == "test"

    def test_extract_json_raw(self):
        """Test extracting raw JSON."""
        mock_client = MagicMock()
        analyzer = SceneAnalyzer(mock_client)

        text = '{"objects": [], "obstacles": [], "traversable_regions": [], "summary": "raw"}'
        result = analyzer._extract_json(text)
        data = json.loads(result)
        assert data["summary"] == "raw"

    def test_parse_bbox(self):
        """Test parsing bounding box."""
        mock_client = MagicMock()
        analyzer = SceneAnalyzer(mock_client)

        bbox_data = {"x": 10, "y": 20, "width": 100, "height": 50}
        bbox = analyzer._parse_bbox(bbox_data)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_parse_response_complete(self):
        """Test parsing a complete response."""
        mock_client = MagicMock()
        analyzer = SceneAnalyzer(mock_client)

        response_text = json.dumps({
            "objects": [
                {"label": "chair", "bbox": {"x": 100, "y": 200, "width": 50, "height": 80}, "confidence": 0.95}
            ],
            "obstacles": [
                {"label": "table", "bbox": {"x": 150, "y": 100, "width": 200, "height": 100}}
            ],
            "traversable_regions": [
                {"description": "floor", "bbox": {"x": 0, "y": 300, "width": 640, "height": 180}}
            ],
            "summary": "Indoor scene with furniture"
        })

        result = analyzer._parse_response(response_text)

        assert len(result.objects) == 1
        assert result.objects[0].label == "chair"
        assert result.objects[0].confidence == 0.95

        assert len(result.obstacles) == 1
        assert result.obstacles[0].label == "table"

        assert len(result.traversable_regions) == 1
        assert result.traversable_regions[0].description == "floor"

        assert result.summary == "Indoor scene with furniture"

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        mock_client = MagicMock()
        analyzer = SceneAnalyzer(mock_client)

        result = analyzer._parse_response("not valid json")

        assert result.summary == "Failed to parse scene analysis"
        assert result.objects == []

    @pytest.mark.asyncio
    async def test_analyze(self):
        """Test full analysis flow."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "objects": [],
            "obstacles": [],
            "traversable_regions": [],
            "summary": "Empty scene"
        })
        mock_client.generate.return_value = mock_response

        analyzer = SceneAnalyzer(mock_client)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = await analyzer.analyze(frame)

        assert result.summary == "Empty scene"
        mock_client.generate.assert_called_once()


class TestSchemaValidation:
    """Tests for JSON schema validation."""

    def test_valid_complete_data(self):
        """Test validation of complete valid data."""
        data = {
            "objects": [
                {"label": "chair", "bbox": {"x": 0, "y": 0, "width": 100, "height": 100}, "confidence": 0.9}
            ],
            "obstacles": [
                {"label": "wall", "bbox": {"x": 0, "y": 0, "width": 50, "height": 200}}
            ],
            "traversable_regions": [
                {"description": "floor", "bbox": {"x": 0, "y": 300, "width": 640, "height": 180}}
            ],
            "summary": "Test scene"
        }

        is_valid, errors = validate_scene_analysis(data)
        assert is_valid is True
        assert errors == []

    def test_missing_required_field(self):
        """Test validation with missing required field."""
        data = {
            "objects": [],
            "obstacles": [],
            # missing traversable_regions and summary
        }

        is_valid, errors = validate_scene_analysis(data)
        assert is_valid is False
        assert any("traversable_regions" in e for e in errors)
        assert any("summary" in e for e in errors)

    def test_invalid_bbox(self):
        """Test validation with invalid bounding box."""
        data = {
            "objects": [
                {"label": "chair", "bbox": {"x": 0, "y": 0}}  # missing width/height
            ],
            "obstacles": [],
            "traversable_regions": [],
            "summary": "Test"
        }

        is_valid, errors = validate_scene_analysis(data)
        assert is_valid is False
        assert any("width" in e for e in errors)

    def test_invalid_confidence(self):
        """Test validation with invalid confidence value."""
        data = {
            "objects": [
                {"label": "chair", "bbox": {"x": 0, "y": 0, "width": 100, "height": 100}, "confidence": 1.5}
            ],
            "obstacles": [],
            "traversable_regions": [],
            "summary": "Test"
        }

        is_valid, errors = validate_scene_analysis(data)
        assert is_valid is False
        assert any("confidence" in e for e in errors)

    def test_empty_arrays_valid(self):
        """Test that empty arrays are valid."""
        data = {
            "objects": [],
            "obstacles": [],
            "traversable_regions": [],
            "summary": ""
        }

        is_valid, errors = validate_scene_analysis(data)
        assert is_valid is True
