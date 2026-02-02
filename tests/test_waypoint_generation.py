"""Unit tests for waypoint generation module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from embodied_nav.scene_analysis import SceneAnalysisResult
from embodied_nav.task_reasoning import TaskReasoningResult
from embodied_nav.waypoint_generation.generator import (
    Waypoint,
    WaypointGenerationResult,
    WaypointGenerator,
)


class TestWaypoint:
    """Tests for Waypoint class."""

    def test_creation(self):
        """Test creating a waypoint."""
        wp = Waypoint(index=1, x=320, y=480)
        assert wp.index == 1
        assert wp.x == 320
        assert wp.y == 480


class TestWaypointGenerator:
    """Tests for WaypointGenerator class."""

    def test_init_default(self):
        """Test default initialization."""
        gen = WaypointGenerator()
        assert gen.llm_client is None
        assert gen.num_waypoints == 5

    def test_init_with_client(self):
        """Test initialization with LLM client."""
        mock_client = MagicMock()
        gen = WaypointGenerator(llm_client=mock_client, num_waypoints=5)
        assert gen.llm_client == mock_client

    def test_calculate_vertical_positions(self):
        """Test vertical position calculation."""
        gen = WaypointGenerator(num_waypoints=5)
        positions = gen._calculate_vertical_positions(height=480)

        # Should have 5 positions
        assert len(positions) == 5

        # First should be at bottom (480)
        assert positions[0] == 480

        # Last should be near top (with margin)
        assert positions[-1] < 100

        # Should be in descending order (bottom to top)
        for i in range(len(positions) - 1):
            assert positions[i] > positions[i + 1]

        # Should have equal spacing
        spacing = positions[0] - positions[1]
        for i in range(len(positions) - 1):
            assert abs((positions[i] - positions[i + 1]) - spacing) < 2

    @pytest.mark.asyncio
    async def test_generate_without_llm(self):
        """Test generation without LLM client."""
        gen = WaypointGenerator(num_waypoints=5)
        result = await gen.generate(image_size=(640, 480))

        assert len(result.waypoints) == 5

        # First waypoint at bottom center
        assert result.waypoints[0].index == 1
        assert result.waypoints[0].x == 320  # center of 640
        assert result.waypoints[0].y == 480  # bottom

        # All waypoints should be at center x without LLM
        for wp in result.waypoints:
            assert wp.x == 320

    @pytest.mark.asyncio
    async def test_generate_with_llm(self):
        """Test generation with LLM client."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "waypoint_x_positions": [300, 280, 260, 240],
            "reasoning": "Avoiding obstacle on right"
        })
        mock_client.generate.return_value = mock_response

        gen = WaypointGenerator(llm_client=mock_client, num_waypoints=5)

        scene = SceneAnalysisResult(summary="Hallway with obstacle")
        task = TaskReasoningResult(intent="turn left")

        result = await gen.generate(
            image_size=(640, 480),
            scene_analysis=scene,
            task_reasoning=task,
        )

        assert len(result.waypoints) == 5
        assert result.waypoints[0].x == 320  # First always center
        assert result.waypoints[1].x == 300  # From LLM
        assert result.waypoints[2].x == 280
        assert result.waypoints[3].x == 260
        assert result.waypoints[4].x == 240

    @pytest.mark.asyncio
    async def test_generate_llm_failure_fallback(self):
        """Test fallback when LLM fails."""
        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("API error")

        gen = WaypointGenerator(llm_client=mock_client, num_waypoints=5)
        scene = SceneAnalysisResult(summary="Test")

        result = await gen.generate(
            image_size=(640, 480),
            scene_analysis=scene,
        )

        # Should fall back to center positions
        assert len(result.waypoints) == 5
        for wp in result.waypoints:
            assert wp.x == 320

    def test_get_smooth_curve(self):
        """Test smooth curve generation."""
        gen = WaypointGenerator()

        waypoints = [
            Waypoint(index=1, x=320, y=480),
            Waypoint(index=2, x=300, y=390),
            Waypoint(index=3, x=280, y=300),
            Waypoint(index=4, x=260, y=210),
            Waypoint(index=5, x=240, y=120),
        ]

        curve = gen.get_smooth_curve(waypoints, num_points=50)

        # Should have requested number of points
        assert len(curve) == 50

        # All points should be tuples of (x, y)
        for point in curve:
            assert len(point) == 2
            assert isinstance(point[0], int)
            assert isinstance(point[1], int)

    def test_get_smooth_curve_few_points(self):
        """Test smooth curve with few waypoints."""
        gen = WaypointGenerator()

        waypoints = [
            Waypoint(index=1, x=320, y=480),
            Waypoint(index=2, x=300, y=240),
        ]

        curve = gen.get_smooth_curve(waypoints, num_points=20)

        assert len(curve) > 0

    def test_get_smooth_curve_single_point(self):
        """Test smooth curve with single waypoint."""
        gen = WaypointGenerator()

        waypoints = [Waypoint(index=1, x=320, y=480)]

        curve = gen.get_smooth_curve(waypoints, num_points=10)

        assert len(curve) == 1
        assert curve[0] == (320, 480)

    def test_linear_interpolate(self):
        """Test linear interpolation fallback."""
        gen = WaypointGenerator()

        waypoints = [
            Waypoint(index=1, x=0, y=100),
            Waypoint(index=2, x=100, y=0),
        ]

        points = gen._linear_interpolate(waypoints, num_points=10)

        assert len(points) > 0
        # First point should be near first waypoint
        assert points[0][0] == 0
        assert points[0][1] == 100
        # Last point should be at last waypoint
        assert points[-1] == (100, 0)

    def test_extract_json(self):
        """Test JSON extraction."""
        gen = WaypointGenerator()

        # From code block
        text1 = '```json\n{"waypoint_x_positions": [1, 2, 3, 4]}\n```'
        result1 = gen._extract_json(text1)
        assert json.loads(result1)["waypoint_x_positions"] == [1, 2, 3, 4]

        # Raw JSON
        text2 = '{"waypoint_x_positions": [5, 6, 7, 8]}'
        result2 = gen._extract_json(text2)
        assert json.loads(result2)["waypoint_x_positions"] == [5, 6, 7, 8]
