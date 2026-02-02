"""Unit tests for visualization module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embodied_nav.scene_analysis import BoundingBox, DetectedObject, Obstacle, SceneAnalysisResult, TraversableRegion
from embodied_nav.task_reasoning import TaskReasoningResult
from embodied_nav.waypoint_generation import Waypoint, WaypointGenerator
from embodied_nav.visualization.visualizer import Visualizer


class TestVisualizer:
    """Tests for Visualizer class."""

    def test_init_default(self):
        """Test default initialization."""
        vis = Visualizer()
        assert vis.window_name == "Embodied Navigation"
        assert vis.show_coordinates is True
        assert vis.near_color == (0, 255, 0)
        assert vis.far_color == (0, 0, 255)

    def test_init_custom(self):
        """Test custom initialization."""
        vis = Visualizer(
            window_name="Test Window",
            show_coordinates=False,
            near_color=(255, 0, 0),
            far_color=(0, 255, 0),
        )
        assert vis.window_name == "Test Window"
        assert vis.show_coordinates is False

    def test_render_empty_frame(self):
        """Test rendering with no annotations."""
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = vis.render(frame)

        assert result.shape == frame.shape
        # Should be a copy, not the same object
        assert result is not frame

    def test_render_with_scene_analysis(self):
        """Test rendering with scene analysis."""
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        scene = SceneAnalysisResult(
            objects=[DetectedObject(
                label="chair",
                bbox=BoundingBox(x=100, y=100, width=50, height=80),
            )],
            obstacles=[Obstacle(
                label="box",
                bbox=BoundingBox(x=200, y=200, width=60, height=60),
            )],
            traversable_regions=[TraversableRegion(
                description="floor",
                bbox=BoundingBox(x=0, y=300, width=640, height=180),
            )],
            summary="Test scene",
        )

        result = vis.render(frame, scene_analysis=scene)

        # Frame should be modified (not all zeros)
        assert not np.array_equal(result, frame)

    def test_render_with_task_reasoning(self):
        """Test rendering with task reasoning."""
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        task = TaskReasoningResult(
            task_understanding="Clear hallway ahead",
            intent="move forward",
            reasoning="No obstacles detected",
        )

        result = vis.render(frame, task_reasoning=task)

        # Frame should be modified
        assert not np.array_equal(result, frame)

    def test_render_with_waypoints(self):
        """Test rendering with waypoints."""
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        waypoints = [
            Waypoint(index=1, x=320, y=480),
            Waypoint(index=2, x=300, y=390),
            Waypoint(index=3, x=280, y=300),
            Waypoint(index=4, x=260, y=210),
            Waypoint(index=5, x=240, y=120),
        ]

        result = vis.render(frame, waypoints=waypoints)

        # Frame should be modified
        assert not np.array_equal(result, frame)

    def test_render_with_waypoints_and_generator(self):
        """Test rendering with waypoints and smooth curve."""
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        waypoints = [
            Waypoint(index=1, x=320, y=480),
            Waypoint(index=2, x=300, y=390),
            Waypoint(index=3, x=280, y=300),
            Waypoint(index=4, x=260, y=210),
            Waypoint(index=5, x=240, y=120),
        ]

        generator = WaypointGenerator()

        result = vis.render(frame, waypoints=waypoints, waypoint_generator=generator)

        # Frame should be modified
        assert not np.array_equal(result, frame)

    def test_interpolate_color(self):
        """Test color interpolation."""
        vis = Visualizer()

        # At t=0, should be near color (green)
        color0 = vis._interpolate_color((0, 255, 0), (0, 0, 255), 0.0)
        assert color0[1] > color0[2]  # More green than red

        # At t=1, should be far color (red)
        color1 = vis._interpolate_color((0, 255, 0), (0, 0, 255), 1.0)
        assert color1[2] > color1[1]  # More red than green

    def test_get_waypoint_string(self):
        """Test waypoint string formatting."""
        vis = Visualizer()

        waypoints = [
            Waypoint(index=1, x=320, y=480),
            Waypoint(index=2, x=300, y=390),
        ]

        result = vis.get_waypoint_string(waypoints)

        assert "Waypoints:" in result
        assert "Point 1: (320, 480)" in result
        assert "Point 2: (300, 390)" in result

    @patch("cv2.namedWindow")
    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    def test_show(self, mock_waitkey, mock_imshow, mock_namedwindow):
        """Test showing frame in window."""
        mock_waitkey.return_value = ord('q')

        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        key = vis.show(frame)

        mock_namedwindow.assert_called_once()
        mock_imshow.assert_called_once()
        assert key == ord('q')

    @patch("cv2.destroyWindow")
    def test_close(self, mock_destroy):
        """Test closing window."""
        vis = Visualizer()
        vis._window_created = True

        vis.close()

        mock_destroy.assert_called_once_with(vis.window_name)
        assert vis._window_created is False

    def test_close_not_created(self):
        """Test closing when window not created."""
        vis = Visualizer()

        # Should not raise
        vis.close()

    def test_draw_bbox(self):
        """Test bounding box drawing."""
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        bbox = BoundingBox(x=100, y=100, width=50, height=80)
        vis._draw_bbox(frame, bbox, color=(255, 0, 0), label="test")

        # Check that something was drawn
        assert np.any(frame != 0)

    def test_render_full_pipeline(self):
        """Test full rendering pipeline."""
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        scene = SceneAnalysisResult(
            objects=[DetectedObject(
                label="door",
                bbox=BoundingBox(x=300, y=50, width=100, height=200),
            )],
            obstacles=[],
            summary="Hallway with door",
        )

        task = TaskReasoningResult(
            task_understanding="Door ahead",
            intent="approach door",
            reasoning="Clear path to door",
        )

        waypoints = [
            Waypoint(index=1, x=320, y=480),
            Waypoint(index=2, x=330, y=390),
            Waypoint(index=3, x=340, y=300),
            Waypoint(index=4, x=350, y=210),
            Waypoint(index=5, x=350, y=120),
        ]

        generator = WaypointGenerator()

        result = vis.render(
            frame,
            scene_analysis=scene,
            task_reasoning=task,
            waypoints=waypoints,
            waypoint_generator=generator,
        )

        # Should have rendered everything
        assert result.shape == frame.shape
        assert not np.array_equal(result, np.zeros_like(result))
