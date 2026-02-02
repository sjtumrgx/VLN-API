"""Visualization for navigation system."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..waypoint_generation import Waypoint, WaypointGenerator

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualizes navigation analysis results."""

    def __init__(
        self,
        window_name: str = "Embodied Navigation",
        show_coordinates: bool = True,
        near_color: Tuple[int, int, int] = (0, 255, 0),  # Green (BGR)
        far_color: Tuple[int, int, int] = (0, 0, 255),   # Red (BGR)
    ):
        """Initialize visualizer.

        Args:
            window_name: Name of the OpenCV window
            show_coordinates: Whether to show waypoint coordinates
            near_color: Color for near waypoints (BGR)
            far_color: Color for far waypoints (BGR)
        """
        self.window_name = window_name
        self.show_coordinates = show_coordinates
        self.near_color = near_color
        self.far_color = far_color
        self._window_created = False

    def _ensure_window(self):
        """Ensure window is created."""
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

    def render(
        self,
        frame: np.ndarray,
        scene_summary: str = "",
        task_understanding: str = "",
        intent: str = "",
        waypoints: Optional[List[Waypoint]] = None,
        waypoint_generator: Optional[WaypointGenerator] = None,
    ) -> np.ndarray:
        """Render visualization on frame.

        Args:
            frame: Input frame (BGR)
            scene_summary: Brief scene description
            task_understanding: Task understanding text
            intent: Navigation intent
            waypoints: List of waypoints
            waypoint_generator: Generator for smooth curve

        Returns:
            Annotated frame
        """
        # Make a copy to avoid modifying original
        vis_frame = frame.copy()

        # Draw text overlay
        vis_frame = self._draw_text_overlay(vis_frame, scene_summary, task_understanding, intent)

        # Draw waypoints and curve
        if waypoints:
            vis_frame = self._draw_waypoints(vis_frame, waypoints, waypoint_generator)

        return vis_frame

    def show(self, frame: np.ndarray) -> int:
        """Display frame in window.

        Args:
            frame: Frame to display

        Returns:
            Key code pressed (or -1 if none)
        """
        self._ensure_window()
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1)

    def close(self):
        """Close the visualization window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

    def _draw_text_overlay(
        self,
        frame: np.ndarray,
        scene_summary: str,
        task_understanding: str,
        intent: str,
    ) -> np.ndarray:
        """Draw text overlay with scene and task info."""
        height, width = frame.shape[:2]

        # Create semi-transparent overlay for text background
        overlay = frame.copy()

        # Draw background box at top
        box_height = 80
        cv2.rectangle(overlay, (0, 0), (width, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color = (255, 255, 255)

        # Scene summary
        if scene_summary:
            text = f"Scene: {scene_summary[:70]}"
            cv2.putText(frame, text, (10, 22), font, font_scale, color, 1)

        # Task understanding
        if task_understanding:
            text = f"Task: {task_understanding[:70]}"
            cv2.putText(frame, text, (10, 47), font, font_scale, (0, 255, 255), 1)

        # Intent
        if intent:
            text = f"Intent: {intent[:70]}"
            cv2.putText(frame, text, (10, 72), font, font_scale, (0, 255, 0), 1)

        return frame

    def _draw_waypoints(
        self,
        frame: np.ndarray,
        waypoints: List[Waypoint],
        waypoint_generator: Optional[WaypointGenerator] = None,
    ) -> np.ndarray:
        """Draw waypoints with color gradient and smooth curve."""
        if not waypoints:
            return frame

        num_waypoints = len(waypoints)

        # Draw smooth curve first (so waypoints are on top)
        if waypoint_generator and num_waypoints >= 2:
            curve_points = waypoint_generator.get_smooth_curve(waypoints, num_points=100)
            self._draw_gradient_curve(frame, curve_points)

        # Draw waypoint markers
        for i, wp in enumerate(waypoints):
            # Calculate color gradient (near=green, far=red)
            t = i / max(num_waypoints - 1, 1)
            color = self._interpolate_color(self.near_color, self.far_color, t)

            # Draw waypoint circle
            cv2.circle(frame, (wp.x, wp.y), 12, color, -1)
            cv2.circle(frame, (wp.x, wp.y), 12, (255, 255, 255), 2)

            # Draw index number
            cv2.putText(
                frame,
                str(wp.index),
                (wp.x - 5, wp.y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Draw coordinates if enabled
            if self.show_coordinates:
                coord_text = f"({wp.x}, {wp.y})"
                cv2.putText(
                    frame,
                    coord_text,
                    (wp.x + 15, wp.y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        return frame

    def _draw_gradient_curve(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
    ):
        """Draw curve with color gradient."""
        if len(points) < 2:
            return

        num_points = len(points)
        for i in range(num_points - 1):
            t = i / max(num_points - 1, 1)
            color = self._interpolate_color(self.near_color, self.far_color, t)

            pt1 = points[i]
            pt2 = points[i + 1]
            cv2.line(frame, pt1, pt2, color, 3)

    def _interpolate_color(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        t: float,
    ) -> Tuple[int, int, int]:
        """Interpolate between two colors using HSV for smooth gradient."""
        # Convert BGR to HSV
        c1 = np.uint8([[color1]])
        c2 = np.uint8([[color2]])
        hsv1 = cv2.cvtColor(c1, cv2.COLOR_BGR2HSV)[0][0].astype(np.int32)
        hsv2 = cv2.cvtColor(c2, cv2.COLOR_BGR2HSV)[0][0].astype(np.int32)

        # Interpolate in HSV space
        h = int(hsv1[0] + t * (hsv2[0] - hsv1[0]))
        s = int(hsv1[1] + t * (hsv2[1] - hsv1[1]))
        v = int(hsv1[2] + t * (hsv2[2] - hsv1[2]))

        # Clamp values to valid range
        h = max(0, min(179, h))  # OpenCV HSV hue is 0-179
        s = max(0, min(255, s))
        v = max(0, min(255, v))

        # Convert back to BGR
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]

        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def log_waypoints(self, waypoints: List[Waypoint]):
        """Log waypoint coordinates to console."""
        logger.info("Waypoint coordinates:")
        for wp in waypoints:
            logger.info(f"  Point {wp.index}: ({wp.x}, {wp.y})")

    def get_waypoint_string(self, waypoints: List[Waypoint]) -> str:
        """Get formatted string of waypoint coordinates."""
        lines = ["Waypoints:"]
        for wp in waypoints:
            lines.append(f"  Point {wp.index}: ({wp.x}, {wp.y})")
        return "\n".join(lines)
