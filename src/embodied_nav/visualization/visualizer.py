"""Visualization for navigation system."""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..scene_analysis import SceneAnalysisResult, BoundingBox
from ..task_reasoning import TaskReasoningResult
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
        scene_analysis: Optional[SceneAnalysisResult] = None,
        task_reasoning: Optional[TaskReasoningResult] = None,
        waypoints: Optional[List[Waypoint]] = None,
        waypoint_generator: Optional[WaypointGenerator] = None,
    ) -> np.ndarray:
        """Render visualization on frame.

        Args:
            frame: Input frame (BGR)
            scene_analysis: Scene analysis result
            task_reasoning: Task reasoning result
            waypoints: List of waypoints
            waypoint_generator: Generator for smooth curve

        Returns:
            Annotated frame
        """
        # Make a copy to avoid modifying original
        vis_frame = frame.copy()

        # Draw scene analysis annotations
        if scene_analysis:
            vis_frame = self._draw_scene_analysis(vis_frame, scene_analysis)

        # Draw task reasoning overlay
        if task_reasoning:
            vis_frame = self._draw_task_reasoning(vis_frame, task_reasoning)

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

    def _draw_scene_analysis(
        self,
        frame: np.ndarray,
        scene_analysis: SceneAnalysisResult,
    ) -> np.ndarray:
        """Draw scene analysis annotations."""
        # Draw objects (blue boxes)
        for obj in scene_analysis.objects:
            self._draw_bbox(
                frame,
                obj.bbox,
                color=(255, 200, 0),  # Light blue
                label=obj.label,
            )

        # Draw obstacles (red boxes)
        for obs in scene_analysis.obstacles:
            self._draw_bbox(
                frame,
                obs.bbox,
                color=(0, 0, 255),  # Red
                label=f"[!] {obs.label}",
            )

        # Draw traversable regions (green boxes, semi-transparent)
        for region in scene_analysis.traversable_regions:
            self._draw_bbox(
                frame,
                region.bbox,
                color=(0, 255, 0),  # Green
                label=region.description,
                thickness=1,
            )

        return frame

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        label: str = "",
        thickness: int = 2,
    ):
        """Draw a bounding box with label."""
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # Draw label background
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)

            cv2.rectangle(
                frame,
                (x, y - text_h - 6),
                (x + text_w + 4, y),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x + 2, y - 4),
                font,
                font_scale,
                (255, 255, 255),
                1,
            )

    def _draw_task_reasoning(
        self,
        frame: np.ndarray,
        task_reasoning: TaskReasoningResult,
    ) -> np.ndarray:
        """Draw task reasoning overlay."""
        height, width = frame.shape[:2]

        # Create semi-transparent overlay for text background
        overlay = frame.copy()

        # Draw background box at top
        box_height = 80
        cv2.rectangle(overlay, (0, 0), (width, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)

        # Task understanding
        if task_reasoning.task_understanding:
            text = f"Scene: {task_reasoning.task_understanding[:60]}"
            cv2.putText(frame, text, (10, 25), font, font_scale, color, 1)

        # Intent
        if task_reasoning.intent:
            text = f"Intent: {task_reasoning.intent[:60]}"
            cv2.putText(frame, text, (10, 50), font, font_scale, (0, 255, 255), 1)

        # Reasoning
        if task_reasoning.reasoning:
            text = f"Reason: {task_reasoning.reasoning[:60]}"
            cv2.putText(frame, text, (10, 75), font, font_scale, color, 1)

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
