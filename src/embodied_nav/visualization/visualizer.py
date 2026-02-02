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

    def _ensure_window(self, frame: np.ndarray = None):
        """Ensure window is created and sized appropriately."""
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
            # Set initial window size to match frame
            if frame is not None:
                height, width = frame.shape[:2]
                cv2.resizeWindow(self.window_name, width, height)

    def render(
        self,
        frame: np.ndarray,
        scene_summary: str = "",
        task_understanding: str = "",
        intent: str = "",
        waypoints: Optional[List[Waypoint]] = None,
        waypoint_generator: Optional[WaypointGenerator] = None,
        linear_velocity: float = 0.0,
        angular_velocity: float = 0.0,
        pad_h: int = 0,
    ) -> np.ndarray:
        """Render visualization on frame.

        Args:
            frame: Input frame (BGR)
            scene_summary: Brief scene description
            task_understanding: Task understanding text
            intent: Navigation intent
            waypoints: List of waypoints
            waypoint_generator: Generator for smooth curve
            linear_velocity: Linear velocity in m/s
            angular_velocity: Angular velocity in rad/s
            pad_h: Vertical padding from letterbox (for aligning overlays)

        Returns:
            Annotated frame
        """
        # Make a copy to avoid modifying original
        vis_frame = frame.copy()

        # Draw text overlay
        vis_frame = self._draw_text_overlay(vis_frame, scene_summary, task_understanding, intent, pad_h)

        # Draw waypoints and curve
        if waypoints:
            vis_frame = self._draw_waypoints(vis_frame, waypoints, waypoint_generator)

        # Draw velocity info
        vis_frame = self._draw_velocity(vis_frame, linear_velocity, angular_velocity, pad_h)

        return vis_frame

    def show(self, frame: np.ndarray) -> int:
        """Display frame in window.

        Args:
            frame: Frame to display

        Returns:
            Key code pressed (or -1 if none)
        """
        self._ensure_window(frame)
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1)

    def close(self):
        """Close the visualization window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False

    def _get_scale_factor(self, frame: np.ndarray) -> float:
        """Calculate scale factor based on frame resolution."""
        height, width = frame.shape[:2]
        # Base resolution is 640x480, scale up for larger frames
        base_width = 640
        return max(1.0, width / base_width)

    def _draw_text_overlay(
        self,
        frame: np.ndarray,
        scene_summary: str,
        task_understanding: str,
        intent: str,
        pad_h: int = 0,
    ) -> np.ndarray:
        """Draw text overlay with three colored boxes in top-left corner."""
        height, width = frame.shape[:2]
        scale = self._get_scale_factor(frame)

        # Box parameters
        box_width = int(width * 0.22)
        margin = int(8 * scale)
        padding = int(6 * scale)
        corner_radius = int(8 * scale)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38 * scale
        thickness = max(1, int(scale))
        line_height = int(16 * scale)

        # Colors (BGR) - softer, more aesthetic colors
        colors = [
            ((180, 120, 60), (255, 220, 180)),    # Scene: dark blue bg, light blue text
            ((60, 140, 140), (180, 255, 255)),    # Task: dark teal bg, light cyan text
            ((60, 120, 60), (180, 255, 180)),     # Intent: dark green bg, light green text
        ]

        texts = [
            ("Scene", scene_summary),
            ("Task", task_understanding),
            ("Intent", intent),
        ]

        # Start from image top (after letterbox padding)
        current_y = pad_h + margin

        for (label, content), (bg_color, text_color) in zip(texts, colors):
            if not content:
                continue

            # Wrap text to fit box width
            wrapped_lines = self._wrap_text(content, font, font_scale, thickness, box_width - padding * 2)

            # Calculate box height based on content
            num_lines = len(wrapped_lines) + 1  # +1 for label
            box_height = padding * 2 + num_lines * line_height

            # Draw rounded rectangle background
            overlay = frame.copy()
            x1, y1 = margin, current_y
            x2, y2 = margin + box_width, current_y + box_height

            # Draw rounded rectangle
            self._draw_rounded_rect(overlay, (x1, y1), (x2, y2), bg_color, corner_radius)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Draw label
            label_y = y1 + padding + line_height - int(4 * scale)
            cv2.putText(frame, label, (x1 + padding, label_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Draw wrapped content
            for i, line in enumerate(wrapped_lines):
                text_y = label_y + (i + 1) * line_height
                cv2.putText(frame, line, (x1 + padding, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            current_y = y2 + margin

        return frame

    def _wrap_text(
        self,
        text: str,
        font: int,
        font_scale: float,
        thickness: int,
        max_width: int,
    ) -> List[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

            if text_size[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    def _draw_rounded_rect(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        radius: int,
    ):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Clamp radius
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

        # Draw rectangles for the body
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

        # Draw circles for corners
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

    def _draw_waypoints(
        self,
        frame: np.ndarray,
        waypoints: List[Waypoint],
        waypoint_generator: Optional[WaypointGenerator] = None,
    ) -> np.ndarray:
        """Draw waypoints with color gradient and smooth curve."""
        if not waypoints:
            return frame

        scale = self._get_scale_factor(frame)
        num_waypoints = len(waypoints)

        # Draw smooth curve first (so waypoints are on top)
        if waypoint_generator and num_waypoints >= 2:
            curve_points = waypoint_generator.get_smooth_curve(waypoints, num_points=100)
            self._draw_gradient_curve(frame, curve_points, scale)

        # Scaled parameters
        circle_radius = int(12 * scale)
        border_thickness = max(2, int(2 * scale))
        font_scale = 0.5 * scale
        font_thickness = max(1, int(scale))
        coord_font_scale = 0.4 * scale

        # Draw waypoint markers
        for i, wp in enumerate(waypoints):
            # Calculate color gradient (near=green, far=red)
            t = i / max(num_waypoints - 1, 1)
            color = self._interpolate_color(self.near_color, self.far_color, t)

            # Draw waypoint circle
            cv2.circle(frame, (wp.x, wp.y), circle_radius, color, -1)
            cv2.circle(frame, (wp.x, wp.y), circle_radius, (255, 255, 255), border_thickness)

            # Draw index number (centered in circle) - black with white outline for visibility
            text = str(wp.index)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = wp.x - text_size[0] // 2
            text_y = wp.y + text_size[1] // 2
            # Draw black outline
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness + 2,
                cv2.LINE_AA,
            )
            # Draw white text on top
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

            # Draw coordinates if enabled
            if self.show_coordinates:
                coord_text = f"({wp.x}, {wp.y})"
                cv2.putText(
                    frame,
                    coord_text,
                    (wp.x + circle_radius + 5, wp.y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    coord_font_scale,
                    color,
                    font_thickness,
                    cv2.LINE_AA,
                )

        return frame

    def _draw_gradient_curve(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        scale: float = 1.0,
    ):
        """Draw curve with color gradient."""
        if len(points) < 2:
            return

        line_thickness = max(3, int(3 * scale))
        num_points = len(points)
        for i in range(num_points - 1):
            t = i / max(num_points - 1, 1)
            color = self._interpolate_color(self.near_color, self.far_color, t)

            pt1 = points[i]
            pt2 = points[i + 1]
            cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)

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

    def _draw_velocity(
        self,
        frame: np.ndarray,
        linear_velocity: float,
        angular_velocity: float,
        pad_h: int = 0,
    ) -> np.ndarray:
        """Draw velocity info at bottom-right corner."""
        height, width = frame.shape[:2]
        scale = self._get_scale_factor(frame)

        # Create semi-transparent overlay for velocity background
        overlay = frame.copy()

        # Box dimensions
        box_width = int(120 * scale)
        box_height = int(50 * scale)
        margin = int(10 * scale)

        # Bottom-right corner position (above letterbox padding)
        box_x1 = width - box_width - margin
        box_y1 = height - pad_h - box_height - margin
        box_x2 = width - margin
        box_y2 = height - pad_h - margin

        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw velocity text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 * scale
        thickness = max(1, int(scale))

        # Linear velocity (cyan)
        v_text = f"V: {linear_velocity:.2f} m/s"
        v_y = box_y1 + int(20 * scale)
        cv2.putText(frame, v_text, (box_x1 + int(5 * scale), v_y), font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA)

        # Angular velocity (magenta)
        w_text = f"W: {angular_velocity:.2f} r/s"
        w_y = box_y1 + int(40 * scale)
        cv2.putText(frame, w_text, (box_x1 + int(5 * scale), w_y), font, font_scale, (255, 0, 255), thickness, cv2.LINE_AA)

        return frame
