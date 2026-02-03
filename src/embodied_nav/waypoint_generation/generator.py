"""Waypoint data structures and curve generation for visualization."""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """A navigation waypoint."""
    index: int
    x: int
    y: int


class WaypointGenerator:
    """Generates smooth curves through waypoints for visualization."""

    def __init__(self, num_waypoints: int = 5):
        """Initialize waypoint generator.

        Args:
            num_waypoints: Number of waypoints (default 5)
        """
        self.num_waypoints = num_waypoints

    def get_smooth_curve(
        self,
        waypoints: List[Waypoint],
        num_points: int = 50,
    ) -> List[Tuple[int, int]]:
        """Generate smooth curve through waypoints using spline interpolation.

        Args:
            waypoints: List of waypoints
            num_points: Number of points on the curve

        Returns:
            List of (x, y) points along the smooth curve
        """
        if len(waypoints) < 2:
            return [(w.x, w.y) for w in waypoints]

        # Extract coordinates
        x_coords = [w.x for w in waypoints]
        y_coords = [w.y for w in waypoints]

        # Use y as parameter since waypoints are ordered by y (bottom to top)
        # Reverse for interpolation (ascending y)
        y_coords_rev = y_coords[::-1]
        x_coords_rev = x_coords[::-1]

        try:
            # Create spline interpolation
            if len(waypoints) >= 4:
                # Cubic spline for 4+ points
                tck, u = interpolate.splprep([x_coords_rev, y_coords_rev], s=0, k=3)
            else:
                # Linear or quadratic for fewer points
                k = min(len(waypoints) - 1, 2)
                tck, u = interpolate.splprep([x_coords_rev, y_coords_rev], s=0, k=k)

            # Generate smooth curve
            u_new = np.linspace(0, 1, num_points)
            smooth_coords = interpolate.splev(u_new, tck)

            return [(int(x), int(y)) for x, y in zip(smooth_coords[0], smooth_coords[1])]

        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}, using linear")
            # Fallback to linear interpolation
            return self._linear_interpolate(waypoints, num_points)

    def _linear_interpolate(
        self,
        waypoints: List[Waypoint],
        num_points: int,
    ) -> List[Tuple[int, int]]:
        """Linear interpolation fallback."""
        if len(waypoints) < 2:
            return [(w.x, w.y) for w in waypoints]

        points = []
        for i in range(len(waypoints) - 1):
            w1, w2 = waypoints[i], waypoints[i + 1]
            segment_points = num_points // (len(waypoints) - 1)

            for j in range(segment_points):
                t = j / segment_points
                x = int(w1.x + t * (w2.x - w1.x))
                y = int(w1.y + t * (w2.y - w1.y))
                points.append((x, y))

        points.append((waypoints[-1].x, waypoints[-1].y))
        return points
