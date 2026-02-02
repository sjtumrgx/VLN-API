"""Waypoint generation for navigation paths."""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import interpolate

from ..llm_client import BaseLLMClient
from ..scene_analysis import SceneAnalysisResult
from ..task_reasoning import TaskReasoningResult

logger = logging.getLogger(__name__)


@dataclass
class Waypoint:
    """A navigation waypoint."""
    index: int
    x: int
    y: int


@dataclass
class WaypointGenerationResult:
    """Result of waypoint generation."""
    waypoints: List[Waypoint]
    raw_response: str = ""


WAYPOINT_PROMPT = """Based on the scene analysis and navigation intent, suggest horizontal positions for 5 waypoints.

**Scene:** {scene_summary}
**Intent:** {intent}
**Image size:** {width}x{height}

The waypoints go from bottom (near) to top (far) with equal vertical spacing.
Waypoint 1 starts at bottom center (x={center_x}).

Suggest x-coordinates for waypoints 2-5 based on the scene. Consider:
- Obstacles to avoid
- Clear paths to follow
- The navigation intent

Output JSON:
```json
{{
  "waypoint_x_positions": [x2, x3, x4, x5],
  "reasoning": "Brief explanation"
}}
```

x values should be between 0 and {width}."""

WAYPOINT_SYSTEM_PROMPT = """You are a path planning system.
Given scene analysis, suggest waypoint positions for navigation.
Output valid JSON with x-coordinates."""


class WaypointGenerator:
    """Generates navigation waypoints."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        num_waypoints: int = 5,
    ):
        """Initialize waypoint generator.

        Args:
            llm_client: Optional LLM client for intelligent positioning
            num_waypoints: Number of waypoints to generate (default 5)
        """
        self.llm_client = llm_client
        self.num_waypoints = num_waypoints

    async def generate(
        self,
        image_size: Tuple[int, int],
        scene_analysis: Optional[SceneAnalysisResult] = None,
        task_reasoning: Optional[TaskReasoningResult] = None,
    ) -> WaypointGenerationResult:
        """Generate navigation waypoints.

        Args:
            image_size: (width, height) of the image
            scene_analysis: Optional scene analysis result
            task_reasoning: Optional task reasoning result

        Returns:
            WaypointGenerationResult with 5 waypoints
        """
        width, height = image_size

        # Calculate vertical positions (equal spacing from bottom to top)
        y_positions = self._calculate_vertical_positions(height)

        # First waypoint is always at bottom center
        center_x = width // 2
        x_positions = [center_x]

        # Get horizontal positions for remaining waypoints
        if self.llm_client and scene_analysis:
            llm_x_positions, raw_response = await self._get_llm_positions(
                image_size, scene_analysis, task_reasoning
            )
            if llm_x_positions:
                x_positions.extend(llm_x_positions)
            else:
                # Fallback to center
                x_positions.extend([center_x] * (self.num_waypoints - 1))
                raw_response = "LLM positioning failed, using center fallback"
        else:
            # No LLM, use center positions
            x_positions.extend([center_x] * (self.num_waypoints - 1))
            raw_response = "No LLM client, using center positions"

        # Create waypoints
        waypoints = []
        for i in range(self.num_waypoints):
            waypoints.append(Waypoint(
                index=i + 1,
                x=int(x_positions[i]),
                y=int(y_positions[i]),
            ))

        return WaypointGenerationResult(
            waypoints=waypoints,
            raw_response=raw_response,
        )

    def _calculate_vertical_positions(self, height: int) -> List[int]:
        """Calculate equal vertical spacing for waypoints.

        Args:
            height: Image height

        Returns:
            List of y-coordinates from bottom to top
        """
        # Start from bottom (height) to top
        # Leave some margin at top (10% of height)
        top_margin = int(height * 0.1)
        bottom = height
        top = top_margin

        # Equal spacing
        step = (bottom - top) / (self.num_waypoints - 1)
        return [int(bottom - i * step) for i in range(self.num_waypoints)]

    async def _get_llm_positions(
        self,
        image_size: Tuple[int, int],
        scene_analysis: SceneAnalysisResult,
        task_reasoning: Optional[TaskReasoningResult],
    ) -> Tuple[Optional[List[int]], str]:
        """Get horizontal positions from LLM.

        Returns:
            Tuple of (x_positions for waypoints 2-5, raw_response)
        """
        width, height = image_size
        center_x = width // 2

        intent = task_reasoning.intent if task_reasoning else "navigate forward"

        prompt = WAYPOINT_PROMPT.format(
            scene_summary=scene_analysis.summary,
            intent=intent,
            width=width,
            height=height,
            center_x=center_x,
        )

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=WAYPOINT_SYSTEM_PROMPT,
            )

            # Parse response
            json_str = self._extract_json(response.text)
            data = json.loads(json_str)

            x_positions = data.get("waypoint_x_positions", [])
            if len(x_positions) >= self.num_waypoints - 1:
                # Clamp values to valid range
                x_positions = [max(0, min(width, int(x))) for x in x_positions[:self.num_waypoints - 1]]
                return x_positions, response.text

        except Exception as e:
            logger.error(f"Failed to get LLM positions: {e}")

        return None, ""

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text."""
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return text

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
