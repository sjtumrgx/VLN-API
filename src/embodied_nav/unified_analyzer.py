"""Unified analyzer combining scene analysis, task reasoning, and waypoint generation."""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .llm_client import BaseLLMClient, encode_image_to_base64
from .waypoint_generation import Waypoint

logger = logging.getLogger(__name__)


@dataclass
class UnifiedAnalysisResult:
    """Combined result of scene analysis, task reasoning, and waypoint generation."""
    scene_summary: str
    task_understanding: str
    intent: str
    reasoning: str
    waypoints: List[Waypoint]
    linear_velocity: float = 0.0  # m/s
    angular_velocity: float = 0.0  # rad/s
    task_english: str = ""  # User task translated to English
    raw_response: str = ""


UNIFIED_PROMPT = """Task: {task}

Image size: {width}x{height}. Waypoint y-coordinates (FIXED): y1={y1}, y2={y2}, y3={y3}, y4={y4}, y5={y5}
Point 1 is fixed at x=0.5 (center). Provide x-coordinates for points 2-5 as ratios (0.0=left edge, 1.0=right edge).

Waypoints MUST land on walkable ground or the target object, NOT on obstacles.

Output JSON (ALL text in English only):
```json
{{
  "task_en": "user task translated to English",
  "scene": "brief scene description",
  "task": "target location and situation",
  "intent": "forward/left/right/approach",
  "waypoints": [x2_ratio, x3_ratio, x4_ratio, x5_ratio],
  "v":  "Give an estimate of linear velocity (0.0-1.0 m/s)",
  "w": "Give an estimate of angular velocity (-1.0 to 1.0 rad/s, positive=left)"
}}
```
task_en: translate the user task to English (keep original if already English)
waypoints: x-coordinates as ratios (0.0-1.0), e.g. 0.5=center, 0.3=left, 0.7=right
v: linear velocity (0.0-1.0 m/s), w: angular velocity (-1.0 to 1.0 rad/s, positive=left)"""

UNIFIED_SYSTEM_PROMPT = """Robot navigation system. Origin top-left, x right, y down. Output JSON only. All text must be in English."""


class UnifiedAnalyzer:
    """Unified analyzer for single-request scene understanding and navigation."""

    def __init__(self, llm_client: BaseLLMClient, num_waypoints: int = 5, jpeg_quality: int = 95):
        """Initialize unified analyzer.

        Args:
            llm_client: LLM client for vision analysis
            num_waypoints: Number of waypoints to generate (default 5)
            jpeg_quality: JPEG compression quality for image encoding (default 95)
        """
        self.llm_client = llm_client
        self.num_waypoints = num_waypoints
        self.jpeg_quality = jpeg_quality

    async def analyze(
        self,
        frame: np.ndarray,
        task: str,
        pad_h: int = 0,
    ) -> UnifiedAnalysisResult:
        """Analyze frame for navigation in a single API call.

        Args:
            frame: Image frame as numpy array (BGR)
            task: Navigation task/goal
            pad_h: Vertical padding from letterbox

        Returns:
            UnifiedAnalysisResult with analysis and waypoints
        """
        height, width = frame.shape[:2]

        # Calculate waypoint y-positions
        y_positions = self._calculate_vertical_positions(height, pad_h)

        # Encode image (no letterbox needed - frame is already resized)
        image_input, _ = encode_image_to_base64(
            frame, apply_letterbox=False, jpeg_quality=self.jpeg_quality
        )

        # Build prompt
        center_x = width // 2
        prompt = UNIFIED_PROMPT.format(
            task=task,
            width=width,
            height=height,
            y1=y_positions[0],
            y2=y_positions[1],
            y3=y_positions[2],
            y4=y_positions[3],
            y5=y_positions[4],
        )

        # Call LLM (single request)
        response = await self.llm_client.generate(
            prompt=prompt,
            image=image_input,
            system_prompt=UNIFIED_SYSTEM_PROMPT,
        )

        # Parse response
        return self._parse_response(response.text, y_positions, width)

    def _calculate_vertical_positions(self, height: int, pad_h: int = 0) -> List[int]:
        """Calculate vertical positions for waypoints within actual image region."""
        image_top = pad_h
        image_bottom = height - pad_h
        image_height = image_bottom - image_top

        bottom = image_bottom
        top = image_top + image_height // 2

        step = (bottom - top) / (self.num_waypoints - 1)
        return [int(bottom - i * step) for i in range(self.num_waypoints)]

    def _parse_response(
        self, text: str, y_positions: List[int], image_width: int = 640
    ) -> UnifiedAnalysisResult:
        """Parse LLM response into unified result."""
        # Default values
        scene_summary = ""
        task_understanding = ""
        intent = "continue forward"
        linear_velocity = 0.0
        angular_velocity = 0.0
        task_english = ""
        waypoints = []

        center_x = image_width // 2

        try:
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            task_english = data.get("task_en", "")
            scene_summary = data.get("scene", "")
            task_understanding = data.get("task", "")
            intent = data.get("intent", "continue forward")
            linear_velocity = float(data.get("v", 0.0))
            angular_velocity = float(data.get("w", 0.0))

            # Clamp velocities to valid range
            linear_velocity = max(0.0, min(1.0, linear_velocity))
            angular_velocity = max(-1.0, min(1.0, angular_velocity))

            # Parse waypoints (x values are now ratios 0.0-1.0)
            x_ratios = data.get("waypoints", [])

            if len(x_ratios) >= self.num_waypoints - 1:
                # x_ratios contains x2, x3, x4, x5 as ratios (4 values)
                waypoints.append(Waypoint(index=1, x=center_x, y=int(y_positions[0])))
                for i in range(self.num_waypoints - 1):
                    # Convert ratio to pixel coordinate
                    ratio = float(x_ratios[i])
                    ratio = max(0.0, min(1.0, ratio))  # Clamp to valid range
                    x_pixel = int(ratio * image_width)
                    waypoints.append(Waypoint(
                        index=i + 2,
                        x=x_pixel,
                        y=int(y_positions[i + 1]),
                    ))
            else:
                # Fallback to center
                for i in range(self.num_waypoints):
                    waypoints.append(Waypoint(
                        index=i + 1,
                        x=center_x,
                        y=int(y_positions[i]),
                    ))

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            scene_summary = "Failed to parse response"
            # Fallback waypoints
            for i in range(self.num_waypoints):
                waypoints.append(Waypoint(
                    index=i + 1,
                    x=center_x,
                    y=int(y_positions[i]),
                ))

        return UnifiedAnalysisResult(
            scene_summary=scene_summary,
            task_understanding=task_understanding,
            intent=intent,
            reasoning="",
            waypoints=waypoints,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            task_english=task_english,
            raw_response=text,
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        # Log raw response for debugging
        logger.debug(f"Raw LLM response: {text[:500]}...")

        # Try markdown code block first
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1)

        # Try to find a single JSON object (non-greedy)
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if match:
            return match.group(0)

        # Fallback: find first { to last } but be careful
        start = text.find('{')
        if start != -1:
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]

        return text
