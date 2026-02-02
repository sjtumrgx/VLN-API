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
    raw_response: str = ""


UNIFIED_PROMPT = """Analyze this image for robot navigation.

**Task:** {task}

**Coordinate system:** Origin (0,0) at TOP-LEFT. X increases RIGHT, Y increases DOWN.

**Waypoint positions (y-coordinates are FIXED):**
- Point 1: x=320 (FIXED at center), y={y1} (nearest)
- Point 2: y={y2}
- Point 3: y={y3}
- Point 4: y={y4}
- Point 5: y={y5} (farthest)

Analyze the scene and provide:
1. Brief scene description (what's relevant to the task)
2. Task understanding (where is the target, current situation)
3. Navigation intent (what action to take)
4. x-coordinates for waypoints 2-5 to guide the robot

**CRITICAL:** The 4 waypoints (x2-x5) represent the robot's future footsteps!
- They MUST land on the GROUND (floor, road, walkable surface) or the FINAL TARGET object
- They MUST NOT land on walls, furniture, obstacles, or any irrelevant objects
- Think of them as where the robot will actually step/move to

Output JSON only:
```json
{{
  "scene": "Brief scene description relevant to task",
  "task_understanding": "Target location and current situation",
  "intent": "Action: move forward / turn left / turn right / approach target",
  "reasoning": "Why this action",
  "waypoints": [x2, x3, x4, x5]
}}
```

- waypoints: x-coordinates (0-640) for points 2-5 only (point 1 is fixed at x=320)
- Guide path toward task target on walkable ground
- Keep response concise."""

UNIFIED_SYSTEM_PROMPT = """You are a robot navigation system.
Analyze the image, understand the task, and output navigation waypoints.
Coordinate: origin top-left, x right, y down. Image is 640x640.
Output valid JSON only."""


class UnifiedAnalyzer:
    """Unified analyzer for single-request scene understanding and navigation."""

    def __init__(self, llm_client: BaseLLMClient, num_waypoints: int = 5):
        """Initialize unified analyzer.

        Args:
            llm_client: LLM client for vision analysis
            num_waypoints: Number of waypoints to generate (default 5)
        """
        self.llm_client = llm_client
        self.num_waypoints = num_waypoints

    async def analyze(
        self,
        frame: np.ndarray,
        task: str,
        pad_h: int = 0,
    ) -> UnifiedAnalysisResult:
        """Analyze frame for navigation in a single API call.

        Args:
            frame: Image frame as numpy array (BGR), expected to be 640x640
            task: Navigation task/goal
            pad_h: Vertical padding from letterbox

        Returns:
            UnifiedAnalysisResult with analysis and waypoints
        """
        # Calculate waypoint y-positions
        y_positions = self._calculate_vertical_positions(640, pad_h)

        # Encode image (no letterbox needed - frame is already 640x640)
        image_input, _ = encode_image_to_base64(frame, apply_letterbox=False)

        # Build prompt
        prompt = UNIFIED_PROMPT.format(
            task=task,
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
        return self._parse_response(response.text, y_positions)

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
        self, text: str, y_positions: List[int]
    ) -> UnifiedAnalysisResult:
        """Parse LLM response into unified result."""
        # Default values
        scene_summary = ""
        task_understanding = ""
        intent = "continue forward"
        reasoning = ""
        waypoints = []

        try:
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            scene_summary = data.get("scene", "")
            task_understanding = data.get("task_understanding", "")
            intent = data.get("intent", "continue forward")
            reasoning = data.get("reasoning", "")

            # Parse waypoints
            x_positions = data.get("waypoints", [])

            # Waypoint 1 is always at center (x=320)
            center_x = 320

            if len(x_positions) >= self.num_waypoints - 1:
                # x_positions contains x2, x3, x4, x5 (4 values)
                waypoints.append(Waypoint(index=1, x=center_x, y=int(y_positions[0])))
                for i in range(self.num_waypoints - 1):
                    waypoints.append(Waypoint(
                        index=i + 2,
                        x=max(0, min(640, int(x_positions[i]))),
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
                    x=320,
                    y=int(y_positions[i]),
                ))

        return UnifiedAnalysisResult(
            scene_summary=scene_summary,
            task_understanding=task_understanding,
            intent=intent,
            reasoning=reasoning,
            waypoints=waypoints,
            raw_response=text,
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return text
