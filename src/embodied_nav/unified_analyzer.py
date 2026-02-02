"""Unified analyzer combining scene analysis, task reasoning, and waypoint generation."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .llm_client import BaseLLMClient, encode_image_to_base64
from .scene_analysis import BoundingBox, DetectedObject, Obstacle, TraversableRegion, SceneAnalysisResult
from .task_reasoning import TaskReasoningResult
from .waypoint_generation import Waypoint, WaypointGenerationResult

logger = logging.getLogger(__name__)


@dataclass
class UnifiedAnalysisResult:
    """Combined result of scene analysis, task reasoning, and waypoint generation."""
    scene_analysis: SceneAnalysisResult
    task_reasoning: TaskReasoningResult
    waypoints: List[Waypoint]
    raw_response: str = ""


UNIFIED_PROMPT = """Analyze this 640x640 image for robot navigation with the following task:

**Task:** {task}

**Coordinate system:** Origin (0,0) at TOP-LEFT corner. X increases RIGHT, Y increases DOWN.

**Waypoint y-coordinates (FIXED):**
- Waypoint 1: y={y1} (bottom, nearest)
- Waypoint 2: y={y2}
- Waypoint 3: y={y3}
- Waypoint 4: y={y4}
- Waypoint 5: y={y5} (top, farthest)

**IMPORTANT:** Only detect and mark objects that are DIRECTLY RELEVANT to the task!
- If task is "find red trash bin", only mark red trash bins and obstacles blocking the path to it
- If task is "go to the door", only mark doors and obstacles in the way
- Do NOT mark irrelevant objects (furniture, decorations, etc. that don't affect the task)
- Focus attention on the TASK TARGET and PATH OBSTACLES only

Output JSON:
```json
{{
  "target_objects": [
    {{"label": "task-relevant object", "bbox": {{"x": 0, "y": 0, "width": 100, "height": 100}}, "is_target": true}}
  ],
  "obstacles": [
    {{"label": "obstacle blocking path", "bbox": {{"x": 0, "y": 0, "width": 100, "height": 100}}}}
  ],
  "traversable_path": {{
    "description": "clear path description",
    "bbox": {{"x": 0, "y": 0, "width": 100, "height": 100}}
  }},
  "scene_summary": "Brief description focusing on task target location",
  "task_understanding": "Current situation relative to task",
  "intent": "Action to take (move forward, turn left/right, approach target, etc.)",
  "reasoning": "Why this action helps complete the task",
  "waypoint_x_positions": [x2, x3, x4, x5]
}}
```

- waypoint_x_positions: x-coordinates for waypoints 2-5 (waypoint 1 is at x=320)
- x values should be 0-640, guiding robot toward the task target while avoiding obstacles
- Keep response concise. Only mark what's essential for the task."""

UNIFIED_SYSTEM_PROMPT = """You are a focused robot navigation system.
Given an image and a specific task, analyze ONLY what's relevant to completing that task.
Do NOT detect or mark objects unrelated to the task.
Coordinate system: origin at top-left, x increases rightward, y increases downward.
Output valid JSON with task-focused analysis and navigation waypoints."""


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
            UnifiedAnalysisResult with scene analysis, task reasoning, and waypoints
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
        # Initialize default results
        scene_analysis = SceneAnalysisResult(raw_response=text)
        task_reasoning = TaskReasoningResult(raw_response=text)
        waypoints = []

        try:
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            # Parse target objects (task-relevant)
            for obj in data.get("target_objects", []):
                try:
                    bbox = self._parse_bbox(obj.get("bbox", {}))
                    scene_analysis.objects.append(DetectedObject(
                        label=obj.get("label", "unknown"),
                        bbox=bbox,
                        confidence=1.0 if obj.get("is_target") else 0.8,
                    ))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse target object: {e}")

            # Parse obstacles
            for obs in data.get("obstacles", []):
                try:
                    bbox = self._parse_bbox(obs.get("bbox", {}))
                    scene_analysis.obstacles.append(Obstacle(
                        label=obs.get("label", "unknown"),
                        bbox=bbox,
                    ))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse obstacle: {e}")

            # Parse traversable path
            if "traversable_path" in data and data["traversable_path"]:
                try:
                    path = data["traversable_path"]
                    bbox = self._parse_bbox(path.get("bbox", {}))
                    scene_analysis.traversable_regions.append(TraversableRegion(
                        description=path.get("description", ""),
                        bbox=bbox,
                    ))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse traversable path: {e}")

            # Parse scene summary
            scene_analysis.summary = data.get("scene_summary", "")

            # Parse task reasoning
            task_reasoning.task_understanding = data.get("task_understanding", "")
            task_reasoning.intent = data.get("intent", "")
            task_reasoning.reasoning = data.get("reasoning", "")

            # Parse waypoints
            center_x = 320
            x_positions = [center_x]
            llm_x_positions = data.get("waypoint_x_positions", [])

            if len(llm_x_positions) >= self.num_waypoints - 1:
                x_positions.extend([
                    max(0, min(640, int(x)))
                    for x in llm_x_positions[:self.num_waypoints - 1]
                ])
            else:
                # Fallback to center
                x_positions.extend([center_x] * (self.num_waypoints - 1))

            for i in range(self.num_waypoints):
                waypoints.append(Waypoint(
                    index=i + 1,
                    x=int(x_positions[i]),
                    y=int(y_positions[i]),
                ))

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            scene_analysis.summary = "Failed to parse response"
            task_reasoning.intent = "continue forward"
            # Fallback waypoints
            for i in range(self.num_waypoints):
                waypoints.append(Waypoint(
                    index=i + 1,
                    x=320,
                    y=int(y_positions[i]),
                ))

        return UnifiedAnalysisResult(
            scene_analysis=scene_analysis,
            task_reasoning=task_reasoning,
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

    def _parse_bbox(self, bbox_data: dict) -> BoundingBox:
        """Parse bounding box from dict."""
        return BoundingBox(
            x=int(bbox_data.get("x", 0)),
            y=int(bbox_data.get("y", 0)),
            width=int(bbox_data.get("width", 0)),
            height=int(bbox_data.get("height", 0)),
        )
