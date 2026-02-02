"""Scene analyzer using LLM for environment understanding."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..llm_client import BaseLLMClient, encode_image_to_base64
from .prompts import SCENE_ANALYSIS_PROMPT, SCENE_ANALYSIS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for detected objects."""
    x: int
    y: int
    width: int
    height: int


@dataclass
class DetectedObject:
    """Detected object in the scene."""
    label: str
    bbox: BoundingBox
    confidence: float = 1.0


@dataclass
class Obstacle:
    """Detected obstacle."""
    label: str
    bbox: BoundingBox


@dataclass
class TraversableRegion:
    """Traversable region in the scene."""
    description: str
    bbox: BoundingBox


@dataclass
class SceneAnalysisResult:
    """Result of scene analysis."""
    objects: List[DetectedObject] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    traversable_regions: List[TraversableRegion] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""


class SceneAnalyzer:
    """Analyzes scenes using LLM vision capabilities."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize scene analyzer.

        Args:
            llm_client: LLM client for vision analysis
        """
        self.llm_client = llm_client

    async def analyze(self, frame: np.ndarray) -> SceneAnalysisResult:
        """Analyze a frame for navigation.

        Args:
            frame: Image frame as numpy array (BGR)

        Returns:
            SceneAnalysisResult with detected objects, obstacles, and traversable regions
        """
        # Encode image
        image_input = encode_image_to_base64(frame)

        # Call LLM
        response = await self.llm_client.generate(
            prompt=SCENE_ANALYSIS_PROMPT,
            image=image_input,
            system_prompt=SCENE_ANALYSIS_SYSTEM_PROMPT,
        )

        # Parse response
        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> SceneAnalysisResult:
        """Parse LLM response into structured result.

        Args:
            text: Raw LLM response text

        Returns:
            Parsed SceneAnalysisResult
        """
        result = SceneAnalysisResult(raw_response=text)

        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            # Parse objects
            for obj in data.get("objects", []):
                try:
                    bbox = self._parse_bbox(obj.get("bbox", {}))
                    result.objects.append(DetectedObject(
                        label=obj.get("label", "unknown"),
                        bbox=bbox,
                        confidence=obj.get("confidence", 1.0),
                    ))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse object: {e}")

            # Parse obstacles
            for obs in data.get("obstacles", []):
                try:
                    bbox = self._parse_bbox(obs.get("bbox", {}))
                    result.obstacles.append(Obstacle(
                        label=obs.get("label", "unknown"),
                        bbox=bbox,
                    ))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse obstacle: {e}")

            # Parse traversable regions
            for region in data.get("traversable_regions", []):
                try:
                    bbox = self._parse_bbox(region.get("bbox", {}))
                    result.traversable_regions.append(TraversableRegion(
                        description=region.get("description", ""),
                        bbox=bbox,
                    ))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse traversable region: {e}")

            # Parse summary
            result.summary = data.get("summary", "")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            result.summary = "Failed to parse scene analysis"

        return result

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1)

        # Try to find raw JSON object
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
