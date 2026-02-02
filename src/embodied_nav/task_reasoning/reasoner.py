"""Task reasoning using LLM for navigation decisions."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from ..llm_client import BaseLLMClient
from ..scene_analysis import SceneAnalysisResult
from .prompts import TASK_REASONING_SYSTEM_PROMPT, format_task_reasoning_prompt

logger = logging.getLogger(__name__)


@dataclass
class TaskReasoningResult:
    """Result of task reasoning."""
    task_understanding: str = ""
    intent: str = ""
    reasoning: str = ""
    raw_response: str = ""


class TaskReasoner:
    """Reasons about navigation tasks using LLM."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize task reasoner.

        Args:
            llm_client: LLM client for reasoning
        """
        self.llm_client = llm_client

    async def reason(
        self,
        scene_analysis: SceneAnalysisResult,
        goal: str = "Navigate forward safely",
    ) -> TaskReasoningResult:
        """Reason about the navigation task.

        Args:
            scene_analysis: Result from scene analysis
            goal: Navigation goal

        Returns:
            TaskReasoningResult with understanding, intent, and reasoning
        """
        # Format prompt with scene data
        objects = [{"label": obj.label} for obj in scene_analysis.objects]
        obstacles = [{"label": obs.label} for obs in scene_analysis.obstacles]

        prompt = format_task_reasoning_prompt(
            scene_summary=scene_analysis.summary,
            objects=objects,
            obstacles=obstacles,
            goal=goal,
        )

        # Call LLM
        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=TASK_REASONING_SYSTEM_PROMPT,
        )

        # Parse response
        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> TaskReasoningResult:
        """Parse LLM response into structured result.

        Args:
            text: Raw LLM response text

        Returns:
            Parsed TaskReasoningResult
        """
        result = TaskReasoningResult(raw_response=text)

        try:
            # Extract JSON from response
            json_str = self._extract_json(text)
            data = json.loads(json_str)

            result.task_understanding = data.get("task_understanding", "")
            result.intent = data.get("intent", "")
            result.reasoning = data.get("reasoning", "")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract meaningful content from raw text
            result.task_understanding = "Failed to parse response"
            result.intent = "continue forward"
            result.reasoning = "Default action due to parse failure"

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
