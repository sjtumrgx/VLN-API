"""Unit tests for task reasoning module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from embodied_nav.scene_analysis import BoundingBox, DetectedObject, Obstacle, SceneAnalysisResult
from embodied_nav.task_reasoning.prompts import format_task_reasoning_prompt
from embodied_nav.task_reasoning.reasoner import TaskReasoner, TaskReasoningResult


class TestTaskReasoningResult:
    """Tests for TaskReasoningResult class."""

    def test_default_values(self):
        """Test default values."""
        result = TaskReasoningResult()
        assert result.task_understanding == ""
        assert result.intent == ""
        assert result.reasoning == ""
        assert result.raw_response == ""


class TestFormatPrompt:
    """Tests for prompt formatting."""

    def test_format_with_objects_and_obstacles(self):
        """Test formatting with objects and obstacles."""
        prompt = format_task_reasoning_prompt(
            scene_summary="Indoor hallway",
            objects=[{"label": "door"}, {"label": "chair"}],
            obstacles=[{"label": "box"}],
            goal="Reach the door",
        )

        assert "Indoor hallway" in prompt
        assert "door" in prompt
        assert "chair" in prompt
        assert "box" in prompt
        assert "Reach the door" in prompt

    def test_format_with_empty_lists(self):
        """Test formatting with empty lists."""
        prompt = format_task_reasoning_prompt(
            scene_summary="Empty room",
            objects=[],
            obstacles=[],
            goal="Explore",
        )

        assert "Empty room" in prompt
        assert "none" in prompt
        assert "Explore" in prompt

    def test_format_with_string_objects(self):
        """Test formatting with string objects."""
        prompt = format_task_reasoning_prompt(
            scene_summary="Test",
            objects=["chair", "table"],
            obstacles=["wall"],
            goal="Navigate",
        )

        assert "chair" in prompt
        assert "table" in prompt
        assert "wall" in prompt


class TestTaskReasoner:
    """Tests for TaskReasoner class."""

    def test_init(self):
        """Test reasoner initialization."""
        mock_client = MagicMock()
        reasoner = TaskReasoner(mock_client)
        assert reasoner.llm_client == mock_client

    def test_extract_json_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        mock_client = MagicMock()
        reasoner = TaskReasoner(mock_client)

        text = '''Analysis:
```json
{"task_understanding": "test", "intent": "forward", "reasoning": "clear path"}
```
'''
        result = reasoner._extract_json(text)
        data = json.loads(result)
        assert data["intent"] == "forward"

    def test_extract_json_raw(self):
        """Test extracting raw JSON."""
        mock_client = MagicMock()
        reasoner = TaskReasoner(mock_client)

        text = '{"task_understanding": "test", "intent": "left", "reasoning": "obstacle"}'
        result = reasoner._extract_json(text)
        data = json.loads(result)
        assert data["intent"] == "left"

    def test_parse_response_complete(self):
        """Test parsing a complete response."""
        mock_client = MagicMock()
        reasoner = TaskReasoner(mock_client)

        response_text = json.dumps({
            "task_understanding": "Hallway with door ahead",
            "intent": "move forward",
            "reasoning": "Path is clear to the door"
        })

        result = reasoner._parse_response(response_text)

        assert result.task_understanding == "Hallway with door ahead"
        assert result.intent == "move forward"
        assert result.reasoning == "Path is clear to the door"

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        mock_client = MagicMock()
        reasoner = TaskReasoner(mock_client)

        result = reasoner._parse_response("not valid json")

        assert result.task_understanding == "Failed to parse response"
        assert result.intent == "continue forward"

    @pytest.mark.asyncio
    async def test_reason(self):
        """Test full reasoning flow."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "task_understanding": "Clear hallway",
            "intent": "move forward",
            "reasoning": "No obstacles detected"
        })
        mock_client.generate.return_value = mock_response

        reasoner = TaskReasoner(mock_client)

        # Create scene analysis result
        scene = SceneAnalysisResult(
            objects=[DetectedObject(
                label="door",
                bbox=BoundingBox(x=100, y=100, width=50, height=100),
            )],
            obstacles=[],
            summary="Hallway with door",
        )

        result = await reasoner.reason(scene, goal="Reach the door")

        assert result.intent == "move forward"
        mock_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_reason_with_obstacles(self):
        """Test reasoning with obstacles."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "task_understanding": "Obstacle blocking path",
            "intent": "turn left",
            "reasoning": "Box blocking direct path"
        })
        mock_client.generate.return_value = mock_response

        reasoner = TaskReasoner(mock_client)

        scene = SceneAnalysisResult(
            objects=[],
            obstacles=[Obstacle(
                label="box",
                bbox=BoundingBox(x=300, y=200, width=100, height=100),
            )],
            summary="Path blocked by box",
        )

        result = await reasoner.reason(scene, goal="Move forward")

        assert result.intent == "turn left"
        assert "box" in result.reasoning.lower() or "blocking" in result.reasoning.lower()
