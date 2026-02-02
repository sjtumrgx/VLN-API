"""Prompt templates for task reasoning."""

TASK_REASONING_PROMPT = """Based on the scene analysis and navigation goal, determine the action to take.

**Scene Analysis:**
{scene_summary}

**Objects detected:** {objects}
**Obstacles:** {obstacles}

**Navigation Goal:** {goal}

Analyze the situation and output JSON:
```json
{{
  "task_understanding": "Brief description of current situation",
  "intent": "Action to take (e.g., move forward, turn left, avoid obstacle)",
  "reasoning": "Why this action is appropriate"
}}
```

Be concise. Focus on immediate navigation decisions."""

TASK_REASONING_SYSTEM_PROMPT = """You are a navigation decision system for robots.
Given scene analysis and a goal, determine the best action.
Output structured JSON with clear, actionable decisions.
Keep responses brief and focused on navigation."""


def format_task_reasoning_prompt(
    scene_summary: str,
    objects: list,
    obstacles: list,
    goal: str,
) -> str:
    """Format the task reasoning prompt with scene data.

    Args:
        scene_summary: Summary of the scene
        objects: List of detected objects
        obstacles: List of detected obstacles
        goal: Navigation goal

    Returns:
        Formatted prompt string
    """
    objects_str = ", ".join(obj if isinstance(obj, str) else obj.get("label", "unknown") for obj in objects) or "none"
    obstacles_str = ", ".join(obs if isinstance(obs, str) else obs.get("label", "unknown") for obs in obstacles) or "none"

    return TASK_REASONING_PROMPT.format(
        scene_summary=scene_summary or "No scene summary available",
        objects=objects_str,
        obstacles=obstacles_str,
        goal=goal or "Navigate forward safely",
    )
