"""Prompt templates for scene analysis."""

SCENE_ANALYSIS_PROMPT = """Analyze this 640x640 image for robot navigation. Identify:

1. **Objects**: Key objects in the scene (furniture, doors, people, etc.)
2. **Obstacles**: Things that block movement
3. **Traversable areas**: Safe paths for navigation

**Coordinate system**: Origin (0,0) is at the TOP-LEFT corner. X increases to the RIGHT, Y increases DOWNWARD.

Output JSON format:
```json
{{
  "objects": [
    {{"label": "object name", "bbox": {{"x": 0, "y": 0, "width": 100, "height": 100}}, "confidence": 0.9}}
  ],
  "obstacles": [
    {{"label": "obstacle name", "bbox": {{"x": 0, "y": 0, "width": 100, "height": 100}}}}
  ],
  "traversable_regions": [
    {{"description": "floor area", "bbox": {{"x": 0, "y": 0, "width": 100, "height": 100}}}}
  ],
  "summary": "Brief 1-2 sentence scene description"
}}
```

Be concise. Coordinates are in pixels (0-640 range). Only include clearly visible items."""

SCENE_ANALYSIS_SYSTEM_PROMPT = """You are a vision system for robot navigation.
Analyze images and output structured JSON for path planning.
Coordinate system: origin at top-left, x increases rightward, y increases downward.
Focus on navigation-relevant information only.
Always respond with valid JSON."""
