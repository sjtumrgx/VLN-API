# Embodied Navigation System

LLM-powered visual navigation system for embodied agents.

## Features

- Real-time video capture from camera or video files
- LLM-based scene analysis (objects, obstacles, traversable regions)
- Task reasoning for navigation decisions
- Waypoint generation with smooth path curves
- Visualization with color-coded waypoints (green=near, red=far)
- Supports Gemini native API and OpenAI compatible formats

## Installation

```bash
uv sync
```

## Usage

```bash
# Run with camera (default)
uv run python -m embodied_nav --api-key YOUR_API_KEY

# Run with video file
uv run python -m embodied_nav --source video.mp4 --api-key YOUR_API_KEY

# Use OpenAI compatible format
uv run python -m embodied_nav --api-format openai --api-key YOUR_API_KEY
```

## License

MIT
