# Embodied Navigation System

LLM-powered visual navigation system for embodied agents. The system uses vision-language models to analyze scenes, reason about navigation tasks, and generate waypoints for robot navigation.

## Features

- **Real-time Video Processing**: Support for camera devices and video files with automatic offline mode detection
- **LLM-based Scene Analysis**: Detect objects, obstacles, and traversable regions using vision-language models
- **Task Reasoning**: Intelligent navigation decision-making based on scene understanding and user-defined goals
- **Waypoint Generation**: Generate navigation waypoints with smooth spline-interpolated paths
- **Visualization**: Real-time display with color-coded waypoints (green=near, red=far), bounding boxes, and navigation intent overlay
- **Multi-API Support**: Supports both Gemini native API and OpenAI-compatible API formats
- **Profile-based Configuration**: Define multiple API configurations and switch between them via CLI
- **Self-hosted Inference**: Optional local Qwen3-VL inference server with vLLM for cost-effective deployment

## Architecture

```
src/embodied_nav/
├── main.py                 # Main application entry point
├── llm_client/             # LLM API clients
│   ├── base.py             # Base client interface
│   ├── gemini.py           # Gemini native API client
│   ├── openai_compat.py    # OpenAI-compatible API client
│   └── image_utils.py      # Image encoding utilities
├── scene_analysis/         # Scene understanding module
│   ├── analyzer.py         # LLM-based scene analyzer
│   ├── prompts.py          # Scene analysis prompts
│   └── schema.py           # Data structures
├── task_reasoning/         # Navigation reasoning module
│   ├── reasoner.py         # Task reasoning logic
│   └── prompts.py          # Reasoning prompts
├── waypoint_generation/    # Path planning module
│   └── generator.py        # Waypoint generation with spline interpolation
├── video_capture/          # Video input module
│   └── capture.py          # Camera/file capture with threading
└── visualization/          # Display module
    └── visualizer.py       # OpenCV-based visualization

inference/                  # Self-hosted VLM inference server
├── config.py               # Model paths and server configuration
├── start_server.py         # vLLM server launcher
├── manage.sh               # Service management (start/stop/status/health)
└── deploy.sh               # Deployment script for remote server
```

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Configuration

Configuration is managed via `config/config.yaml`. The system supports multiple API profiles for easy switching between different LLM providers.

### API Profiles

```yaml
api:
  # Default profile to use
  default_profile: "dev"

  # Define multiple API configurations
  profiles:
    - name: "dev"
      base_url: "http://localhost:8000"
      api_key: "your-dev-key"
      model: "gemini-2.5-flash"
      format: "gemini"  # or "openai"

    - name: "prod"
      base_url: "https://api.example.com"
      api_key: "your-prod-key"
      model: "gemini-3-pro-preview"
      format: "gemini"
```

### Video Settings

```yaml
video:
  source: 0              # Camera index or video file path
  queue_size: 1          # Frame buffer size
  reconnect_attempts: 3  # Reconnection attempts on disconnect
  reconnect_delay: 1.0   # Delay between reconnection attempts
```

### Visualization Settings

```yaml
visualization:
  window_name: "Embodied Navigation"
  show_coordinates: true
  near_color: [0, 255, 0]  # BGR - green for near waypoints
  far_color: [0, 0, 255]   # BGR - red for far waypoints
```

## Usage

### Basic Usage

```bash
# Run with default profile and camera
uv run python -m embodied_nav

# Run with a specific profile
uv run python -m embodied_nav --profile prod

# Run with video file (auto-detects offline mode)
uv run python -m embodied_nav --source video.mp4
```

### CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--config` | `-c` | Path to config file (default: `config/config.yaml`) |
| `--source` | `-s` | Video source (camera index or file path) |
| `--profile` | `-p` | API profile name to use |
| `--api-format` | `-f` | API format: `gemini` or `openai` (overrides profile) |
| `--api-key` | `-k` | API key (overrides profile) |
| `--model` | `-m` | Model name (overrides profile) |
| `--goal` | `-g` | Navigation goal (default: "Navigate forward safely") |
| `--task` | `-t` | Custom task instruction (overrides --goal) |
| `--offline` | | Force offline video processing mode |
| `--sample-interval` | `-i` | Frame sampling interval for offline mode |
| `--log-level` | `-l` | Logging level: DEBUG, INFO, WARNING, ERROR |

### Examples

```bash
# Use a specific profile with custom goal
uv run python -m embodied_nav --profile prod --goal "Find the red door"

# Process video file with custom task
uv run python -m embodied_nav --source demo.mp4 --task "Navigate to the kitchen"

# Override profile settings via CLI
uv run python -m embodied_nav --profile dev --model gemini-3-flash --api-key "new-key"

# Offline mode with custom frame sampling (every 30 frames)
uv run python -m embodied_nav --source video.mp4 --sample-interval 30

# Debug mode
uv run python -m embodied_nav --log-level DEBUG
```

## Processing Pipeline

1. **Video Capture**: Frames are captured from camera or video file in a separate thread
2. **Scene Analysis**: Each frame is sent to the LLM for scene understanding
   - Object detection with bounding boxes
   - Obstacle identification
   - Traversable region detection
3. **Task Reasoning**: Based on scene analysis and user goal, the system reasons about navigation intent
4. **Waypoint Generation**: Navigation waypoints are generated considering:
   - Scene obstacles
   - Traversable paths
   - Navigation intent
5. **Visualization**: Results are rendered with:
   - Color-coded waypoints (green→red gradient from near to far)
   - Smooth spline curve through waypoints
   - Bounding boxes for objects/obstacles
   - Navigation intent overlay

## Keyboard Controls

- `q` or `Q`: Quit the application
- Any other key (offline mode): Continue to next frame

## API Formats

### Gemini Native Format

Uses the Gemini v1beta API format:
- Endpoint: `/v1beta/models/{model}:generateContent`
- Authentication: `x-goog-api-key` header
- Supports inline image data and system instructions

### OpenAI Compatible Format

Uses the OpenAI chat completions format:
- Endpoint: `/v1/chat/completions`
- Authentication: `Authorization: Bearer {api_key}`
- Supports vision with base64-encoded images

## Dependencies

- Python >= 3.9
- opencv-python >= 4.8.0
- httpx >= 0.25.0
- numpy >= 1.24.0
- scipy >= 1.11.0
- Pillow >= 10.0.0
- matplotlib >= 3.7.0
- pyyaml >= 6.0.0

## License

MIT

## Self-hosted Inference Server

For cost-effective deployment, you can run your own Qwen3-VL inference server using vLLM. This provides an OpenAI-compatible API endpoint.

### Requirements

- NVIDIA GPU(s) with CUDA support (tested on 4x RTX 5000 Ada, 32GB each)
- Pre-downloaded Qwen3-VL model weights

### Deployment

```bash
# Deploy to remote server (from local machine)
cd inference
./deploy.sh
```

The deploy script will:
1. Install uv (if not present)
2. Create a Python virtual environment
3. Install vLLM and dependencies
4. Transfer configuration files

### Server Management

On the remote server:

```bash
cd /data1/Qwen3VL/inference

# Start server with 8B model (default)
./manage.sh start --model 8b --daemon

# Start server with 30B-A3B model
./manage.sh start --model 30b --daemon

# Check status
./manage.sh status

# Health check (API + GPU status)
./manage.sh health

# Stop server
./manage.sh stop

# Restart with different model
./manage.sh restart --model 30b
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Model to load: `8b` or `30b` | `8b` |
| `--port`, `-p` | API port | `8000` |
| `--host` | Bind address | `0.0.0.0` |
| `--tensor-parallel-size`, `-tp` | Number of GPUs | Auto-detect (4) |
| `--daemon`, `-d` | Run in background | Foreground |

### Using with Embodied Navigation

Add a profile in `config/config.yaml`:

```yaml
api:
  profiles:
    - name: "local-qwen"
      base_url: "http://192.168.1.100:8000"
      api_key: "not-needed"
      model: "Qwen3-VL-8B-Instruct"
      format: "openai"
```

Then run:

```bash
uv run python -m embodied_nav --profile local-qwen
```

### Supported Models

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| Qwen3-VL-8B-Instruct | ~8GB per GPU | Fast, good for real-time |
| Qwen3-VL-30B-A3B-Instruct | ~12GB per GPU | MoE, better quality |
