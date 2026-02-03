"""Configuration constants for Qwen3-VL inference server."""

# Model paths on server
MODEL_BASE_PATH = "/data1/Qwen3VL"

MODELS = {
    "8b": {
        "name": "Qwen3-VL-8B-Instruct",
        "path": f"{MODEL_BASE_PATH}/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    },
    "8b-thinking": {
        "name": "Qwen3-VL-8B-Thinking",
        "path": f"{MODEL_BASE_PATH}/models--Qwen--Qwen3-VL-8B-Thinking/snapshots/92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
    },
    "30b": {
        "name": "Qwen3-VL-30B-A3B-Instruct",
        "path": f"{MODEL_BASE_PATH}/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
    },
    "30b-thinking": {
        "name": "Qwen3-VL-30B-A3B-Thinking",
        "path": f"{MODEL_BASE_PATH}/models--Qwen--Qwen3-VL-30B-A3B-Thinking/snapshots/d0ed0380729be07a546fdefafbb4fe411f341e92",
    },
    "32b": {
        "name": "Qwen3-VL-32B-Instruct",
        "path": f"{MODEL_BASE_PATH}/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb",
    },
}

# Default server settings
DEFAULT_MODEL = "8b"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_TENSOR_PARALLEL_SIZE = 4  # Use all 4 GPUs by default

# GPU memory utilization (0.0-1.0)
GPU_MEMORY_UTILIZATION = 0.8

# Log file location
LOG_DIR = "/data1/Qwen3VL/logs"
PID_FILE = "/data1/Qwen3VL/vllm.pid"
