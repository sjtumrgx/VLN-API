#!/usr/bin/env python3
"""Start Qwen3-VL inference server using vLLM."""

import argparse
import os
import subprocess
import sys

from config import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    GPU_MEMORY_UTILIZATION,
    MODELS,
)


def get_gpu_count() -> int:
    """Auto-detect available GPU count."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return len(result.stdout.strip().split("\n"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 1


def main():
    parser = argparse.ArgumentParser(description="Start Qwen3-VL inference server")
    parser.add_argument(
        "--model",
        "-m",
        choices=["8b", "30b"],
        default=os.environ.get("MODEL", DEFAULT_MODEL),
        help=f"Model to load: 8b or 30b (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help=f"Number of GPUs for tensor parallelism (default: auto-detect, max {DEFAULT_TENSOR_PARALLEL_SIZE})",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", DEFAULT_HOST),
        help=f"Host to bind (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.environ.get("PORT", DEFAULT_PORT)),
        help=f"Port to bind (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEMORY_UTILIZATION,
        help=f"GPU memory utilization ratio (default: {GPU_MEMORY_UTILIZATION})",
    )

    args = parser.parse_args()

    # Get model config
    model_config = MODELS.get(args.model)
    if not model_config:
        print(f"Error: Unknown model '{args.model}'", file=sys.stderr)
        sys.exit(1)

    model_path = model_config["path"]
    model_name = model_config["name"]

    # Determine tensor parallel size
    if args.tensor_parallel_size is None:
        tp_size = min(get_gpu_count(), DEFAULT_TENSOR_PARALLEL_SIZE)
    else:
        tp_size = args.tensor_parallel_size

    print(f"Starting {model_name}...")
    print(f"  Model path: {model_path}")
    print(f"  Tensor parallel size: {tp_size}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print()

    # Build vLLM serve command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--tensor-parallel-size",
        str(tp_size),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--served-model-name",
        model_name,
        "--max-num-seqs",
        "64",  # Reduce from default 256 to avoid OOM during warmup
    ]

    # Execute vLLM server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Server exited with error: {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
