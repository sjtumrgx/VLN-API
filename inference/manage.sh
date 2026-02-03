#!/bin/bash
# Service management script for Qwen3-VL inference server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/vllm.pid"
LOG_FILE="$LOG_DIR/vllm.log"

# Default values
MODEL="${MODEL:-8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

usage() {
    echo "Usage: $0 {start|stop|restart|status|health} [options]"
    echo ""
    echo "Commands:"
    echo "  start    Start the inference server"
    echo "  stop     Stop the inference server"
    echo "  restart  Restart the inference server"
    echo "  status   Check if server is running"
    echo "  health   Check server health and GPU status"
    echo ""
    echo "Options for start/restart:"
    echo "  --model, -m       Model to load: 8b, 8b-thinking, 30b, 30b-thinking, or 32b (default: $MODEL)"
    echo "  --port, -p        Port to bind (default: $PORT)"
    echo "  --host            Host to bind (default: $HOST)"
    echo "  --tensor-parallel-size, -tp  Number of GPUs (default: $TENSOR_PARALLEL_SIZE)"
    echo "  --max-model-len   Maximum context length (default: 32768)"
    echo "  --cpu-offload-gb  GiB of CPU memory for KV cache (default: 0)"
    echo "  --daemon, -d      Run in background mode"
    echo ""
    echo "Examples:"
    echo "  $0 start --model 8b --daemon"
    echo "  $0 stop"
    echo "  $0 status"
}

ensure_dirs() {
    mkdir -p "$LOG_DIR"
}

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    fi
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

do_start() {
    local daemon=false
    local extra_args=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model|-m)
                MODEL="$2"
                shift 2
                ;;
            --port|-p)
                PORT="$2"
                shift 2
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --tensor-parallel-size|-tp)
                TENSOR_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --daemon|-d)
                daemon=true
                shift
                ;;
            *)
                extra_args+=("$1")
                shift
                ;;
        esac
    done

    if is_running; then
        echo "Server is already running (PID: $(get_pid))"
        exit 1
    fi

    ensure_dirs

    echo "Starting Qwen3-VL inference server..."
    echo "  Model: $MODEL"
    echo "  Host: $HOST"
    echo "  Port: $PORT"
    echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    cd "$SCRIPT_DIR"

    if [ "$daemon" = true ]; then
        echo "  Mode: daemon (logging to $LOG_FILE)"
        nohup python start_server.py \
            --model "$MODEL" \
            --host "$HOST" \
            --port "$PORT" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            "${extra_args[@]}" \
            > "$LOG_FILE" 2>&1 &

        local pid=$!
        echo "$pid" > "$PID_FILE"
        echo ""
        echo "Server started in background (PID: $pid)"
        echo "View logs: tail -f $LOG_FILE"
    else
        echo "  Mode: foreground (Ctrl+C to stop)"
        echo ""
        python start_server.py \
            --model "$MODEL" \
            --host "$HOST" \
            --port "$PORT" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            "${extra_args[@]}"
    fi
}

do_stop() {
    local pid=$(get_pid)

    # Find vllm processes by various patterns (case-insensitive for VLLM:: processes)
    local vllm_pids=$(pgrep -f "vllm.*--port $PORT" 2>/dev/null || true)
    local server_pids=$(pgrep -f "start_server.py" 2>/dev/null || true)
    # Match VLLM:: worker and engine processes (these have uppercase names)
    local worker_pids=$(pgrep -f "VLLM::" 2>/dev/null || true)

    # Combine all PIDs
    local all_pids="$pid $vllm_pids $server_pids $worker_pids"
    all_pids=$(echo "$all_pids" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')

    if [ -z "$all_pids" ] || [ "$all_pids" = " " ]; then
        echo "Server is not running"
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
        return 0
    fi

    echo "Found processes to stop: $all_pids"

    # Send SIGTERM to all processes
    for p in $all_pids; do
        if kill -0 "$p" 2>/dev/null; then
            echo "Sending SIGTERM to process $p..."
            kill -TERM "$p" 2>/dev/null || true
        fi
    done

    # Wait for processes to terminate (max 15 seconds)
    echo -n "Waiting for processes to terminate"
    local count=0
    local still_running=true
    while [ "$still_running" = true ] && [ $count -lt 15 ]; do
        still_running=false
        for p in $all_pids; do
            if kill -0 "$p" 2>/dev/null; then
                still_running=true
                break
            fi
        done
        if [ "$still_running" = true ]; then
            sleep 1
            count=$((count + 1))
            echo -n "."
        fi
    done
    echo ""

    # Force kill any remaining processes
    for p in $all_pids; do
        if kill -0 "$p" 2>/dev/null; then
            echo "Force killing process $p..."
            kill -9 "$p" 2>/dev/null || true
        fi
    done

    # Final cleanup: kill any remaining VLLM processes
    local remaining=$(pgrep -f "VLLM::" 2>/dev/null || true)
    remaining="$remaining $(pgrep -f "vllm" 2>/dev/null || true)"
    remaining=$(echo "$remaining" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' ')
    if [ -n "$remaining" ] && [ "$remaining" != " " ]; then
        echo "Cleaning up remaining processes: $remaining"
        for p in $remaining; do
            kill -9 "$p" 2>/dev/null || true
        done
        sleep 1
    fi

    rm -f "$PID_FILE"
    echo "Server stopped"
}

do_restart() {
    do_stop
    sleep 2
    do_start "$@"
}

do_status() {
    if is_running; then
        local pid=$(get_pid)
        echo "Server is running (PID: $pid)"
        echo "  Port: $PORT"
        echo "  Log: $LOG_FILE"
        return 0
    else
        echo "Server is not running"
        return 1
    fi
}

do_health() {
    echo "=== Health Check ==="
    echo ""

    # Check if server is running
    if ! is_running; then
        echo "[FAIL] Server is not running"
        return 1
    fi
    echo "[OK] Server process is running (PID: $(get_pid))"

    # Check API endpoint
    echo ""
    echo "Checking API endpoint..."
    local api_url="http://localhost:$PORT/v1/models"
    if curl -s --max-time 5 "$api_url" > /dev/null 2>&1; then
        echo "[OK] API endpoint responding at $api_url"
        echo ""
        echo "Available models:"
        curl -s "$api_url" | python -c "import sys,json; data=json.load(sys.stdin); print('  ' + '\n  '.join(m['id'] for m in data.get('data', [])))" 2>/dev/null || echo "  (unable to parse response)"
    else
        echo "[FAIL] API endpoint not responding"
    fi

    # Check GPU status
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | while read line; do
        echo "  $line"
    done
}

# Main
case "${1:-}" in
    start)
        shift
        do_start "$@"
        ;;
    stop)
        do_stop
        ;;
    restart)
        shift
        do_restart "$@"
        ;;
    status)
        do_status
        ;;
    health)
        do_health
        ;;
    *)
        usage
        exit 1
        ;;
esac
