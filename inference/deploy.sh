#!/bin/bash
# Deploy inference server to remote RTX5000 server

set -e

# Configuration
SSH_KEY="/Users/gexu/.ssh/id_mac_to_rtx5000"
SSH_HOST="eilab@192.168.1.100"
SSH_PORT="22"
REMOTE_DIR="/data1/Qwen3VL/inference"
VENV_DIR="/data1/Qwen3VL/.venv"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SSH command helper
ssh_cmd() {
    ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_HOST" "$@"
}

# SCP command helper
scp_cmd() {
    scp -i "$SSH_KEY" -P "$SSH_PORT" "$@"
}

echo "=== Qwen3-VL Inference Server Deployment ==="
echo ""
echo "Target: $SSH_HOST"
echo "Remote directory: $REMOTE_DIR"
echo ""

# Step 1: Check SSH connection
echo "[1/5] Checking SSH connection..."
if ! ssh_cmd "echo 'Connection OK'" > /dev/null 2>&1; then
    echo "ERROR: Cannot connect to server"
    exit 1
fi
echo "  OK"

# Step 2: Check and install uv if needed
echo ""
echo "[2/5] Checking uv installation..."
if ! ssh_cmd "which uv" > /dev/null 2>&1; then
    echo "  uv not found, installing..."
    ssh_cmd "curl -LsSf https://astral.sh/uv/install.sh | sh"
    # Add to PATH for current session
    ssh_cmd "echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    echo "  uv installed"
else
    echo "  uv already installed: $(ssh_cmd 'uv --version')"
fi

# Step 3: Detect CUDA version
echo ""
echo "[3/5] Detecting CUDA version..."
CUDA_VERSION=$(ssh_cmd "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
echo "  NVIDIA Driver: $CUDA_VERSION"
NVCC_VERSION=$(ssh_cmd "nvcc --version 2>/dev/null | grep 'release' | sed 's/.*release //' | sed 's/,.*//' || echo 'not found'")
echo "  NVCC: $NVCC_VERSION"

# Step 4: Transfer files
echo ""
echo "[4/5] Transferring files to server..."
ssh_cmd "mkdir -p $REMOTE_DIR"

# Copy all inference files
scp_cmd "$SCRIPT_DIR/pyproject.toml" "$SSH_HOST:$REMOTE_DIR/"
scp_cmd "$SCRIPT_DIR/config.py" "$SSH_HOST:$REMOTE_DIR/"
scp_cmd "$SCRIPT_DIR/start_server.py" "$SSH_HOST:$REMOTE_DIR/"
scp_cmd "$SCRIPT_DIR/manage.sh" "$SSH_HOST:$REMOTE_DIR/"

# Make scripts executable
ssh_cmd "chmod +x $REMOTE_DIR/manage.sh $REMOTE_DIR/start_server.py"
echo "  Files transferred"

# Step 5: Setup virtual environment and install dependencies
echo ""
echo "[5/5] Setting up Python environment..."

# Create venv if not exists
ssh_cmd "
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    cd $REMOTE_DIR

    if [ ! -d '$VENV_DIR' ]; then
        echo '  Creating virtual environment...'
        uv venv '$VENV_DIR' --python 3.10
    fi

    echo '  Installing dependencies...'
    source '$VENV_DIR/bin/activate'
    uv pip install 'vllm>=0.6.0' 'transformers>=4.45.0' 'accelerate>=0.34.0' 'qwen-vl-utils>=0.0.8'
"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To start the server, SSH to the server and run:"
echo "  ssh -i $SSH_KEY -p $SSH_PORT $SSH_HOST"
echo "  cd $REMOTE_DIR"
echo "  ./manage.sh start --model 8b --daemon"
echo ""
echo "Or run directly:"
echo "  ssh -i $SSH_KEY -p $SSH_PORT $SSH_HOST 'cd $REMOTE_DIR && ./manage.sh start --model 8b --daemon'"
