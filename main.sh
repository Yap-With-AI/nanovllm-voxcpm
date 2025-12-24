#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    VoxCPM Setup and Test Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================
# Step 1: Check for NVIDIA GPU
# ============================================
echo -e "\n${YELLOW}[1/7] Checking GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. NVIDIA GPU required.${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -e "${GREEN}✓ GPU detected${NC}"

# ============================================
# Step 2: Create virtual environment
# ============================================
echo -e "\n${YELLOW}[2/7] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Created venv${NC}"
else
    echo -e "${GREEN}✓ venv already exists${NC}"
fi

source venv/bin/activate
echo -e "${GREEN}✓ Activated venv${NC}"

# ============================================
# Step 3: Install dependencies
# ============================================
echo -e "\n${YELLOW}[3/7] Installing dependencies...${NC}"

# Upgrade pip
pip install --upgrade pip -q

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Check if flash-attn is installed
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "Installing Flash Attention build dependencies..."
    pip install ninja psutil packaging -q
    echo "Installing Flash Attention (this may take several minutes - compiling CUDA kernels)..."
    pip install flash-attn --no-build-isolation
fi
echo -e "${GREEN}✓ Flash Attention installed${NC}"

# Install project dependencies
echo "Installing project dependencies..."
pip install safetensors huggingface_hub hf_transfer transformers soundfile numpy pydantic xxhash tqdm -q

# Install FastAPI dependencies
pip install fastapi uvicorn httpx -q

# Install the project itself
pip install -e . -q 2>/dev/null || true

echo -e "${GREEN}✓ All dependencies installed${NC}"

# ============================================
# Step 4: Verify PyTorch CUDA
# ============================================
echo -e "\n${YELLOW}[4/7] Verifying CUDA...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"
echo -e "${GREEN}✓ CUDA verified${NC}"

# ============================================
# Step 5: Download model from HuggingFace
# ============================================
echo -e "\n${YELLOW}[5/8] Downloading model from HuggingFace...${NC}"

python3 -c "
from huggingface_hub import snapshot_download
import os

model_id = 'openbmb/VoxCPM1.5'
print(f'Downloading {model_id}...')
path = snapshot_download(repo_id=model_id)
print(f'Model downloaded to: {path}')
"
echo -e "${GREEN}✓ Model downloaded${NC}"

# ============================================
# Step 6: Start FastAPI server
# ============================================
echo -e "\n${YELLOW}[6/8] Starting FastAPI server...${NC}"

# Kill any existing server on port 8000
pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1

# Start server in background
cd "$SCRIPT_DIR/fastapi"
uvicorn app:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
cd "$SCRIPT_DIR"

echo "Waiting for server to initialize (loading model into GPU)..."

# Wait for server to be ready (with timeout)
MAX_WAIT=300  # 5 minutes for model download
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ FastAPI server ready${NC}"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Still loading... (${WAITED}s elapsed)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}ERROR: Server failed to start within ${MAX_WAIT}s${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Step 7: Run example.py twice
# ============================================
echo -e "\n${YELLOW}[7/8] Running example.py (Run 1 - warmup)...${NC}"
echo "----------------------------------------"

python3 example.py \
    --text "Hello, this is the first test run. The model is warming up and loading into GPU memory." \
    --output output_run1.wav \
    --temperature 1.0 \
    --cfg-value 1.5

echo -e "\n${GREEN}✓ Run 1 complete - saved to output_run1.wav${NC}"

echo -e "\n${YELLOW}[7/8] Running example.py (Run 2 - actual performance)...${NC}"
echo "----------------------------------------"

python3 example.py \
    --text "This is the second test run. The model should now be fully warmed up and running at optimal speed. This demonstrates real-time streaming text to speech synthesis." \
    --output output_run2.wav \
    --temperature 1.0 \
    --cfg-value 1.5

echo -e "\n${GREEN}✓ Run 2 complete - saved to output_run2.wav${NC}"

# ============================================
# Step 8: Print summary and cleanup
# ============================================
echo -e "\n${YELLOW}[8/8] Cleanup and Summary${NC}"

# Stop the server
kill $SERVER_PID 2>/dev/null || true
echo -e "${GREEN}✓ FastAPI server stopped${NC}"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}              SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Output files:"
echo "  - output_run1.wav (warmup run)"
echo "  - output_run2.wav (performance run)"
echo ""
echo "To run again:"
echo "  source venv/bin/activate"
echo "  python example.py --text \"Your text here\""
echo ""
echo "To run benchmark:"
echo "  source venv/bin/activate"
echo "  python benchmark.py --concurrency 16 --num-requests 32"
echo ""
echo "To start FastAPI server:"
echo "  source venv/bin/activate"
echo "  cd fastapi && uvicorn app:app --host 0.0.0.0 --port 8000"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}            SETUP COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"

