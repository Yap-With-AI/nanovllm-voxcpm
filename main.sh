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
# Step 0.1: Check for HF_TOKEN (required for private LoRA model)
# ============================================
echo -e "\n${YELLOW}[0.1/8] Checking HuggingFace token...${NC}"

if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}ERROR: HF_TOKEN environment variable is not set.${NC}"
    echo -e "${RED}The LoRA model is private and requires authentication.${NC}"
    echo -e "${YELLOW}Please export your HuggingFace token:${NC}"
    echo -e "  export HF_TOKEN=your_huggingface_token"
    echo -e ""
    echo -e "You can get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

echo -e "${GREEN}✓ HF_TOKEN is set${NC}"

# Export for huggingface_hub to use
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_TOKEN="$HF_TOKEN"

# ============================================
# Step 0: Cleanup - kill existing processes
# ============================================
echo -e "\n${YELLOW}[0/8] Cleaning up existing processes...${NC}"

# Kill any existing uvicorn/python processes related to this project
pkill -f "uvicorn app:app" 2>/dev/null || true
pkill -f "nanovllm_voxcpm" 2>/dev/null || true
sleep 1

# Kill ALL spawned Python processes using the GPU (catches orphaned child processes)
# This is aggressive but necessary because multiprocessing spawned children don't get caught by pkill -f
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    echo "Killing GPU process: $pid"
    kill -9 "$pid" 2>/dev/null || true
done
sleep 2

# Clear GPU memory via PyTorch
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print('GPU memory cache cleared')
" 2>/dev/null || true

echo -e "${GREEN}✓ Cleanup done${NC}"

# ============================================
# Step 1: Check for NVIDIA GPU
# ============================================
echo -e "\n${YELLOW}[1/8] Checking GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. NVIDIA GPU required.${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -e "${GREEN}✓ GPU detected${NC}"

# ============================================
# Step 2: Create virtual environment
# ============================================
echo -e "\n${YELLOW}[2/8] Setting up virtual environment...${NC}"
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
echo -e "\n${YELLOW}[3/8] Installing dependencies...${NC}"

# Upgrade pip
pip install --upgrade pip -q

# Install PyTorch with CUDA
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Check if flash-attn is installed
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "Installing Flash Attention..."
    pip install einops -q
    
    # Get Python version for wheel selection
    PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
    
    echo "Trying pre-built wheel: $WHEEL_URL"
    if ! pip install "$WHEEL_URL" -q 2>/dev/null; then
        echo "Pre-built wheel not available, building from source..."
        pip install ninja psutil packaging wheel -q
        # Set TMPDIR to avoid cross-device link errors in containers
        TMPDIR="$SCRIPT_DIR/.tmp" pip install flash-attn --no-build-isolation
    fi
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
echo -e "\n${YELLOW}[4/8] Verifying CUDA...${NC}"
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

# Download the model repo (private, requires HF_TOKEN)
# This repo contains:
#   - Base model (model.safetensors, tokenizer files) at root
#   - LoRA weights under lora/female/ and lora/male/
python3 -c "
from huggingface_hub import snapshot_download
import os

model_repo = 'yapwithai/vox-1.5-orpheus-distil'
print(f'Downloading model: {model_repo}...')
print('(This is a private repo - using HF_TOKEN for authentication)')

path = snapshot_download(
    repo_id=model_repo,
    token=os.environ.get('HF_TOKEN'),
)
print(f'Model downloaded to: {path}')

# Verify base model files exist
model_file = os.path.join(path, 'model.safetensors')
config_file = os.path.join(path, 'config.json')

if os.path.exists(model_file):
    print(f'  ✓ model.safetensors')
else:
    print(f'  ✗ model.safetensors not found!')
    exit(1)

if os.path.exists(config_file):
    print(f'  ✓ config.json')
else:
    print(f'  ✗ config.json not found!')
    exit(1)

# Verify the female voice LoRA exists
lora_female_path = os.path.join(path, 'lora', 'female')
if os.path.isdir(lora_female_path):
    print(f'Female voice LoRA found at: {lora_female_path}')
    
    # Verify required files exist
    lora_config_file = os.path.join(lora_female_path, 'lora_config.json')
    lora_weights_file = os.path.join(lora_female_path, 'lora_weights.safetensors')
    
    if os.path.exists(lora_config_file):
        print(f'  ✓ lora_config.json')
    else:
        print(f'  ✗ lora_config.json not found!')
        exit(1)
    
    if os.path.exists(lora_weights_file):
        print(f'  ✓ lora_weights.safetensors')
    else:
        lora_weights_file_alt = os.path.join(lora_female_path, 'lora_weights.bin')
        if os.path.exists(lora_weights_file_alt):
            print(f'  ✓ lora_weights.bin')
        else:
            print(f'  ✗ LoRA weights not found!')
            exit(1)
else:
    print(f'ERROR: Female voice LoRA not found at {lora_female_path}')
    exit(1)

print('All required files verified!')
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
    --cfg-value 2.0

echo -e "\n${GREEN}✓ Run 1 complete - saved to output_run1.wav${NC}"

echo -e "\n${YELLOW}[7/8] Running example.py (Run 2 - actual performance)...${NC}"
echo "----------------------------------------"

python3 example.py \
    --text "This is the second test run. The model should now be fully warmed up and running at optimal speed. This demonstrates real-time streaming text to speech synthesis." \
    --output output_run2.wav \
    --temperature 1.0 \
    --cfg-value 2.0

echo -e "\n${GREEN}✓ Run 2 complete - saved to output_run2.wav${NC}"

# ============================================
# Step 8: Summary (server keeps running)
# ============================================
echo -e "\n${YELLOW}[8/8] Summary${NC}"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}              SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}✓ FastAPI server is running on http://0.0.0.0:8000${NC}"
echo "  Server PID: $SERVER_PID"
echo ""
echo "Output files:"
echo "  - output_run1.wav (warmup run)"
echo "  - output_run2.wav (performance run)"
echo ""
echo "Run example:"
echo "  source venv/bin/activate"
echo "  python example.py --text \"Your text here\""
echo ""
echo "Run benchmark:"
echo "  source venv/bin/activate"
echo "  python benchmark.py --concurrency 16 --num-requests 32"
echo ""
echo "Stop server:"
echo "  kill $SERVER_PID"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}            SETUP COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"

