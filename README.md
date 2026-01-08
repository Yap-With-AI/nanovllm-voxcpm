# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests with fair FIFO queuing
- Two-level resource management (connections + GPU batching)
- Request cancellation for queued requests
- Multi-voice LoRA hotswapping
- Friendly async API, easy to use in FastAPI (see [fastapi/app.py](fastapi/app.py))

## Quick Start

```bash
# Run the setup script (installs deps, downloads model, starts server)
bash main.sh
```

The script will:
1. Create a virtual environment
2. Install all dependencies (PyTorch, Flash Attention, etc.)
3. Download the model from HuggingFace
4. Start the FastAPI server
5. Run example tests
6. Keep the server running

## Manual Setup

```bash
# Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install -e .

# Start the FastAPI server
cd fastapi && uvicorn app:app --host 0.0.0.0 --port 8000
```

## Configuration

### Two-Level Resource Management

The server uses two separate knobs to manage resources efficiently:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `max_num_seqs` | GPU batch capacity - how many sequences can run in parallel on the GPU | 52 |
| `max_concurrent_connections` | Hard limit on connected clients - rejects beyond this | 104 |

```
┌────────────────────────────────────────────────────────────────┐
│  Level 1: Connection Limit (Hard Reject)                       │
│  max_concurrent_connections = 104                              │
│  → Immediate 503 rejection when full                           │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Level 2: Inference Queue (Fair FIFO)                          │
│  max_num_seqs = 52 (matches GPU batch capacity)                │
│  → Requests 53-104 wait in queue                               │
│  → Clients can cancel pending requests                         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  GPU Inference Engine (Continuous Batching)                    │
│  → Processes up to 52 sequences in parallel                    │
│  → As sequences finish, queued ones start immediately          │
└────────────────────────────────────────────────────────────────┘
```

## Usage

### Run Example (calls API)

```bash
source venv/bin/activate
python example.py --text "Hello, this is a test."
python example.py --text "Your text here" --temperature 0.8 --cfg-value 2
```

### Run Benchmark

```bash
source venv/bin/activate

# Basic benchmark
python benchmark.py

# Custom settings
python benchmark.py --concurrency 32 --num-requests 64

# All options
python benchmark.py \
  --server http://localhost:8000 \
  --num-requests 64 \
  --concurrency 32 \
  --temperature 1.0 \
  --cfg-value 2 \
  --max-generate-length 400 \
  --warmup 4
```

### Benchmark Output

```
============================================================
                    TTS BENCHMARK RESULTS
============================================================

-----------------------CONFIGURATION------------------------
  Total Requests:      32
  Concurrency:         16
  Wall Clock Time:     8.45s

-----------------TIME TO FIRST BYTE (TTFB)-----------------
  P50:       156.2 ms
  P90:       234.1 ms
  P95:       278.3 ms

---------REAL-TIME FACTOR (RTF) - lower is better----------
  P50:       0.037x
  P90:       0.045x
  P95:       0.052x

------------------------THROUGHPUT--------------------------
  Total Audio Generated:    89.6 s
  Aggregate XRT:           27.22x real-time

✅ REAL-TIME CAPABLE: P95 RTF < 1.0
============================================================
```

## API Endpoints

Once the server is running:

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with server stats |
| `/status` | GET | Detailed status for load balancers |
| `/voices` | GET | List available voice names |

**GET /health** response:
```json
{
  "status": "ok",
  "sample_rate": 24000,
  "available_voices": ["female", "male"],
  "active_connections": 10,
  "max_concurrent_connections": 104,
  "active_inference": 8,
  "queued_inference": 2,
  "max_inference_slots": 52
}
```

**GET /status** response (for load balancers):
```json
{
  "status": "ready",
  "can_accept_requests": true,
  "connections": {
    "active": 10,
    "max": 104,
    "utilization_pct": 9.6
  },
  "inference": {
    "active": 8,
    "queued": 2,
    "max_slots": 52,
    "utilization_pct": 15.4
  }
}
```

### Audio Generation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate audio (streaming) |
| `/add_prompt` | POST | Add voice prompt for cloning |
| `/remove_prompt` | POST | Remove voice prompt |

**POST /generate** request:
```json
{
  "target_text": "Hello, world!",
  "voice": "female",
  "temperature": 1.0,
  "cfg_value": 2.0,
  "max_generate_length": 400,
  "request_id": "optional-client-id"
}
```

Response headers include:
- `X-Request-Id`: Request ID for cancellation
- `X-Sample-Rate`: Audio sample rate
- `X-Queue-Status`: Current queue depth

### Queue Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/queue` | GET | List pending request IDs |
| `/cancel` | POST | Cancel a queued request |

**GET /queue** response:
```json
{
  "pending_requests": ["req-123", "req-456"],
  "queue_depth": 2
}
```

**POST /cancel** request:
```json
{
  "request_id": "req-123"
}
```

Response:
```json
{
  "status": "cancelled",
  "request_id": "req-123"
}
```

> **Note:** Only queued requests can be cancelled. Requests already running on GPU cannot be stopped.

## Client Example with Cancellation

```python
import httpx
import uuid
import asyncio

async def generate_with_cancel():
    request_id = str(uuid.uuid4())
    
    async with httpx.AsyncClient() as client:
        # Start generation (may queue if server busy)
        async with client.stream("POST", "http://localhost:8000/generate", json={
            "target_text": "Hello, this is a long text...",
            "request_id": request_id
        }) as response:
            # Request ID also available in header
            print(f"Request ID: {response.headers['X-Request-Id']}")
            
            async for chunk in response.aiter_bytes():
                # Process audio chunk...
                
                # If user cancels (e.g., navigates away)
                if should_cancel:
                    # Cancel if still in queue (won't stop if already running)
                    await client.post("http://localhost:8000/cancel", 
                                     json={"request_id": request_id})
                    break
```

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## License

MIT License

## Known Issue

If you see the error below:
```
ValueError: Missing parameters: ['base_lm.embed_tokens.weight', ...]
```

It's because nanovllm reads model parameters from `.safetensors` files, but the original format of VoxCPM is `.pt`. You can download the [safetensor](https://huggingface.co/euphoricpenguin22/VoxCPM-0.5B-Safetensors/blob/main/model.safetensors) file manually and put it into the model folder.
