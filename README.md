# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API, easy to use in FastAPI (see [fastapi/app.py](fastapi/app.py))

## Quick Start

```bash
# Run the setup script (installs deps, downloads model, starts server)
./main.sh
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

## Usage

### Run Example (calls API)

```bash
source venv/bin/activate
python example.py --text "Hello, this is a test."
python example.py --text "Your text here" --temperature 0.8 --cfg-value 2.0
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
  --cfg-value 1.5 \
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

- `GET /health` - Health check
- `POST /generate` - Generate audio (streaming)
- `POST /add_prompt` - Add voice prompt for cloning
- `POST /remove_prompt` - Remove voice prompt

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
