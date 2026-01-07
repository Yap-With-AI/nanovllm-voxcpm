from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from nanovllm_voxcpm import VoxCPM
import base64
import os
from pydantic import BaseModel

app = FastAPI()

# Model repo contains base model + LoRA weights
# Structure:
#   - model.safetensors, tokenizer files at root (base model)
#   - lora/female/ (female voice LoRA)
#   - lora/male/ (male voice LoRA)
MODEL_REPO = "yapwithai/vox-1.5-orpheus-distil"
LORA_VOICE = "female"  # Use female voice by default

global_instances = {}


def get_model_paths() -> tuple[str, str]:
    """Get paths to the model and LoRA weights.
    
    Returns:
        Tuple of (model_path, lora_path)
    """
    from huggingface_hub import snapshot_download
    
    # Download the repo (contains both base model and LoRA)
    repo_path = snapshot_download(repo_id=MODEL_REPO)
    
    # Base model is at the root of the repo
    model_path = repo_path
    
    # LoRA weights are under lora/{voice}/
    lora_path = os.path.join(repo_path, "lora", LORA_VOICE)
    
    print(f"Model path: {model_path}")
    print(f"LoRA path: {lora_path}")
    
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    
    return model_path, lora_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Get model and LoRA paths (downloads if necessary)
    model_path, lora_path = get_model_paths()
    
    global_instances["server"] = VoxCPM.from_pretrained(
        model=model_path,
        max_num_batched_tokens=8192,   # Smaller batches for lower TTFB
        max_num_seqs=16,               # Smaller batches = faster individual TTFB
        max_model_len=512,             # 60 input + 375 audio (15s) + buffer
        gpu_memory_utilization=0.92,   # Slightly higher GPU use for batching
        enforce_eager=False,
        devices=[0],
        lora_path=lora_path,
    )
    await global_instances["server"].wait_for_ready()
    yield
    await global_instances["server"].stop()
    del global_instances["server"]

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    server = global_instances.get("server")
    sample_rate = getattr(server, "sample_rate", None)
    return {"status": "ok", "sample_rate": sample_rate}


class AddPromptRequest(BaseModel):
    wav_base64: str
    wav_format: str
    prompt_text: str

@app.post("/add_prompt")
async def add_prompt(request: AddPromptRequest):
    wav = base64.b64decode(request.wav_base64)
    server = global_instances["server"]

    prompt_id = await server.add_prompt(wav, request.wav_format, request.prompt_text)
    return {"prompt_id": prompt_id}

class RemovePromptRequest(BaseModel):
    prompt_id: str

@app.post("/remove_prompt")
async def remove_prompt(request: RemovePromptRequest):
    server = global_instances["server"]
    await server.remove_prompt(request.prompt_id)
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    target_text : str
    prompt_id : str | None = None
    max_generate_length : int = 400  # ~15 seconds max
    temperature : float = 1.0
    cfg_value : float = 2.0


async def numpy_to_bytes(gen) :
    async for data in gen:
        yield data.tobytes()

@app.post("/generate")
async def generate(request: GenerateRequest):
    server = global_instances["server"]
    sample_rate = getattr(server, "sample_rate", None)
    return StreamingResponse(
        numpy_to_bytes(
            server.generate(
                target_text=request.target_text,
                prompt_latents=None,
                prompt_text="",
                prompt_id=request.prompt_id,
                max_generate_length=request.max_generate_length,
                temperature=request.temperature,
                cfg_value=request.cfg_value,
            )
        ),
        media_type="audio/raw",
        headers={
            "X-Sample-Rate": str(sample_rate) if sample_rate else "",
            "X-Dtype": "float32",
        },
    )
