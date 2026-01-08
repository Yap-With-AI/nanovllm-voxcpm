from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from nanovllm_voxcpm import VoxCPM
import base64
import os
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Model repo contains base model + LoRA weights
# Structure:
#   - model.safetensors, tokenizer files at root (base model)
#   - lora/female/ (female voice LoRA)
#   - lora/male/ (male voice LoRA)
MODEL_REPO = "yapwithai/vox-1.5-orpheus-distil"
DEFAULT_VOICE = "female"  # Default voice

global_instances = {}


def get_multi_lora_paths() -> tuple[str, dict[str, str]]:
    """Get paths to the model and all LoRA weights for hotswapping.
    
    Returns:
        Tuple of (model_path, lora_paths_dict)
    """
    from huggingface_hub import snapshot_download
    
    # Download the repo (contains both base model and LoRAs)
    repo_path = snapshot_download(repo_id=MODEL_REPO)
    
    # Base model is at the root of the repo
    model_path = repo_path
    
    # Discover all available LoRA voices
    lora_base = os.path.join(repo_path, "lora")
    lora_paths = {}
    
    # Look for female and male LoRAs
    for voice in ["female", "male"]:
        voice_path = os.path.join(lora_base, voice)
        if os.path.isdir(voice_path):
            lora_paths[voice] = voice_path
            print(f"Found LoRA for voice '{voice}': {voice_path}")
        else:
            print(f"Warning: LoRA path not found for voice '{voice}': {voice_path}")
    
    if not lora_paths:
        raise FileNotFoundError(f"No LoRA paths found in {lora_base}")
    
    print(f"Model path: {model_path}")
    print(f"LoRA paths: {lora_paths}")
    
    return model_path, lora_paths


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Get model and multi-LoRA paths (downloads if necessary)
    model_path, lora_paths = get_multi_lora_paths()
    
    global_instances["server"] = VoxCPM.from_pretrained(
        model=model_path,
        max_num_batched_tokens=23296,  # 52 * 448
        max_num_seqs=52,               # Max concurrent sequences
        max_model_len=448,             # 60 input + 375 audio (~15s)
        gpu_memory_utilization=0.93,   # Higher GPU use for KV cache
        enforce_eager=False,
        devices=[0],
        # Multi-LoRA hotswapping: load both female and male at startup
        lora_paths=lora_paths,
        default_voice=DEFAULT_VOICE,
        # torch.compile for DiT estimator: 10-20% TTFB improvement
        # The estimator runs inference_timesteps (12) times per token - highest ROI target
        use_torch_compile=True,
        compile_mode="max-autotune-no-cudagraphs",  # Avoids conflict with nanovllm's CUDA graph capture
        compile_targets=["estimator"],   # Only compile the DiT, not the full model
    )
    await global_instances["server"].wait_for_ready()
    
    # Log available voices
    voices = await global_instances["server"].get_available_voices()
    print(f"Server ready with voices: {voices}")
    
    yield
    await global_instances["server"].stop()
    del global_instances["server"]

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    server = global_instances.get("server")
    sample_rate = getattr(server, "sample_rate", None)
    available_voices = await server.get_available_voices() if server else []
    return {
        "status": "ok",
        "sample_rate": sample_rate,
        "available_voices": available_voices,
    }


@app.get("/voices")
async def list_voices() -> List[str]:
    """Get list of available voice names for hotswapping."""
    server = global_instances.get("server")
    if server:
        return await server.get_available_voices()
    return []


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
    voice : str | None = None  # Voice selection: "female" or "male" (default: female)


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
                voice=request.voice,
            )
        ),
        media_type="audio/raw",
        headers={
            "X-Sample-Rate": str(sample_rate) if sample_rate else "",
            "X-Dtype": "float32",
            "X-Voice": request.voice or DEFAULT_VOICE,
        },
    )
