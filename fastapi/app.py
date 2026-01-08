from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm.server import ConnectionLimitExceeded, RequestCancelled
import base64
import os
import uuid
from pydantic import BaseModel
from typing import List, Optional

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
    
    # Connection stats (Level 1: hard limit)
    active_connections = getattr(server, "active_connections", None)
    max_connections = getattr(server, "max_concurrent_connections", None)
    
    # Inference queue stats (Level 2: GPU batch capacity)
    active_inference = getattr(server, "active_inference", None)
    queued_inference = getattr(server, "queued_inference", None)
    max_inference_slots = getattr(server, "max_inference_slots", None)
    
    return {
        "status": "ok",
        "sample_rate": sample_rate,
        "available_voices": available_voices,
        # Connection layer
        "active_connections": active_connections,
        "max_concurrent_connections": max_connections,
        # Inference layer
        "active_inference": active_inference,
        "queued_inference": queued_inference,
        "max_inference_slots": max_inference_slots,
    }


@app.get("/voices")
async def list_voices() -> List[str]:
    """Get list of available voice names for hotswapping."""
    server = global_instances.get("server")
    if server:
        return await server.get_available_voices()
    return []


@app.get("/status")
async def status():
    """Get detailed server status including queue depth - useful for load balancers."""
    server = global_instances.get("server")
    if not server:
        return {"status": "not_ready"}
    
    active_connections = getattr(server, "active_connections", 0)
    max_connections = getattr(server, "max_concurrent_connections", 0)
    active_inference = getattr(server, "active_inference", 0)
    queued_inference = getattr(server, "queued_inference", 0)
    max_inference_slots = getattr(server, "max_inference_slots", 0)
    
    # Calculate capacity percentages
    connection_utilization = (active_connections / max_connections * 100) if max_connections > 0 else 0
    inference_utilization = (active_inference / max_inference_slots * 100) if max_inference_slots > 0 else 0
    
    # Determine if server can accept new requests
    can_accept = active_connections < max_connections
    
    return {
        "status": "ready" if can_accept else "at_capacity",
        "can_accept_requests": can_accept,
        "connections": {
            "active": active_connections,
            "max": max_connections,
            "utilization_pct": round(connection_utilization, 1),
        },
        "inference": {
            "active": active_inference,
            "queued": queued_inference,
            "max_slots": max_inference_slots,
            "utilization_pct": round(inference_utilization, 1),
        },
    }


@app.get("/queue")
async def get_queue():
    """Get list of request IDs currently waiting in queue."""
    server = global_instances.get("server")
    if not server:
        return {"pending_requests": []}
    
    pending_ids = getattr(server, "pending_request_ids", [])
    queued_count = getattr(server, "queued_inference", 0)
    
    return {
        "pending_requests": pending_ids,
        "queue_depth": queued_count,
    }


class CancelRequest(BaseModel):
    request_id: str


@app.post("/cancel")
async def cancel_request(request: CancelRequest):
    """Cancel a pending (queued) request. Only works for requests still waiting, not running."""
    server = global_instances.get("server")
    if not server:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    # Check if request is pending
    if hasattr(server, 'is_request_pending') and not server.is_request_pending(request.request_id):
        raise HTTPException(
            status_code=404, 
            detail=f"Request {request.request_id} not found in queue. It may have already started processing or completed."
        )
    
    # Cancel the request
    if hasattr(server, 'cancel_pending_request'):
        cancelled = server.cancel_pending_request(request.request_id)
        if cancelled:
            return {"status": "cancelled", "request_id": request.request_id}
    
    raise HTTPException(
        status_code=404,
        detail=f"Request {request.request_id} could not be cancelled."
    )


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
    request_id : str | None = None  # Optional client-provided request ID for cancellation


async def numpy_to_bytes(gen) :
    async for data in gen:
        yield data.tobytes()


@app.post("/generate")
async def generate(request: GenerateRequest):
    server = global_instances["server"]
    sample_rate = getattr(server, "sample_rate", None)
    
    # Generate request_id if not provided by client
    request_id = request.request_id or str(uuid.uuid4())
    
    # Check connection limit (Level 1: hard reject)
    if hasattr(server, 'active_connections') and hasattr(server, 'max_concurrent_connections'):
        if server.active_connections >= server.max_concurrent_connections:
            raise HTTPException(
                status_code=503,
                detail=f"Server has reached maximum concurrent connections ({server.max_concurrent_connections}). Please try again later.",
                headers={"Retry-After": "5"},
            )
    
    # Create generator - queuing happens inside when iteration starts
    gen = server.generate(
        target_text=request.target_text,
        prompt_latents=None,
        prompt_text="",
        prompt_id=request.prompt_id,
        max_generate_length=request.max_generate_length,
        temperature=request.temperature,
        cfg_value=request.cfg_value,
        voice=request.voice,
        request_id=request_id,
    )
    
    # Wrap generator to handle cancellation during streaming
    async def streaming_with_error_handling():
        try:
            async for data in gen:
                yield data.tobytes()
        except RequestCancelled:
            # Request was cancelled while waiting in queue
            # Stream ends cleanly - client can check X-Request-Id to correlate
            return
    
    # Current queue status for debugging
    queue_status = ""
    if hasattr(server, 'queued_inference') and hasattr(server, 'active_inference'):
        queue_status = f"queue={server.queued_inference},active={server.active_inference}"
    
    return StreamingResponse(
        streaming_with_error_handling(),
        media_type="audio/raw",
        headers={
            "X-Sample-Rate": str(sample_rate) if sample_rate else "",
            "X-Dtype": "float32",
            "X-Voice": request.voice or DEFAULT_VOICE,
            "X-Request-Id": request_id,  # Client can use this to cancel
            "X-Queue-Status": queue_status,
        },
    )
