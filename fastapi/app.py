from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from nanovllm_voxcpm import VoxCPM
import base64
from pydantic import BaseModel

app = FastAPI()

MODEL = "openbmb/VoxCPM1.5"
global_instances = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # VoxCPM.from_pretrained handles HuggingFace download automatically
    global_instances["server"] = VoxCPM.from_pretrained(
        model=MODEL,
        max_num_batched_tokens=8192,   # Allow larger text/audio batches for 64-way concurrency
        max_num_seqs=64,               # Maximized concurrent streams on single GPU
        max_model_len=512,             # 60 input + 375 audio (15s) + buffer
        gpu_memory_utilization=0.92,   # Slightly higher GPU use for batching
        enforce_eager=False,
        devices=[0],
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
    cfg_value : float = 1.5


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
