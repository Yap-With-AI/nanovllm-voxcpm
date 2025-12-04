from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool
import base64
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = "~/VoxCPM-0.5B"
global_instances = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global_instances["server"] = AsyncVoxCPMServerPool(model_path=MODEL_PATH,
                                                       max_num_batched_tokens=8192,
                                                       max_num_seqs=16,
                                                       max_model_len=4096,
                                                       gpu_memory_utilization=0.95,
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
    return {"status": "ok"}


class AddPromptRequest(BaseModel):
    wav_base64: str
    wav_format: str
    prompt_text: str

@app.post("/add_prompt")
async def add_prompt(request: AddPromptRequest):
    wav = base64.b64decode(request.wav_base64)
    server : AsyncVoxCPMServerPool = global_instances["server"]

    prompt_id = await server.add_prompt(wav, request.wav_format, request.prompt_text)
    return {"prompt_id": prompt_id}

class RemovePromptRequest(BaseModel):
    prompt_id: str

@app.post("/remove_prompt")
async def remove_prompt(request: RemovePromptRequest):
    server : AsyncVoxCPMServerPool = global_instances["server"]
    await server.remove_prompt(request.prompt_id)
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    target_text : str
    prompt_id : str | None = None
    max_generate_length : int = 2000
    temperature : float = 1.0
    cfg_value : float = 1.5


async def numpy_to_bytes(gen) :
    async for data in gen:
        yield data.tobytes()

@app.post("/generate")
async def generate(request: GenerateRequest):
    server : AsyncVoxCPMServerPool = global_instances["server"]
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
    )
