import os
import json
from huggingface_hub import snapshot_download
from typing import List, Optional
import asyncio
import torch

# Enable fast matmul on Ampere+/Hopper; improves throughput with no quality loss for this use-case.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

try:
    # check if flash-attn is installed
    import flash_attn
except ImportError:
    raise ImportError("flash-attn is not installed. Please install it with `pip install flash-attn --no-build-isolation`")


class VoxCPM:
    @staticmethod
    def from_pretrained(
        model: str,
        inference_timesteps : int = 12,
        max_num_batched_tokens : int = 8192,
        max_num_seqs : int = 16,
        max_model_len : int = 512,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices : List[int] = [],
        lora_path: Optional[str] = None,
        **kwargs,
    ):
        """Load VoxCPM model from pretrained weights.
        
        Args:
            model: HuggingFace model ID or local path to the base model
            inference_timesteps: Number of diffusion timesteps for inference
            max_num_batched_tokens: Maximum number of tokens in a batch
            max_num_seqs: Maximum number of concurrent sequences
            max_model_len: Maximum model context length
            gpu_memory_utilization: GPU memory utilization target
            enforce_eager: Disable CUDA graph optimization
            devices: List of GPU device indices to use
            lora_path: Path to LoRA weights directory containing lora_config.json 
                       and lora_weights.safetensors (or lora_weights.bin)
            **kwargs: Additional arguments
        
        Returns:
            VoxCPM server instance (async or sync depending on context)
        """
        if "~" in model:
            model_path = os.path.expanduser(model)
            if not os.path.isdir(model_path):
                raise ValueError(f"Model path {model_path} does not exist")
        else:
            if not os.path.isdir(model):
                model_path = snapshot_download(repo_id=model)
            else:
                model_path = model
        
        # Resolve lora_path if provided
        resolved_lora_path = None
        if lora_path is not None:
            if "~" in lora_path:
                resolved_lora_path = os.path.expanduser(lora_path)
            else:
                resolved_lora_path = lora_path
            
            if not os.path.isdir(resolved_lora_path):
                raise ValueError(f"LoRA path {resolved_lora_path} does not exist")
        
        config_file = os.path.expanduser(os.path.join(model_path, "config.json"))
        
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file `{config_file}` not found")
        
        config = json.load(open(config_file))

        arch = config["architecture"]

        if len(devices) == 0:
            devices = [0]
        
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            is_async_mode = False
        else:
            is_async_mode = True
            

        if arch == "voxcpm":
            from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool, SyncVoxCPMServerPool

            if is_async_mode:
                return AsyncVoxCPMServerPool(
                    model_path=model_path,
                    inference_timesteps=inference_timesteps,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enforce_eager=enforce_eager,
                    devices=devices,
                    lora_path=resolved_lora_path,
                    **kwargs,
                )
            else:
                return SyncVoxCPMServerPool(
                    model_path=model_path,
                    inference_timesteps=inference_timesteps,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enforce_eager=enforce_eager,
                    devices=devices,
                    lora_path=resolved_lora_path,
                    **kwargs,
                )
        else:
            raise ValueError(f"Unsupported model architecture: {arch}")
