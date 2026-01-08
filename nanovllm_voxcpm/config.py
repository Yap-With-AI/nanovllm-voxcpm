import os
from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import Generic, TypeVar, List, Optional, Literal

T = TypeVar("T", bound=BaseModel)

@dataclass
class Config(Generic[T]):
    model: str
    max_num_batched_tokens: int = 23296  # 52 * 448
    max_num_seqs: int = 52
    max_model_len: int = 448
    gpu_memory_utilization: float = 0.93
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    model_config: T | None = None
    devices : List[int] | None = None
    
    # LoRA configuration
    lora_path: Optional[str] = None
    # Multi-LoRA support for voice hotswapping (e.g., {"female": "/path/female", "male": "/path/male"})
    lora_paths: Optional[dict[str, str]] = None
    default_voice: str = "female"
    
    # torch.compile configuration
    use_torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = "max-autotune"
    # Which submodules to compile: "all", "estimator", "encoder", "lm", "residual_lm"
    # Default includes all major compute paths for best performance
    compile_targets: List[str] = field(default_factory=lambda: ["estimator", "lm", "residual_lm", "encoder"])
    # Use fullgraph=True for maximum optimization (may fail on complex control flow)
    compile_fullgraph: bool = False
    # Dynamic shapes - set to True for variable batch/seq lengths
    compile_dynamic: bool = True
    
    # Async VAE: run VAE decode on separate CUDA stream to overlap with CPU operations
    async_vae: bool = True
    
    # Chunked prefill: split long prefills into chunks to reduce TTFB under concurrency
    # Smaller = lower TTFB, slightly more scheduling overhead. Set to 0 to disable.
    prefill_chunk_size: int = 48

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0  # Flash Attention paged KV cache requires 256
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.lora_path is not None:
            assert os.path.isdir(self.lora_path), f"LoRA path {self.lora_path} does not exist"
        if self.use_torch_compile:
            valid_targets = {"all", "estimator", "encoder", "lm", "residual_lm"}
            for target in self.compile_targets:
                assert target in valid_targets, f"Invalid compile target: {target}. Valid: {valid_targets}"
