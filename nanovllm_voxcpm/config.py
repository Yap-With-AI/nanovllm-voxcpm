import os
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Generic, TypeVar, List, Optional

T = TypeVar("T", bound=BaseModel)

@dataclass
class Config(Generic[T]):
    model: str
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 16
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    model_config: T | None = None
    devices : List[int] | None = None
    
    # LoRA configuration
    lora_path: Optional[str] = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.lora_path is not None:
            assert os.path.isdir(self.lora_path), f"LoRA path {self.lora_path} does not exist"
