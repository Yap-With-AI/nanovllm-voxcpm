"""VoxCPM 1.5 fine-tuning module.

Provides end-to-end training pipeline for fine-tuning VoxCPM models
with full parameter updates or LoRA adapters.

Structure:
    src/
    ├── main.py          # Single entry point (re-exports from submodules)
    ├── config.py        # Configuration constants
    ├── state.py         # Runtime state dataclasses
    ├── errors.py        # Exception classes
    ├── data/            # Data preparation (manifest, tokenizer)
    ├── hf/              # HuggingFace operations (download, upload, publish)
    └── training/        # Training (trainer, loop, checkpoint, etc.)
"""

from .config import (
    BASE_MODEL_REPO,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_REPO,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    DEFAULT_NUM_ITERS,
    DEFAULT_OUTPUT_REPO,
    LEARNING_RATE_FULL,
    LEARNING_RATE_LORA,
    VOICES,
    VOICE_MAP,
    get_dataset_voice,
    get_public_voice,
)
from .errors import (
    ConfigurationError,
    DatasetError,
    GPUError,
    HuggingFaceError,
    PublishError,
    TokenizerError,
    TrainerError,
    VoxCPMTrainingError,
)
from .state import (
    LoRASettings,
    PipelineConfig,
    TrainingConfig,
)

__all__ = [
    # Config
    "BASE_MODEL_REPO",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DATASET_REPO",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_RANK",
    "DEFAULT_NUM_ITERS",
    "DEFAULT_OUTPUT_REPO",
    "LEARNING_RATE_FULL",
    "LEARNING_RATE_LORA",
    "VOICES",
    "VOICE_MAP",
    "get_dataset_voice",
    "get_public_voice",
    # State
    "LoRASettings",
    "PipelineConfig",
    "TrainingConfig",
    # Errors
    "ConfigurationError",
    "DatasetError",
    "GPUError",
    "HuggingFaceError",
    "PublishError",
    "TokenizerError",
    "TrainerError",
    "VoxCPMTrainingError",
]
