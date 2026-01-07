"""Training utilities for VoxCPM 1.5 LoRA fine-tuning.

Provides accelerator, data loading, batch processing, checkpointing, and
the main training loop for per-voice LoRA training.
"""

from ..state import LoRASettings, TrainingConfig
from .accelerator import Accelerator, detect_optimal_dtype, get_gpu_name
from .batch import (
    BatchProcessor,
    BatchProcessorConfig,
    build_position_ids,
    compute_loss,
    create_empty_batch,
    pad_1d,
    pad_3d,
    remove_padding,
)
from .checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from .dataset import VoxCPMDataset, build_training_dataloader, collate_voxcpm_batch
from .loop import run_training_loop
from .manifest import compute_sequence_lengths, load_train_val_datasets
from .trainer import require_gpu, run_training_job, validate_checkpoint

__all__ = [
    "Accelerator",
    "BatchProcessor",
    "BatchProcessorConfig",
    "LoRASettings",
    "TrainingConfig",
    "VoxCPMDataset",
    "build_position_ids",
    "build_training_dataloader",
    "collate_voxcpm_batch",
    "compute_loss",
    "compute_sequence_lengths",
    "create_empty_batch",
    "detect_optimal_dtype",
    "find_latest_checkpoint",
    "get_gpu_name",
    "load_checkpoint",
    "load_train_val_datasets",
    "pad_1d",
    "pad_3d",
    "remove_padding",
    "require_gpu",
    "run_training_job",
    "run_training_loop",
    "save_checkpoint",
    "validate_checkpoint",
]
