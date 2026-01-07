"""Main entry point for VoxCPM 1.5 fine-tuning pipeline.

This is the single entry point at the src root that orchestrates all operations.
Import from submodules for specific functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Re-export main functions for convenience
from .data import generate_manifest
from .hf import download_dataset, download_model
from .hf.publish import run_publish_job, stage_artifacts, publish_to_hub
from .training.trainer import run_training_job, require_gpu, validate_checkpoint


logger = logging.getLogger(__name__)


__all__ = [
    # Data preparation
    "generate_manifest",
    # HuggingFace operations
    "download_dataset",
    "download_model",
    # Publishing
    "run_publish_job",
    "stage_artifacts",
    "publish_to_hub",
    # Training
    "run_training_job",
    "require_gpu",
    "validate_checkpoint",
]
