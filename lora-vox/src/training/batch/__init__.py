"""Batch processing utilities for VoxCPM training.

Provides batch transformation, padding, and loss computation utilities.
"""

from .processor import BatchProcessor, BatchProcessorConfig
from .utils import (
    build_position_ids,
    compute_loss,
    create_empty_batch,
    pad_1d,
    pad_3d,
    remove_padding,
)

__all__ = [
    "BatchProcessor",
    "BatchProcessorConfig",
    "build_position_ids",
    "compute_loss",
    "create_empty_batch",
    "pad_1d",
    "pad_3d",
    "remove_padding",
]

