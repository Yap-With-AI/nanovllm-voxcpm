"""Data preparation utilities for VoxCPM fine-tuning.

Provides manifest generation from WebDataset shards.
"""

from .manifest import (
    ManifestEntry,
    ManifestStats,
    generate_manifest,
    log_manifest_stats,
)

__all__ = [
    "ManifestEntry",
    "ManifestStats",
    "generate_manifest",
    "log_manifest_stats",
]
