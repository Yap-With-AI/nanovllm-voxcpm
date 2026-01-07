"""VoxCPM model module.

Provides the VoxCPM model architecture for text-to-speech generation.
This is a vendored copy of the model code from the voxcpm package,
allowing training without the external pip dependency.
"""

from .voxcpm import (
    LoRAConfig,
    VoxCPMConfig,
    VoxCPMDitConfig,
    VoxCPMEncoderConfig,
    VoxCPMModel,
)

__all__ = [
    "LoRAConfig",
    "VoxCPMConfig",
    "VoxCPMDitConfig",
    "VoxCPMEncoderConfig",
    "VoxCPMModel",
]

