# Inference module for VoxCPM 1.5 fine-tuned models.
#
# Provides per-voice LoRA sample generation with hot-swapping.

from .generator import AudioGenerator, GeneratorConfig, GeneratorError, LoRALoadError
from .samples import generate_voice_samples, SampleConfig

__all__ = [
    "AudioGenerator",
    "GeneratorConfig",
    "GeneratorError",
    "LoRALoadError",
    "generate_voice_samples",
    "SampleConfig",
]
