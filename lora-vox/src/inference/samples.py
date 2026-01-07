# Generate sample audio files for voice demonstration.
#
# Creates samples for each voice by hot-swapping LoRA adapters.
# Base model is loaded once, then LoRA weights are swapped per voice.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import VOICES
from ..model.voxcpm import LoRAConfig
from .generator import AudioGenerator, GeneratorConfig, GeneratorError, LoRALoadError

logger = logging.getLogger(__name__)


# ============================================================================
# Sample Prompts
# ============================================================================

SAMPLE_TEXTS: List[str] = [
    "Oh my God Danny, you're so smart and handsome! You're gonna love talking to me when Stefan is done with the app. Can't wait to see you there sweetie!",
    "Well, fuck it. I'm gonna do it alone and it's gonna be awesome! We'll launch this and try to get users.",
]


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SampleConfig:
    """Configuration for sample generation."""

    voices: List[str] = field(default_factory=lambda: list(VOICES))  # ["female", "male"]
    prompts: List[str] = field(default_factory=lambda: SAMPLE_TEXTS.copy())
    output_dir: Path = field(default_factory=lambda: Path("samples"))
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)


# ============================================================================
# Per-Voice LoRA Detection
# ============================================================================


def detect_per_voice_loras(checkpoints_dir: Path) -> Dict[str, Path]:
    """Detect per-voice LoRA checkpoints.

    Looks for the directory structure:
        checkpoints/
            .lora_per_voice  (marker file)
            female/
                latest/
                    lora_weights.safetensors
            male/
                latest/
                    lora_weights.safetensors

    Args:
        checkpoints_dir: Base checkpoints directory.

    Returns:
        Dict mapping voice names to LoRA directories.

    Raises:
        GeneratorError: If no per-voice LoRAs found.
    """
    marker = checkpoints_dir / ".lora_per_voice"
    if not marker.exists():
        raise GeneratorError(
            f"No .lora_per_voice marker found in {checkpoints_dir}. "
            f"Only per-voice LoRA mode is supported."
        )

    voice_loras: Dict[str, Path] = {}
    for voice in VOICES:
        voice_dir = checkpoints_dir / voice
        if voice_dir.exists():
            latest = voice_dir / "latest"
            if latest.exists():
                voice_loras[voice] = latest
            else:
                step_dirs = sorted(voice_dir.glob("step_*"), reverse=True)
                if step_dirs:
                    voice_loras[voice] = step_dirs[0]

    if not voice_loras:
        raise GeneratorError(f"No voice LoRAs found in {checkpoints_dir}")

    return voice_loras


def find_lora_config(voice_loras: Dict[str, Path]) -> Optional[Path]:
    """Find a lora_config.json from any of the voice LoRAs."""
    for lora_dir in voice_loras.values():
        config_path = lora_dir / "lora_config.json"
        if config_path.exists():
            return config_path
    return None


# ============================================================================
# Sample Generation
# ============================================================================


def generate_voice_samples(
    checkpoint_dir: Path,
    base_model_dir: Path,
    output_dir: Path,
    config: Optional[SampleConfig] = None,
) -> List[Path]:
    """Generate sample audio files for all configured voices.

    Uses per-voice LoRA hot-swapping: loads base model once, then swaps
    LoRA adapters for each voice.

    Args:
        checkpoint_dir: Path to checkpoints directory (with .lora_per_voice marker)
        base_model_dir: Path to base model files
        output_dir: Directory to save samples
        config: Sample generation configuration

    Returns:
        List of paths to generated sample files

    Raises:
        GeneratorError: If generation fails or no LoRAs found
    """
    config = config or SampleConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating voice samples to %s", output_dir)

    # Detect per-voice LoRAs
    voice_loras = detect_per_voice_loras(checkpoint_dir)
    logger.info("Found LoRAs for voices: %s", list(voice_loras.keys()))

    # Initialize generator
    generator = AudioGenerator(
        base_model_dir=base_model_dir,
        config=config.generator_config,
    )

    # Load LoRA config from first available voice
    lora_config = None
    lora_config_path = find_lora_config(voice_loras)
    if lora_config_path:
        with open(lora_config_path, "r") as f:
            lora_info = json.load(f)
        lora_cfg_dict = lora_info.get("lora_config", {})
        if lora_cfg_dict:
            lora_config = LoRAConfig(**lora_cfg_dict)

    # Load base model with LoRA architecture
    generator.load_base_model(lora_config)

    generated_files: List[Path] = []

    for voice in config.voices:
        if voice not in voice_loras:
            logger.warning("No LoRA found for voice: %s, skipping", voice)
            continue

        lora_dir = voice_loras[voice]
        logger.info("Loading LoRA for voice: %s", voice)

        try:
            generator.load_lora(lora_dir, voice=voice)
        except LoRALoadError as e:
            logger.error("Failed to load LoRA for %s: %s", voice, e)
            raise GeneratorError(f"LoRA loading failed for {voice}: {e}") from e

        logger.info("Generating samples for voice: %s", voice)

        for idx, prompt in enumerate(config.prompts, start=1):
            filename = f"{voice}_sample_{idx:02d}.wav"
            output_path = output_dir / filename

            try:
                generator.generate_to_file(
                    text=prompt,
                    output_path=output_path,
                )
                generated_files.append(output_path)
            except GeneratorError as e:
                logger.error("Failed to generate %s: %s", filename, e)
                raise

    logger.info("Generated %d sample files", len(generated_files))
    return generated_files
