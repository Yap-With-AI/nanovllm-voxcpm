# Audio generation using fine-tuned VoxCPM 1.5 models.
#
# Per-voice LoRA hot-swapping: Load base model once, swap LoRA adapters per voice.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from ..model import VoxCPMModel
from ..model.voxcpm import LoRAConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Errors
# ============================================================================


class GeneratorError(Exception):
    """Raised when audio generation fails."""


class LoRALoadError(GeneratorError):
    """Raised when LoRA weights fail to load correctly."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for audio generation.
    
    Tuning guide:
        cfg_value: Lower (1.2-1.5) = more expressive/emotional
                   Higher (2.0-3.0) = more clarity, tighter adherence
        inference_timesteps: Higher = better quality but slower
                             10 = fast draft, 25 = good quality, 50 = best
    """

    cfg_value: float = 1.5  # Lower for more expressiveness
    inference_timesteps: int = 25  # Higher for better quality
    max_len: int = 1000  # Longer audio support
    device: str = "cuda"
    dtype: str = "bfloat16"


# ============================================================================
# Generator
# ============================================================================


class AudioGenerator:
    """Generate audio from text using per-voice LoRA adapters.

    Hot-swap mode: Load base model once, swap LoRA adapters per voice.
    Use `load_base_model()` then `load_lora()` for each voice.
    """

    def __init__(
        self,
        base_model_dir: Path,
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        """Initialize the generator.

        Args:
            base_model_dir: Path to base model files
            config: Generation configuration
        """
        self.base_model_dir = Path(base_model_dir)
        self.config = config or GeneratorConfig()
        self._model: Optional[VoxCPMModel] = None
        self._current_lora: Optional[str] = None
        self._lora_config: Optional[LoRAConfig] = None

    # ------------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------------

    def _load_lora_weights_from_path(
        self,
        model: VoxCPMModel,
        lora_dir: Path,
    ) -> None:
        """Load LoRA weights into model with validation.

        Uses the official VoxCPM load_lora_weights() API as per finetune.md:
        - Returns (loaded_keys, skipped_keys)
        - Validates that skipped_keys is empty
        - Handles torch.compile's _orig_mod prefix

        Args:
            model: Model to load weights into.
            lora_dir: Directory containing LoRA weights.

        Raises:
            LoRALoadError: If weights file not found or keys are skipped.
        """
        # Use official VoxCPM API: load_lora_weights() returns (loaded, skipped)
        # Per FAQ #5: "Check load_lora() return value - skipped_keys should be empty"
        loaded_keys, skipped_keys = model.load_lora_weights(str(lora_dir))

        # Validate: skipped_keys must be empty per official docs
        if skipped_keys:
            raise LoRALoadError(
                f"LoRA loading failed: {len(skipped_keys)} keys in checkpoint "
                f"were skipped (not loaded into model). First few: {skipped_keys[:5]}. "
                f"This usually means the model architecture doesn't match the LoRA config."
            )

        logger.info("Loaded %d LoRA keys from %s", len(loaded_keys), lora_dir)

    def load_base_model(self, lora_config: Optional[LoRAConfig] = None) -> VoxCPMModel:
        """Load base model with LoRA layers (weights not loaded yet).

        Args:
            lora_config: LoRA configuration. If None, uses default config.

        Returns:
            The loaded model (also stored in self._model).
        """
        if lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
                r=32,
                alpha=16,
                dropout=0.0,
            )

        self._lora_config = lora_config
        logger.info(
            "Loading base model with LoRA (r=%d, alpha=%d)",
            lora_config.r,
            lora_config.alpha,
        )

        model = VoxCPMModel.from_local(
            str(self.base_model_dir),
            optimize=True,
            training=False,
            lora_config=lora_config,
        )

        # Enable LoRA for inference
        if hasattr(model, "set_lora_enabled"):
            model.set_lora_enabled(True)
            logger.info("LoRA enabled for inference")

        self._model = model
        self._current_lora = None
        return model

    def load_lora(self, lora_dir: Path, voice: Optional[str] = None) -> None:
        """Hot-swap LoRA weights for a specific voice.

        Args:
            lora_dir: Directory containing LoRA weights (lora_weights.safetensors).
            voice: Voice name for logging. Auto-detected from path if not set.

        Raises:
            GeneratorError: If model not loaded.
            LoRALoadError: If LoRA weights have skipped keys.
        """
        if self._model is None:
            raise GeneratorError("Model not loaded. Call load_base_model() first.")

        lora_dir = Path(lora_dir)

        # Auto-detect voice from directory name
        if voice is None:
            voice = lora_dir.name
            if voice == "latest":
                voice = lora_dir.parent.name

        logger.info("Loading LoRA for voice: %s from %s", voice, lora_dir)

        # Handle "latest" symlink
        if (lora_dir / "latest").exists():
            lora_dir = lora_dir / "latest"

        self._load_lora_weights_from_path(self._model, lora_dir)
        self._current_lora = voice

        # Ensure LoRA is enabled
        if hasattr(self._model, "set_lora_enabled"):
            self._model.set_lora_enabled(True)

    @property
    def current_voice(self) -> Optional[str]:
        """Get the currently loaded LoRA voice, or None."""
        return self._current_lora

    # ------------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------------

    def generate(self, text: str) -> np.ndarray:
        """Generate audio from text.

        The loaded LoRA adapter determines the voice.

        Args:
            text: Text to synthesize

        Returns:
            Audio as numpy array (float32, model sample rate)
        """
        if self._model is None:
            raise GeneratorError("Model not loaded. Call load_base_model() first.")

        logger.info("Generating audio for: %s", text[:80] + "..." if len(text) > 80 else text)

        with torch.no_grad():
            audio = self._model.generate(
                target_text=text,
                inference_timesteps=self.config.inference_timesteps,
                cfg_value=self.config.cfg_value,
            )

        if audio is None or len(audio) == 0:
            raise GeneratorError("Model returned empty audio")

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()

        audio = audio.flatten().astype(np.float32)
        audio = self._normalize_audio(audio)

        return audio

    def generate_to_file(self, text: str, output_path: Path) -> Path:
        """Generate audio and save to file.

        Args:
            text: Text to synthesize
            output_path: Output file path

        Returns:
            Path to saved file
        """
        audio = self.generate(text)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sample_rate = self._model.sample_rate if self._model else 44100
        sf.write(str(output_path), audio, sample_rate)

        duration = len(audio) / sample_rate
        logger.info("Saved %s (%.2fs)", output_path.name, duration)

        return output_path

    @staticmethod
    def _normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """Normalize audio to target peak level."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * target_peak
        return audio

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        if self._model is not None:
            return self._model.sample_rate
        return 44100
