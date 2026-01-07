"""Runtime state dataclasses for VoxCPM fine-tuning pipeline.

Contains dataclasses for runtime pipeline state (not static configuration).
Static configuration constants and schemas live in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM_STEPS,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_LOSS_DIFF_WEIGHT,
    DEFAULT_LOSS_STOP_WEIGHT,
    DEFAULT_MAX_BATCH_TOKENS,
    DEFAULT_NUM_ITERS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SAVE_INTERVAL,
    DEFAULT_VALID_INTERVAL,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
    SAMPLE_RATE,
    VOICES,
)


# ============================================================================
# Pipeline Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Runtime configuration for pipeline operations.

    Attributes:
        run_name: Unique name for training run.
        dataset_repo: HuggingFace dataset repository.
        voices: Voices to train (public names: "female", "male").
    """

    run_name: str
    dataset_repo: str
    voices: tuple[str, ...] = VOICES  # Default: all voices

    def __post_init__(self):
        # Validate voices
        for voice in self.voices:
            if voice not in VOICES:
                raise ValueError(f"Unknown voice: {voice}. Must be one of: {VOICES}")


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Runtime training loop configuration.

    Attributes:
        pretrained_path: Path to pretrained model directory.
        train_manifest: Path to training manifest JSONL.
        val_manifest: Path to validation manifest (optional).
        save_path: Directory for checkpoints.
        tensorboard_path: Directory for TensorBoard logs.
        voice: Voice being trained (public name: "female" or "male").
    """

    pretrained_path: Path
    train_manifest: Path
    val_manifest: Path | None
    save_path: Path
    tensorboard_path: Path | None

    voice: str = ""  # Voice being trained (for per-voice LoRA)

    sample_rate: int = SAMPLE_RATE
    hf_model_id: str = ""

    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS
    num_workers: int = DEFAULT_NUM_WORKERS
    num_iters: int = DEFAULT_NUM_ITERS
    learning_rate: float = 1e-4
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS

    log_interval: int = DEFAULT_LOG_INTERVAL
    valid_interval: int = DEFAULT_VALID_INTERVAL
    save_interval: int = DEFAULT_SAVE_INTERVAL

    loss_diff_weight: float = DEFAULT_LOSS_DIFF_WEIGHT
    loss_stop_weight: float = DEFAULT_LOSS_STOP_WEIGHT


# ============================================================================
# LoRA Configuration
# ============================================================================

@dataclass
class LoRASettings:
    """LoRA configuration for fine-tuning.

    Attributes:
        enable_lm: Enable LoRA for language model layers.
        enable_dit: Enable LoRA for diffusion transformer layers.
        enable_proj: Enable LoRA for projection layers.
        r: LoRA rank.
        alpha: LoRA alpha scaling factor.
        dropout: LoRA dropout rate.
    """

    enable_lm: bool = True
    enable_dit: bool = True
    enable_proj: bool = False
    r: int = 32
    alpha: int = 16
    dropout: float = 0.0
    target_modules_lm: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    target_modules_dit: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
