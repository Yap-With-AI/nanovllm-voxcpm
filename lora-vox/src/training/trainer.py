"""Training execution for VoxCPM 1.5 LoRA fine-tuning.

Handles model loading, LoRA configuration, and training job orchestration.
Always trains per-voice LoRA adapters.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from ..config import (
    DEFAULT_GRAD_ACCUM_STEPS,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_BATCH_TOKENS,
    DEFAULT_SAVE_INTERVAL,
    DEFAULT_VALID_INTERVAL,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
    SAMPLE_RATE,
)
from ..errors import GPUError, TrainerError
from ..state import LoRASettings, TrainingConfig
from .checkpoint import find_latest_checkpoint
from .loop import run_training_loop


logger = logging.getLogger(__name__)


# ============================================================================
# Private: GPU Utilities
# ============================================================================

def _check_gpu_available() -> bool:
    """Check if CUDA GPU is available."""
    return torch.cuda.is_available()


def _get_gpu_info() -> dict:
    """Get information about available GPUs."""
    if not _check_gpu_available():
        return {"available": False}

    device_count = torch.cuda.device_count()
    devices = []
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "compute_capability": (props.major, props.minor),
        })

    return {
        "available": True,
        "device_count": device_count,
        "devices": devices,
    }


def _log_gpu_info() -> None:
    """Log GPU information."""
    info = _get_gpu_info()
    if not info["available"]:
        logger.warning("No CUDA GPU available")
        return

    logger.info("GPU devices: %d", info["device_count"])
    for device in info["devices"]:
        logger.info(
            "  GPU %d: %s (%.1f GB, sm_%d%d)",
            device["index"],
            device["name"],
            device["total_memory_gb"],
            device["compute_capability"][0],
            device["compute_capability"][1],
        )


# ============================================================================
# Private: Model Loading
# ============================================================================

def _load_voxcpm_model(
    pretrained_path: Path,
    lora_settings: LoRASettings,
) -> tuple:
    """Load VoxCPM model with LoRA.

    Args:
        pretrained_path: Path to pretrained model.
        lora_settings: LoRA configuration.

    Returns:
        Tuple of (model, tokenizer, audio_vae).
    """
    from ..model import VoxCPMModel
    from ..model.voxcpm import LoRAConfig

    lora_config = LoRAConfig(
        enable_lm=lora_settings.enable_lm,
        enable_dit=lora_settings.enable_dit,
        enable_proj=lora_settings.enable_proj,
        r=lora_settings.r,
        alpha=lora_settings.alpha,
        dropout=lora_settings.dropout,
        target_modules_lm=list(lora_settings.target_modules_lm),
        target_modules_dit=list(lora_settings.target_modules_dit),
    )

    logger.info("Loading model from: %s", pretrained_path)
    logger.info("LoRA: rank=%d, alpha=%d", lora_settings.r, lora_settings.alpha)

    model = VoxCPMModel.from_local(
        str(pretrained_path),
        optimize=False,
        training=True,
        lora_config=lora_config,
    )

    tokenizer = model.text_tokenizer
    audio_vae = model.audio_vae

    return model, tokenizer, audio_vae


# ============================================================================
# Public: Main Entry Points
# ============================================================================

def require_gpu() -> None:
    """Require GPU to be available.

    Raises:
        GPUError: If no GPU is available.
    """
    if not _check_gpu_available():
        raise GPUError("No CUDA GPU available. Training requires GPU.")
    _log_gpu_info()


def run_training_job(
    run_name: str,
    base_model_dir: Path,
    train_manifest: Path,
    val_manifest: Path | None,
    checkpoints_dir: Path,
    logs_dir: Path,
    batch_size: int,
    learning_rate: float,
    num_iters: int,
    voice: str,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
    log_interval: int = DEFAULT_LOG_INTERVAL,
    valid_interval: int = DEFAULT_VALID_INTERVAL,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
    hf_model_id: str = "",
) -> Path:
    """Run VoxCPM LoRA training job.

    Args:
        run_name: Name for this training run.
        base_model_dir: Path to base model.
        train_manifest: Path to training manifest.
        val_manifest: Path to validation manifest (optional).
        checkpoints_dir: Directory for checkpoints.
        logs_dir: Directory for TensorBoard logs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        num_iters: Number of training iterations.
        voice: Voice to train (public name: "female" or "male").
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        warmup_steps: Warmup steps for scheduler.
        weight_decay: Weight decay for optimizer.
        max_batch_tokens: Maximum tokens per batch.
        grad_accum_steps: Gradient accumulation steps.
        log_interval: Steps between log outputs.
        valid_interval: Steps between validation runs.
        save_interval: Steps between checkpoint saves.
        hf_model_id: HuggingFace model ID for LoRA config.

    Returns:
        Path to final checkpoint directory.

    Raises:
        TrainerError: If training fails.
        GPUError: If no GPU available.
    """
    require_gpu()

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    lora_settings = LoRASettings(
        r=lora_rank,
        alpha=lora_alpha,
    )

    # Load model with LoRA
    try:
        model, tokenizer, audio_vae = _load_voxcpm_model(
            base_model_dir,
            lora_settings=lora_settings,
        )
    except Exception as exc:
        raise TrainerError(f"Failed to load model: {exc}") from exc

    # Build training config
    config = TrainingConfig(
        pretrained_path=base_model_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        save_path=checkpoints_dir,
        tensorboard_path=logs_dir,
        voice=voice,
        sample_rate=SAMPLE_RATE,
        hf_model_id=hf_model_id,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        num_workers=2,
        num_iters=num_iters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_batch_tokens=max_batch_tokens,
        log_interval=log_interval,
        valid_interval=valid_interval,
        save_interval=save_interval,
    )

    logger.info("Starting LoRA training: %s", run_name)
    logger.info("  Voice: %s", voice)
    logger.info("  Iterations: %d", num_iters)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Learning rate: %.2e", learning_rate)
    logger.info("  LoRA rank: %d, alpha: %d", lora_rank, lora_alpha)

    # Run training
    try:
        checkpoint_dir = run_training_loop(model, tokenizer, audio_vae, config)
    except Exception as exc:
        raise TrainerError(f"Training failed: {exc}") from exc

    if checkpoint_dir is None or not checkpoint_dir.exists():
        checkpoint_dir = find_latest_checkpoint(checkpoints_dir)

    if checkpoint_dir is None:
        raise TrainerError("Training completed but no checkpoint found")

    logger.info("Training complete. Checkpoint: %s", checkpoint_dir)
    return checkpoint_dir


def validate_checkpoint(checkpoint_dir: Path) -> bool:
    """Validate that a LoRA checkpoint is complete.

    Args:
        checkpoint_dir: Path to checkpoint directory.

    Returns:
        True if checkpoint is valid.

    Raises:
        TrainerError: If checkpoint is invalid.
    """
    required_sets = [
        ["lora_weights.safetensors", "lora_config.json"],
        ["lora_weights.bin", "lora_config.json"],
    ]

    for required in required_sets:
        if all((checkpoint_dir / f).exists() for f in required):
            logger.info("Validated checkpoint: %s", checkpoint_dir)
            return True

    missing = [f for f in required_sets[0] if not (checkpoint_dir / f).exists()]
    raise TrainerError(f"Incomplete checkpoint, missing: {missing}")

