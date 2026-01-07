"""Artifact packaging and publishing for LoRA fine-tuned models.

Handles staging checkpoint files and uploading to HuggingFace Hub.
Supports per-voice LoRA structures only.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import (
    AUDIOVAE_FILE,
    CONFIG_JSON,
    LORA_CONFIG_FILE,
    LORA_WEIGHTS_FILE,
    LORA_WEIGHTS_FILE_ALT,
    MODEL_WEIGHTS_FILE,
    MODEL_WEIGHTS_FILE_ALT,
    TOKENIZER_FILES,
    VOICES,
)
from ..errors import PublishError
from .model_card import write_model_readme
from .token import require_hf_token
from .upload import create_repo_if_needed, upload_folder


logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def detect_per_voice_loras(checkpoint_dir: Path) -> dict[str, Path]:
    """Detect per-voice LoRA checkpoints.

    Args:
        checkpoint_dir: Path to checkpoints directory.

    Returns:
        Dict mapping voice names to LoRA directories.

    Raises:
        PublishError: If no per-voice LoRAs found in LoRA mode.
    """
    marker = checkpoint_dir / ".lora_per_voice"
    if not marker.exists():
        raise PublishError(
            f"No .lora_per_voice marker found in {checkpoint_dir}. "
            f"Only per-voice LoRA mode is supported."
        )

    voice_loras: dict[str, Path] = {}
    for voice in VOICES:
        voice_dir = checkpoint_dir / voice
        if voice_dir.exists():
            latest = voice_dir / "latest"
            if latest.exists():
                voice_loras[voice] = latest
            else:
                step_dirs = sorted(voice_dir.glob("step_*"), reverse=True)
                if step_dirs:
                    voice_loras[voice] = step_dirs[0]

    if not voice_loras:
        raise PublishError(f"No voice LoRAs found in {checkpoint_dir}")

    return voice_loras


def stage_artifacts(
    checkpoint_dir: Path,
    base_model_dir: Path,
    tokenizer_dir: Path,
    staging_dir: Path,
    samples_dir: Path | None = None,
) -> Path:
    """Stage per-voice LoRA checkpoint artifacts for publishing.

    Stages per-voice LoRAs into lora/<voice>/ subdirectories.

    Args:
        checkpoint_dir: Path to training checkpoint.
        base_model_dir: Path to base model (for audiovae.pth).
        tokenizer_dir: Path to tokenizer.
        staging_dir: Target directory for staged files.
        samples_dir: Optional path to generated voice samples.

    Returns:
        Path to staging directory.

    Raises:
        PublishError: If required checkpoint files are missing.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    _clear_directory(staging_dir)

    voice_loras = detect_per_voice_loras(checkpoint_dir)
    _stage_per_voice_loras(voice_loras, staging_dir)
    _stage_base_model(base_model_dir, staging_dir)

    _stage_tokenizer(tokenizer_dir, staging_dir)
    _stage_audiovae(base_model_dir, staging_dir)

    if samples_dir and samples_dir.exists():
        _stage_samples(samples_dir, staging_dir)

    logger.info("Staged artifacts to %s", staging_dir)
    return staging_dir


def publish_to_hub(
    staging_dir: Path,
    output_repo: str,
    *,
    commit_message: str = "Upload fine-tuned model",
) -> str:
    """Upload staged artifacts to HuggingFace Hub.

    Args:
        staging_dir: Directory containing staged artifacts.
        output_repo: HuggingFace repository ID.
        commit_message: Git commit message.

    Returns:
        URL of the commit on HuggingFace Hub.
    """
    token = require_hf_token()
    create_repo_if_needed(output_repo, token)
    return upload_folder(staging_dir, output_repo, token, commit_message=commit_message)


def run_publish_job(
    run_name: str,
    checkpoint_dir: Path,
    base_model_dir: Path,
    tokenizer_dir: Path,
    output_repo: str,
    base_model_repo: str,
    training_params: dict[str, Any] | None = None,
    samples_dir: Path | None = None,
) -> str:
    """Execute complete publish workflow.

    Args:
        run_name: Training run name (used as model title).
        checkpoint_dir: Path to training checkpoint.
        base_model_dir: Path to base model.
        tokenizer_dir: Path to tokenizer.
        output_repo: Target HuggingFace repository.
        base_model_repo: Base model repository ID.
        training_params: Optional training parameters for README.
        samples_dir: Optional path to voice samples directory.

    Returns:
        URL of the commit on HuggingFace Hub.
    """
    staging_dir = checkpoint_dir.parent / "staging"

    stage_artifacts(
        checkpoint_dir=checkpoint_dir,
        base_model_dir=base_model_dir,
        tokenizer_dir=tokenizer_dir,
        staging_dir=staging_dir,
        samples_dir=samples_dir,
    )

    write_model_readme(
        staging_dir / "README.md",
        run_name,
        base_model_repo,
        training_params,
    )

    _save_finetune_summary(staging_dir, run_name, base_model_repo, training_params)

    commit_url = publish_to_hub(staging_dir, output_repo)
    logger.info("Published to %s", output_repo)
    return commit_url


# ============================================================================
# Private: File Staging
# ============================================================================

def _clear_directory(directory: Path) -> None:
    """Remove all files and subdirectories from directory."""
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _stage_per_voice_loras(voice_loras: dict[str, Path], staging_dir: Path) -> None:
    """Stage per-voice LoRA checkpoints.

    Creates:
        staging_dir/lora/<voice>/lora_weights.safetensors
        staging_dir/lora/<voice>/lora_config.json
    """
    lora_dir = staging_dir / "lora"
    lora_dir.mkdir(exist_ok=True)

    for voice, checkpoint_path in voice_loras.items():
        voice_staging = lora_dir / voice
        voice_staging.mkdir(exist_ok=True)

        weights_path = checkpoint_path / LORA_WEIGHTS_FILE
        if not weights_path.exists():
            weights_path = checkpoint_path / LORA_WEIGHTS_FILE_ALT
        if not weights_path.exists():
            raise PublishError(f"LoRA weights not found for {voice} in {checkpoint_path}")

        shutil.copy2(weights_path, voice_staging / weights_path.name)

        config_path = checkpoint_path / LORA_CONFIG_FILE
        if config_path.exists():
            shutil.copy2(config_path, voice_staging / LORA_CONFIG_FILE)

        logger.info("Staged LoRA for voice: %s", voice)


def _stage_tokenizer(tokenizer_dir: Path, staging_dir: Path) -> None:
    """Copy tokenizer files to staging directory."""
    for filename in TOKENIZER_FILES:
        src = tokenizer_dir / filename
        if src.exists():
            shutil.copy2(src, staging_dir / filename)


def _stage_audiovae(base_model_dir: Path, staging_dir: Path) -> None:
    """Copy audio VAE weights from base model."""
    audiovae_path = base_model_dir / AUDIOVAE_FILE
    if audiovae_path.exists():
        shutil.copy2(audiovae_path, staging_dir / AUDIOVAE_FILE)


def _stage_samples(samples_dir: Path, staging_dir: Path) -> None:
    """Copy voice sample audio files."""
    samples_out = staging_dir / "samples"
    samples_out.mkdir(exist_ok=True)

    sample_count = 0
    for wav_file in samples_dir.glob("*.wav"):
        shutil.copy2(wav_file, samples_out / wav_file.name)
        sample_count += 1

    if sample_count > 0:
        logger.info("Staged %d voice samples", sample_count)


def _stage_base_model(base_model_dir: Path, staging_dir: Path) -> None:
    """Copy base model weights and config for self-contained repo."""
    weights_path = base_model_dir / MODEL_WEIGHTS_FILE
    if not weights_path.exists():
        weights_path = base_model_dir / MODEL_WEIGHTS_FILE_ALT
    if weights_path.exists():
        logger.info("Copying base model weights (%s)...", weights_path.name)
        shutil.copy2(weights_path, staging_dir / weights_path.name)

    config_path = base_model_dir / CONFIG_JSON
    if config_path.exists():
        shutil.copy2(config_path, staging_dir / CONFIG_JSON)


def _save_finetune_summary(
    staging_dir: Path,
    run_name: str,
    base_model_repo: str,
    training_params: dict[str, Any] | None,
) -> Path:
    """Save fine-tune summary JSON."""
    lora_dir = staging_dir / "lora"
    voices_trained = []
    if lora_dir.exists():
        voices_trained = [d.name for d in lora_dir.iterdir() if d.is_dir()]

    summary: dict[str, Any] = {
        "run_name": run_name,
        "base_model": base_model_repo,
        "mode": "lora",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "voices": voices_trained,
    }

    if training_params:
        summary["finetune_params"] = training_params

    path = staging_dir / "finetune_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return path
