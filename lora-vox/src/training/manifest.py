"""Training manifest loading and preprocessing.

Loads JSONL manifest files into HuggingFace datasets for training.
Supports filtering by voice for per-voice LoRA training.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

from datasets import Audio, Dataset, DatasetDict, load_dataset

from ..config import (
    AUDIO_COLUMN,
    DURATION_COLUMN,
    get_dataset_voice,
)


logger = logging.getLogger(__name__)


VOICE_COLUMN = "voice"


# ============================================================================
# Public API
# ============================================================================

def load_train_val_datasets(
    train_manifest: Path,
    val_manifest: Path | None,
    sample_rate: int,
    voice: Optional[str] = None,
) -> tuple[Dataset, Dataset | None]:
    """Load training and validation datasets from JSONL manifests.

    Args:
        train_manifest: Path to training manifest JSONL file.
        val_manifest: Path to validation manifest (optional).
        sample_rate: Target audio sample rate for resampling.
        voice: Optional public voice name ("female" or "male") to filter.
               If provided, only samples for this voice are loaded.

    Returns:
        Tuple of (train_dataset, val_dataset). val_dataset is None
        if no validation manifest was provided.

    Raises:
        ValueError: If required columns are missing from manifest.
    """
    data_files = {"train": str(train_manifest)}
    if val_manifest and val_manifest.exists():
        data_files["validation"] = str(val_manifest)

    logger.info("Loading manifests: train=%s", train_manifest)
    if val_manifest:
        logger.info("Loading manifests: val=%s", val_manifest)
    if voice:
        logger.info("Filtering for voice: %s", voice)
    logger.info("Target sample rate: %d Hz (audio will be resampled if needed)", sample_rate)

    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    # Filter by voice if specified
    dataset_voice = None
    if voice:
        dataset_voice = get_dataset_voice(voice)  # Map "female" -> "tara", etc.

    train_ds = _prepare_dataset(dataset_dict["train"], sample_rate, dataset_voice)
    val_ds = (
        _prepare_dataset(dataset_dict["validation"], sample_rate, dataset_voice)
        if "validation" in dataset_dict
        else None
    )

    logger.info("Loaded %d training samples", len(train_ds))
    if val_ds:
        logger.info("Loaded %d validation samples", len(val_ds))

    return train_ds, val_ds


def compute_sequence_lengths(
    dataset: Dataset,
    audio_vae_fps: float,
    patch_size: int,
) -> list[int]:
    """Estimate sequence lengths for filtering long samples.

    Used to pre-filter samples that would exceed max_batch_tokens.

    Args:
        dataset: Dataset with 'text_ids' column (already tokenized).
        audio_vae_fps: Audio VAE frames per second.
        patch_size: Model patch size for audio tokens.

    Returns:
        List of estimated sequence lengths for each sample.
    """
    text_ids_list = dataset["text_ids"]
    text_lengths = [len(ids) for ids in text_ids_list]

    if DURATION_COLUMN in dataset.column_names:
        durations = dataset[DURATION_COLUMN]
    else:
        # Fallback: compute duration from audio
        durations = []
        for i in range(len(dataset)):
            audio = dataset[i][AUDIO_COLUMN]
            duration = len(audio["array"]) / float(audio["sampling_rate"])
            durations.append(duration)

    lengths = []
    for text_len, duration in zip(text_lengths, durations):
        vae_frames = math.ceil(float(duration) * audio_vae_fps)
        audio_tokens = math.ceil(vae_frames / patch_size)
        # Total = text + audio + start/end tokens
        total_length = text_len + audio_tokens + 2
        lengths.append(total_length)

    return lengths


# ============================================================================
# Private Helpers
# ============================================================================

def _prepare_dataset(
    dataset: Dataset,
    sample_rate: int,
    dataset_voice: Optional[str] = None,
) -> Dataset:
    """Prepare raw dataset for training.

    Casts audio column to proper type, filters by voice if specified.
    """
    if AUDIO_COLUMN not in dataset.column_names:
        raise ValueError(f"Manifest missing required '{AUDIO_COLUMN}' column")

    # Filter by voice if specified
    if dataset_voice is not None:
        if VOICE_COLUMN not in dataset.column_names:
            raise ValueError(
                f"Manifest missing '{VOICE_COLUMN}' column, cannot filter by voice. "
                "Regenerate manifests with the updated prepare script."
            )
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: x[VOICE_COLUMN] == dataset_voice)
        logger.info("Filtered %d -> %d samples for voice '%s'", original_len, len(dataset), dataset_voice)

    # Cast audio column with target sample rate
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=sample_rate))

    return dataset

