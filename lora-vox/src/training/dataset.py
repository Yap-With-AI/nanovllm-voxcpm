"""PyTorch dataset wrapper for VoxCPM training.

Wraps HuggingFace datasets for use with PyTorch DataLoader.
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ..config import AUDIO_COLUMN, DATASET_ID_COLUMN, PAD_TOKEN_ID
from .accelerator import Accelerator


# ============================================================================
# Dataset Wrapper
# ============================================================================

class VoxCPMDataset(TorchDataset):
    """PyTorch Dataset wrapper for VoxCPM training data.

    Extracts audio arrays, tokenized text, and metadata from a HuggingFace
    dataset for batch processing.
    """

    def __init__(self, hf_dataset: HFDataset) -> None:
        """Initialize dataset wrapper.

        Args:
            hf_dataset: HuggingFace dataset with tokenized text_ids.
        """
        self._dataset = hf_dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get single sample.

        Returns:
            Dict with text_ids, audio_array, audio_sampling_rate,
            dataset_id, and is_prompt fields.
        """
        item = self._dataset[idx]
        audio = item[AUDIO_COLUMN]

        return {
            "text_ids": item["text_ids"],
            "audio_array": audio["array"],
            "audio_sampling_rate": audio["sampling_rate"],
            "dataset_id": item.get(DATASET_ID_COLUMN, 0),
            "is_prompt": item.get("is_prompt", False),
        }


# ============================================================================
# Collation
# ============================================================================

def collate_voxcpm_batch(
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor | list[bool]]:
    """Collate batch of samples with padding.

    Args:
        batch: List of sample dicts from VoxCPMDataset.

    Returns:
        Dict with padded tensors ready for BatchProcessor.
    """
    text_tensors = [
        torch.tensor(sample["text_ids"], dtype=torch.int32)
        for sample in batch
    ]
    audio_tensors = [
        torch.tensor(sample["audio_array"], dtype=torch.float32)
        for sample in batch
    ]
    dataset_ids = torch.tensor(
        [sample["dataset_id"] for sample in batch],
        dtype=torch.int32,
    )
    is_prompts = [bool(sample.get("is_prompt", False)) for sample in batch]

    text_padded = _pad_sequences(text_tensors, pad_value=PAD_TOKEN_ID)
    audio_padded = _pad_sequences(audio_tensors, pad_value=float(PAD_TOKEN_ID))
    task_ids = torch.ones(len(batch), dtype=torch.int32)

    return {
        "text_tokens": text_padded,
        "audio_tokens": audio_padded,
        "task_ids": task_ids,
        "dataset_ids": dataset_ids,
        "is_prompts": is_prompts,
    }


# ============================================================================
# DataLoader Builder
# ============================================================================

def build_training_dataloader(
    hf_dataset: HFDataset,
    accelerator: Accelerator,
    batch_size: int,
    num_workers: int = 2,
    *,
    drop_last: bool = False,
) -> DataLoader:
    """Build DataLoader for training.

    Args:
        hf_dataset: Tokenized HuggingFace dataset.
        accelerator: Training accelerator for distributed setup.
        batch_size: Batch size per process.
        num_workers: Data loading workers.
        drop_last: Whether to drop incomplete final batch.

    Returns:
        Configured DataLoader.
    """
    torch_dataset = VoxCPMDataset(hf_dataset)

    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_voxcpm_batch,
        drop_last=drop_last,
    )


# ============================================================================
# Private Helpers
# ============================================================================

def _pad_sequences(
    sequences: list[torch.Tensor],
    pad_value: float,
) -> torch.Tensor:
    """Pad list of tensors to common length.

    Args:
        sequences: List of 1D tensors.
        pad_value: Value to use for padding.

    Returns:
        Stacked tensor with shape [batch, max_len].
    """
    if not sequences:
        return torch.empty(0)

    max_len = max(seq.shape[0] for seq in sequences)
    padded = []

    for seq in sequences:
        if seq.shape[0] < max_len:
            padding = (0, max_len - seq.shape[0])
            seq = torch.nn.functional.pad(seq, padding, value=pad_value)
        padded.append(seq)

    return torch.stack(padded)

