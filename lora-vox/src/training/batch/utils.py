"""Batch utility functions for tensor padding and manipulation.

Provides reusable tensor padding, position ID generation, and loss
computation utilities used by the batch processor and training loop.
"""

from __future__ import annotations

import torch

from ...config import PAD_TOKEN_ID


# ============================================================================
# Padding Utilities
# ============================================================================

def remove_padding(tensor: torch.Tensor, pad_value: int = PAD_TOKEN_ID) -> torch.Tensor:
    """Remove padding from tensor.

    Args:
        tensor: 1D tensor potentially containing padding.
        pad_value: Value used for padding.

    Returns:
        Tensor with trailing padding removed.
    """
    pad_positions = (tensor == pad_value).nonzero(as_tuple=True)
    if pad_positions[0].numel() == 0:
        return tensor
    return tensor[: int(pad_positions[0][0])]


def pad_1d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate 1D tensor to target length.

    Args:
        tensor: Input 1D tensor.
        target_len: Desired output length.

    Returns:
        Tensor padded with zeros or truncated to target_len.
    """
    if tensor.size(0) >= target_len:
        return tensor[:target_len]
    padding = torch.zeros(
        target_len - tensor.size(0),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding])


def pad_3d(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate 3D tensor along first dimension.

    Args:
        tensor: Input 3D tensor of shape [T, ...].
        target_len: Desired length for first dimension.

    Returns:
        Tensor padded with zeros or truncated to target_len.
    """
    if tensor.size(0) >= target_len:
        return tensor[:target_len]
    padding = torch.zeros(
        (target_len - tensor.size(0),) + tensor.shape[1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding])


# ============================================================================
# Position ID Generation
# ============================================================================

def build_position_ids(
    lengths: list[int],
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build position IDs tensor for a batch.

    Args:
        lengths: List of sequence lengths for each sample.
        max_len: Maximum sequence length (for padding).
        device: Target device for output tensor.

    Returns:
        Tensor of shape [batch_size, max_len] with position indices.
    """
    position_ids = []
    for length in lengths:
        clipped_len = min(length, max_len)
        pos = torch.arange(clipped_len, device=device)
        if clipped_len < max_len:
            padding = torch.zeros(
                max_len - clipped_len,
                dtype=pos.dtype,
                device=device,
            )
            pos = torch.cat([pos, padding])
        position_ids.append(pos)
    return torch.stack(position_ids)


# ============================================================================
# Empty Batch Factory
# ============================================================================

def create_empty_batch(
    max_length: int,
    patch_size: int,
    feat_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create empty batch for edge cases.

    Args:
        max_length: Maximum sequence length.
        patch_size: Audio patch size.
        feat_dim: Audio feature dimension.
        device: Target device for tensors.

    Returns:
        Dict with zero-filled tensors matching expected batch format.
    """
    shape = (0, max_length)
    return {
        "text_tokens": torch.zeros(shape, dtype=torch.int32, device=device),
        "audio_feats": torch.zeros(
            (0, max_length, patch_size, feat_dim),
            dtype=torch.float32,
            device=device,
        ),
        "text_mask": torch.zeros(shape, dtype=torch.int32, device=device),
        "audio_mask": torch.zeros(shape, dtype=torch.int32, device=device),
        "loss_mask": torch.zeros(shape, dtype=torch.int32, device=device),
        "position_ids": torch.zeros(shape, dtype=torch.int64, device=device),
        "labels": torch.zeros(shape, dtype=torch.int32, device=device),
        "audio_duration_consumed": torch.zeros(1, dtype=torch.int64, device=device),
        "text_token_consumed": torch.zeros(1, dtype=torch.int64, device=device),
    }


# ============================================================================
# Loss Computation
# ============================================================================

def compute_loss(
    outputs: dict[str, torch.Tensor],
    diff_weight: float,
    stop_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted total loss from model outputs.

    Args:
        outputs: Model outputs dict containing loss/* keys.
        diff_weight: Weight for diffusion loss.
        stop_weight: Weight for stop loss.

    Returns:
        Tuple of (total_loss tensor, loss_dict with scalar values).
    """
    device = outputs.get("loss/diff", torch.zeros(1)).device
    total = torch.tensor(0.0, device=device)
    loss_dict: dict[str, float] = {}
    for key, value in outputs.items():
        if key.startswith("loss/"):
            weight = diff_weight if key == "loss/diff" else stop_weight if key == "loss/stop" else 1.0
            total = total + value * weight
            loss_dict[key] = value.item()
    return total, loss_dict

