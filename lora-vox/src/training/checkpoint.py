"""Checkpoint saving and loading for VoxCPM LoRA training.

Handles LoRA checkpoint formats with support for safetensors and PyTorch binary formats.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


logger = logging.getLogger(__name__)


# ============================================================================
# Safetensors Support
# ============================================================================

_safetensors_checked = False
_safetensors_available = False


def _check_safetensors() -> bool:
    """Check if safetensors is available (lazy, logged once)."""
    global _safetensors_checked, _safetensors_available
    if not _safetensors_checked:
        try:
            import safetensors.torch  # noqa: F401
            _safetensors_available = True
        except ImportError:
            _safetensors_available = False
            logger.warning("safetensors not available, using pytorch format")
        _safetensors_checked = True
    return _safetensors_available


# ============================================================================
# Private: File Operations
# ============================================================================

def _save_model_weights(
    state_dict: dict[str, torch.Tensor],
    folder: Path,
    filename: str,
) -> Path:
    """Save model weights in best available format."""
    if _check_safetensors():
        from safetensors.torch import save_file
        path = folder / f"{filename}.safetensors"
        save_file(state_dict, str(path))
    else:
        path = folder / f"{filename}.bin"
        torch.save({"state_dict": state_dict}, str(path))
    return path


def _load_model_weights(folder: Path, filename: str) -> dict[str, torch.Tensor]:
    """Load model weights from best available format."""
    safetensors_path = folder / f"{filename}.safetensors"
    bin_path = folder / f"{filename}.bin"

    if safetensors_path.exists() and _check_safetensors():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    elif bin_path.exists():
        checkpoint = torch.load(str(bin_path), map_location="cpu")
        return checkpoint.get("state_dict", checkpoint)
    else:
        raise FileNotFoundError(f"No weights found at {folder}/{filename}.*")


def _copy_tokenizer_files(tokenizer_dir: Path, target_dir: Path) -> None:
    """Copy tokenizer files (with voice tags) to checkpoint."""
    tokenizer_files = [
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    copied = []
    for filename in tokenizer_files:
        src = tokenizer_dir / filename
        if src.exists():
            shutil.copy2(src, target_dir / filename)
            copied.append(filename)
    if copied:
        logger.info("Copied tokenizer files: %s", ", ".join(copied))


def _update_latest_symlink(save_dir: Path, target_folder: Path) -> None:
    """Update 'latest' symlink to point to most recent checkpoint."""
    latest_link = save_dir / "latest"

    try:
        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        os.symlink(str(target_folder), str(latest_link))
    except OSError:
        # Symlink failed (e.g., Windows), fall back to copy
        try:
            if latest_link.exists():
                if latest_link.is_dir():
                    shutil.rmtree(latest_link)
                else:
                    latest_link.unlink()
            shutil.copytree(target_folder, latest_link)
        except Exception as exc:
            logger.warning("Failed to update latest checkpoint: %s", exc)


# ============================================================================
# Public: Save Checkpoint
# ============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    save_dir: Path,
    step: int,
    base_model_dir: Path | None = None,
    hf_model_id: str = "",
    tokenizer_dir: Path | None = None,
) -> Path:
    """Save LoRA training checkpoint.

    Saves only LoRA weights and config.

    Args:
        model: Model to save (may be DDP-wrapped).
        optimizer: Optimizer state to save.
        scheduler: Scheduler state to save.
        save_dir: Directory for checkpoints.
        step: Current training step.
        base_model_dir: Path to base model (for config reference).
        hf_model_id: HuggingFace model ID for LoRA config.
        tokenizer_dir: Path to tokenizer (for LoRA checkpoints).

    Returns:
        Path to checkpoint folder.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    tag = "latest" if step == 0 else f"step_{step:07d}"
    folder = save_dir / tag
    folder.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed
    unwrapped = model.module if hasattr(model, "module") else model
    full_state = unwrapped.state_dict()

    # Save only LoRA weights
    lora_state = {k: v for k, v in full_state.items() if "lora_" in k}
    _save_model_weights(lora_state, folder, "lora_weights")

    # Save LoRA config
    lora_cfg = getattr(unwrapped, "lora_config", None)
    lora_info: dict[str, Any] = {
        "base_model": hf_model_id or (str(base_model_dir) if base_model_dir else None),
        "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg) if lora_cfg else {},
    }
    with open(folder / "lora_config.json", "w", encoding="utf-8") as f:
        json.dump(lora_info, f, indent=2, ensure_ascii=False)

    # Copy tokenizer to checkpoint
    if tokenizer_dir and tokenizer_dir.exists():
        _copy_tokenizer_files(tokenizer_dir, folder)

    logger.info("Saved LoRA checkpoint: %s", folder)

    # Save optimizer and scheduler state
    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")

    # Save training state
    training_state = {"step": step}
    with open(folder / "training_state.json", "w", encoding="utf-8") as f:
        json.dump(training_state, f)

    _update_latest_symlink(save_dir, folder)

    return folder


# ============================================================================
# Public: Load Checkpoint
# ============================================================================

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    save_dir: Path,
) -> int:
    """Load LoRA checkpoint and return step to resume from.

    Args:
        model: Model to load weights into (may be DDP-wrapped).
        optimizer: Optimizer to restore state.
        scheduler: Scheduler to restore state.
        save_dir: Directory containing checkpoints.

    Returns:
        Step number to resume from, or 0 if no checkpoint found.
    """
    latest_folder = save_dir / "latest"
    if not latest_folder.exists():
        return 0

    unwrapped = model.module if hasattr(model, "module") else model

    try:
        # Load LoRA weights
        state_dict = _load_model_weights(latest_folder, "lora_weights")
        unwrapped.load_state_dict(state_dict, strict=False)
        logger.info("Loaded LoRA weights from %s", latest_folder)
    except FileNotFoundError as exc:
        logger.warning("Could not load model weights: %s", exc)
        return 0

    # Load optimizer state
    optimizer_path = latest_folder / "optimizer.pth"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        logger.info("Loaded optimizer state")

    # Load scheduler state
    scheduler_path = latest_folder / "scheduler.pth"
    if scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        logger.info("Loaded scheduler state")

    # Get resume step
    training_state_path = latest_folder / "training_state.json"
    if training_state_path.exists():
        with open(training_state_path) as f:
            training_state = json.load(f)
            return training_state.get("step", 0)

    # Fallback: infer from folder names
    step_folders = sorted(
        [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )
    if step_folders:
        return int(step_folders[0].name.split("_")[1])

    return 0


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    """Find latest checkpoint directory.

    Args:
        save_dir: Directory containing checkpoints.

    Returns:
        Path to latest checkpoint, or None if none found.
    """
    if not save_dir.exists():
        return None

    latest = save_dir / "latest"
    if latest.exists():
        return latest.resolve() if latest.is_symlink() else latest

    step_folders = sorted(
        [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
        reverse=True,
    )
    return step_folders[0] if step_folders else None

