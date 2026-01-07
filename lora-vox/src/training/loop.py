"""Training loop for VoxCPM fine-tuning.

Orchestrates the training process including forward pass, validation,
logging, checkpointing, and signal handling for graceful shutdown.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..config import (
    AUDIO_VAE_FPS,
    DEFAULT_FEAT_DIM,
    DEFAULT_MAX_LENGTH,
    DEFAULT_PATCH_SIZE,
    MAX_VALIDATION_BATCHES,
)
from ..state import TrainingConfig
from .accelerator import Accelerator, get_gpu_name
from .batch import BatchProcessor, BatchProcessorConfig, compute_loss
from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import build_training_dataloader
from .manifest import compute_sequence_lengths, load_train_val_datasets

if TYPE_CHECKING:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def run_training_loop(
    model: torch.nn.Module,
    tokenizer: Any,
    audio_vae: torch.nn.Module,
    config: TrainingConfig,
) -> Path:
    """Run the complete training loop.

    Args:
        model: VoxCPM model (with LoRA applied if needed).
        tokenizer: Text tokenizer callable.
        audio_vae: AudioVAE model for encoding audio.
        config: Training configuration.

    Returns:
        Path to final checkpoint directory.
    """
    accelerator = Accelerator(amp=True)
    save_dir = config.save_path

    _setup_directories(save_dir, config.tensorboard_path, accelerator)
    writer = _create_tensorboard_writer(config.tensorboard_path or save_dir / "logs", accelerator)

    if accelerator.rank == 0:
        logger.info("GPU: %s, dtype: %s", get_gpu_name(), accelerator.dtype)

    # Prepare data and model
    train_ds, val_ds, voice_count = _prepare_datasets(config, tokenizer, model, audio_vae, accelerator)
    train_loader, val_loader = _build_dataloaders(train_ds, val_ds, config, accelerator)
    batch_processor = _create_batch_processor(model, audio_vae, voice_count, accelerator)

    # Detach audio_vae from model
    audio_vae_ref = model.audio_vae
    model.audio_vae = None

    model = accelerator.prepare_model(model)
    unwrapped = accelerator.unwrap(model)
    unwrapped.train()

    if accelerator.rank == 0:
        _log_trainable_params(model)

    # Optimizer and scheduler
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.num_iters)

    start_step = _resume_training(model, optimizer, scheduler, save_dir, accelerator)
    # WARNING: Mutable dict used for signal handler state sharing - necessary for graceful shutdown
    step_ref = {"step": start_step}

    _setup_signal_handlers(lambda: _checkpoint(model, optimizer, scheduler, save_dir, step_ref, config, accelerator), step_ref)

    # Execute loop
    _run_loop(model, unwrapped, train_loader, val_loader, batch_processor, optimizer, scheduler, accelerator, config, writer, start_step, step_ref, len(train_ds), save_dir)

    if writer:
        writer.close()

    unwrapped.audio_vae = audio_vae_ref
    return save_dir / "latest"


# ============================================================================
# Private: Setup
# ============================================================================

def _setup_directories(save_dir: Path, tb_dir: Path | None, accel: Accelerator) -> None:
    if accel.rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        (tb_dir or save_dir / "logs").mkdir(parents=True, exist_ok=True)
    accel.barrier()


def _create_tensorboard_writer(log_dir: Path, accel: Accelerator) -> SummaryWriter | None:
    if accel.rank != 0:
        return None
    try:
        from tensorboardX import SummaryWriter as TBWriter
        return TBWriter(log_dir=str(log_dir))
    except ImportError:
        logger.warning("tensorboardX not available")
        return None


def _prepare_datasets(config: TrainingConfig, tokenizer: Any, model: torch.nn.Module, audio_vae: torch.nn.Module, accel: Accelerator) -> tuple[Any, Any | None, int]:
    # Pass voice filter if training a specific voice LoRA
    train_ds, val_ds = load_train_val_datasets(
        config.train_manifest,
        config.val_manifest,
        config.sample_rate,
        voice=config.voice if config.voice else None,
    )

    def tokenize(batch: dict[str, Any]) -> dict[str, list[Any]]:
        return {"text_ids": [tokenizer(t) for t in batch["text"]]}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    if val_ds:
        val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    if config.max_batch_tokens > 0:
        # Compute audio VAE FPS dynamically from model
        model_cfg = getattr(model, "config", None)
        audio_vae_fps = audio_vae.sample_rate / audio_vae.hop_length if hasattr(audio_vae, "hop_length") else AUDIO_VAE_FPS
        patch_size = model_cfg.patch_size if model_cfg else DEFAULT_PATCH_SIZE
        lengths = compute_sequence_lengths(train_ds, audio_vae_fps=audio_vae_fps, patch_size=patch_size)
        max_len = config.max_batch_tokens // config.batch_size if config.batch_size > 0 else max(lengths)
        keep = [i for i, l in enumerate(lengths) if l <= max_len]
        if len(keep) < len(train_ds) and accel.rank == 0:
            logger.info("Filtering %d/%d samples exceeding %d tokens", len(train_ds) - len(keep), len(train_ds), max_len)
        train_ds = train_ds.select(keep)

    # For per-voice LoRA training, each LoRA handles a single voice
    voice_count = 1
    if accel.rank == 0:
        voice_info = f" for voice '{config.voice}'" if config.voice else ""
        logger.info("Training samples: %d%s", len(train_ds), voice_info)
    return train_ds, val_ds, voice_count


def _build_dataloaders(train_ds: Any, val_ds: Any | None, config: TrainingConfig, accel: Accelerator) -> tuple[DataLoader, DataLoader | None]:
    train_loader = build_training_dataloader(train_ds, accel, config.batch_size, config.num_workers, drop_last=True)
    val_loader = build_training_dataloader(val_ds, accel, config.batch_size, config.num_workers, drop_last=False) if val_ds else None
    return train_loader, val_loader


def _create_batch_processor(model, audio_vae, voice_count: int, accel: Accelerator) -> BatchProcessor:
    model_cfg = getattr(model, "config", None)
    cfg = BatchProcessorConfig(
        max_length=model_cfg.max_length if model_cfg else DEFAULT_MAX_LENGTH,
        patch_size=model_cfg.patch_size if model_cfg else DEFAULT_PATCH_SIZE,
        feat_dim=model_cfg.feat_dim if model_cfg else DEFAULT_FEAT_DIM,
    )
    return BatchProcessor(cfg, audio_vae, voice_count, accel.device)


def _log_trainable_params(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)


def _resume_training(model, optimizer, scheduler, save_dir: Path, accel: Accelerator) -> int:
    start = 0
    if accel.rank == 0:
        start = load_checkpoint(model, optimizer, scheduler, save_dir)
    t = torch.tensor(start, device=accel.device)
    accel.broadcast(t, src=0)
    start = int(t.item())
    if start > 0 and accel.rank == 0:
        logger.info("Resuming from step %d", start)
    return start


def _setup_signal_handlers(save_fn: Callable[[], None], step_ref: dict[str, int]) -> None:
    def handler(signum: int, frame: Any) -> None:
        logger.info("Signal %d at step %d, saving...", signum, step_ref.get("step", 0))
        try:
            save_fn()
        except Exception as e:
            logger.error("Error: %s", e)
        os._exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _checkpoint(model: torch.nn.Module, optimizer: Any, scheduler: Any, save_dir: Path, step_ref: dict[str, int], config: TrainingConfig, accel: Accelerator) -> None:
    if accel.rank == 0:
        # Tokenizer is at artifacts/<run>/tokenizer/ (sibling to checkpoints)
        tokenizer_dir = save_dir.parent / "tokenizer"
        save_checkpoint(model, optimizer, scheduler, save_dir, step_ref["step"], config.pretrained_path, config.hf_model_id, tokenizer_dir)


# ============================================================================
# Private: Training Execution
# ============================================================================

def _run_loop(
    model: torch.nn.Module,
    unwrapped: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    batch_processor: BatchProcessor,
    optimizer: Any,
    scheduler: Any,
    accel: Accelerator,
    config: TrainingConfig,
    writer: SummaryWriter | None,
    start_step: int,
    step_ref: dict[str, int],
    num_samples: int,
    save_dir: Path,
) -> None:
    """Execute the main training loop."""
    grad_accum = max(config.grad_accum_steps, 1)
    train_iter, epoch = iter(train_loader), 0

    def next_batch():
        nonlocal train_iter, epoch
        try:
            return next(train_iter)
        except StopIteration:
            epoch += 1
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    if accel.rank == 0:
        logger.info("Starting training for %d iterations", config.num_iters)

    for step in range(start_step, config.num_iters):
        step_ref["step"] = step
        optimizer.zero_grad(set_to_none=True)

        loss_dict = _accumulate_grads(model, batch_processor, next_batch, grad_accum, accel, config, step)

        if hasattr(accel.scaler, "unscale_"):
            accel.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), max_norm=1e9)

        accel.step(optimizer)
        accel.update()
        scheduler.step()

        if (step % config.log_interval == 0 or step == config.num_iters - 1) and accel.rank == 0:
            _log_step(writer, loss_dict, optimizer, step, grad_norm, num_samples, config, grad_accum)

        if val_loader and (step % config.valid_interval == 0 or step == config.num_iters - 1):
            _validate(model, val_loader, batch_processor, accel, config, writer, step)

        if (step % config.save_interval == 0 or step == config.num_iters - 1) and accel.rank == 0:
            tokenizer_dir = save_dir.parent / "tokenizer"
            save_checkpoint(model, optimizer, scheduler, save_dir, step, config.pretrained_path, config.hf_model_id, tokenizer_dir)

    if accel.rank == 0:
        tokenizer_dir = save_dir.parent / "tokenizer"
        save_checkpoint(model, optimizer, scheduler, save_dir, config.num_iters, config.pretrained_path, config.hf_model_id, tokenizer_dir)
        logger.info("Training complete. Final checkpoint: %s", save_dir / "latest")


def _accumulate_grads(
    model: torch.nn.Module,
    batch_processor: BatchProcessor,
    get_batch: Callable[[], dict[str, Any]],
    grad_accum: int,
    accel: Accelerator,
    config: TrainingConfig,
    step: int,
) -> dict[str, float]:
    """Accumulate gradients over multiple micro-batches."""
    loss_dict = {}
    for i in range(grad_accum):
        batch = get_batch()
        proc = batch_processor(batch)
        is_last = i == grad_accum - 1
        ctx = contextlib.nullcontext() if is_last else accel.no_sync()

        with ctx, accel.autocast():
            outputs = model(proc["text_tokens"], proc["text_mask"], proc["audio_feats"], proc["audio_mask"], proc["loss_mask"], proc["position_ids"], proc["labels"], progress=step / max(1, config.num_iters))

        loss, loss_dict = compute_loss(outputs, config.loss_diff_weight, config.loss_stop_weight)
        accel.backward(loss / grad_accum)
    return loss_dict


def _log_step(
    writer: SummaryWriter | None,
    loss_dict: dict[str, float],
    optimizer: Any,
    step: int,
    grad_norm: torch.Tensor | float,
    num_samples: int,
    config: TrainingConfig,
    grad_accum: int,
) -> None:
    """Log training metrics to TensorBoard and console."""
    metrics = {**loss_dict, "lr": optimizer.param_groups[0]["lr"], "epoch": (step * grad_accum * config.batch_size) / max(1, num_samples), "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm}
    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", v, step)
    logger.info("step=%d, %s, lr=%.2e", step, ", ".join(f"{k}={v:.4f}" for k, v in loss_dict.items()), metrics["lr"])


def _validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    batch_processor: BatchProcessor,
    accel: Accelerator,
    config: TrainingConfig,
    writer: SummaryWriter | None,
    step: int,
) -> None:
    """Run validation on a subset of the validation set."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= MAX_VALIDATION_BATCHES:
                break
            proc = batch_processor(batch)
            with accel.autocast():
                outputs = model(proc["text_tokens"], proc["text_mask"], proc["audio_feats"], proc["audio_mask"], proc["loss_mask"], proc["position_ids"], proc["labels"], progress=0.0)
            loss, _ = compute_loss(outputs, config.loss_diff_weight, config.loss_stop_weight)
            losses.append(loss.detach())
    model.train()
    if losses:
        mean = torch.stack(losses).mean()
        accel.all_reduce(mean)
        if accel.rank == 0:
            if writer:
                writer.add_scalar("val/loss/total", mean.item(), step)
            logger.info("step=%d [val] loss=%.4f", step, mean.item())
