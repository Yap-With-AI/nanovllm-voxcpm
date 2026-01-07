"""Simplified accelerator for distributed training and mixed precision.

Handles DDP initialization, AMP scaling, and gradient synchronization for
single-GPU and multi-GPU training scenarios.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

if TYPE_CHECKING:
    from torch.optim import Optimizer


# ============================================================================
# GPU Detection
# ============================================================================

def detect_optimal_dtype() -> torch.dtype:
    """Detect optimal dtype based on GPU capability.

    Returns:
        torch.bfloat16 for Ampere+ GPUs (sm_80+), torch.float16 otherwise.
    """
    if not torch.cuda.is_available():
        return torch.float32

    capability = torch.cuda.get_device_capability()
    major, minor = capability

    # Ampere (sm_80) and newer support efficient bf16
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def get_gpu_name() -> str:
    """Get current GPU name or 'cpu' if no GPU available."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


# ============================================================================
# Dummy Scaler for Non-AMP Training
# ============================================================================

class _DummyScaler:
    """No-op scaler when AMP is disabled."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: Optimizer) -> None:
        pass

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass


# ============================================================================
# Accelerator
# ============================================================================

class Accelerator:
    """Simplified training accelerator for DDP and AMP.

    Initializes distributed process group when launched via torchrun.
    Provides helpers for model preparation, gradient scaling, and
    data loading with distributed sampling.

    Attributes:
        rank: Global rank in distributed setup (0 for single-GPU).
        local_rank: Local rank on current node.
        world_size: Total number of processes.
        device: Target device for this process.
        amp: Whether AMP is enabled.
        dtype: Optimal dtype for this GPU (bf16/fp16/fp32).
    """

    def __init__(self, amp: bool = True, seed: int = 42) -> None:
        """Initialize accelerator.

        Args:
            amp: Enable automatic mixed precision.
            seed: Random seed for reproducibility.
        """
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.amp = amp and torch.cuda.is_available()

        self._set_seed(seed)

        self.dtype = detect_optimal_dtype()
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.amp else _DummyScaler()
        )
        self._ddp_model: DistributedDataParallel | None = None

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility across all processes."""
        torch.manual_seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        random.seed(seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + self.rank)

    @property
    def device(self) -> torch.device:
        """Get target device for this process."""
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Distributed Helpers
    # ------------------------------------------------------------------ #

    def barrier(self) -> None:
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()

    def all_reduce(
        self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.AVG
    ) -> torch.Tensor:
        """All-reduce tensor across processes."""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all processes."""
        if dist.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor

    # ------------------------------------------------------------------ #
    # Model Preparation
    # ------------------------------------------------------------------ #

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model for distributed training.

        Moves model to device and wraps with DDP if multi-GPU.

        Args:
            model: Model to prepare.

        Returns:
            Prepared model (possibly wrapped in DDP).
        """
        if hasattr(model, "device"):
            model.device = self.device

        model = model.to(self.device)

        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )
            self._ddp_model = model

        return model

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap DDP module to access underlying model."""
        return model.module if hasattr(model, "module") else model

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        """Context manager to skip gradient sync during accumulation."""
        if self._ddp_model is not None:
            with self._ddp_model.no_sync():
                yield
        else:
            yield

    # ------------------------------------------------------------------ #
    # AMP Helpers
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        """Context manager for automatic mixed precision."""
        if self.amp:
            with torch.amp.autocast("cuda", dtype=self.dtype):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient scaling."""
        self.scaler.scale(loss).backward()

    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step with gradient unscaling."""
        self.scaler.step(optimizer)

    def update(self) -> None:
        """Update gradient scaler."""
        self.scaler.update()

    # ------------------------------------------------------------------ #
    # DataLoader Preparation
    # ------------------------------------------------------------------ #

    def prepare_dataloader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_workers: int = 2,
        shuffle: bool = True,
        collate_fn: Callable | None = None,
        drop_last: bool = False,
    ) -> DataLoader:
        """Create DataLoader with distributed sampling if needed.

        Args:
            dataset: Dataset to load from.
            batch_size: Batch size per process.
            num_workers: Number of data loading workers.
            shuffle: Whether to shuffle data.
            collate_fn: Custom collation function.
            drop_last: Drop incomplete final batch.

        Returns:
            Configured DataLoader.
        """
        sampler: DistributedSampler | None = None

        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )

