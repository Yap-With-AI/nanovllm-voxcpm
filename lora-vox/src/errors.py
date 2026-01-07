"""Custom exceptions for VoxCPM 1.5 fine-tuning pipeline.

Centralizes all exception classes for the training module, enabling callers
to branch on intent and providing consistent error context.
"""

from __future__ import annotations


class VoxCPMTrainingError(Exception):
    """Base exception for all VoxCPM training errors."""

    pass


class ConfigurationError(VoxCPMTrainingError):
    """Raised when pipeline configuration is invalid or missing required fields."""

    pass


class DatasetError(VoxCPMTrainingError):
    """Raised when dataset loading or manifest generation fails."""

    pass


class TokenizerError(VoxCPMTrainingError):
    """Raised when tokenizer modification or saving fails."""

    pass


class TrainerError(VoxCPMTrainingError):
    """Raised when training execution fails."""

    pass


class HuggingFaceError(VoxCPMTrainingError):
    """Raised when HuggingFace Hub operations fail."""

    pass


class PublishError(VoxCPMTrainingError):
    """Raised when artifact packaging or publishing fails."""

    pass


class GPUError(VoxCPMTrainingError):
    """Raised when GPU requirements are not met."""

    pass

