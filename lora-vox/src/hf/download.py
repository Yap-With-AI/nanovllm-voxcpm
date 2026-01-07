"""HuggingFace Hub download operations.

Downloads datasets and models from HuggingFace Hub with caching support.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from ..errors import HuggingFaceError


logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def download_dataset(
    repo_id: str,
    cache_dir: str,
    *,
    token: str | None = None,
) -> Path:
    """Download dataset from HuggingFace Hub.

    Args:
        repo_id: Dataset repository ID (e.g., "user/dataset-name").
        cache_dir: Local directory for caching downloads.
        token: HuggingFace API token for private repositories.

    Returns:
        Path to the downloaded dataset directory.

    Raises:
        HuggingFaceError: If repository not found or download fails.
    """
    logger.info("Downloading dataset: %s", repo_id)

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            token=token,
        )
    except RepositoryNotFoundError as exc:
        raise HuggingFaceError(f"Dataset repository not found: {repo_id}") from exc
    except Exception as exc:
        raise HuggingFaceError(f"Dataset download failed: {exc}") from exc

    logger.info("Dataset downloaded to: %s", path)
    return Path(path)


def download_model(
    repo_id: str,
    cache_dir: str,
    *,
    token: str | None = None,
) -> Path:
    """Download model from HuggingFace Hub.

    Args:
        repo_id: Model repository ID (e.g., "openbmb/VoxCPM1.5").
        cache_dir: Local directory for caching downloads.
        token: HuggingFace API token for private repositories.

    Returns:
        Path to the downloaded model directory.

    Raises:
        HuggingFaceError: If repository not found or download fails.
    """
    logger.info("Downloading model: %s", repo_id)

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            cache_dir=cache_dir,
            token=token,
        )
    except RepositoryNotFoundError as exc:
        raise HuggingFaceError(f"Model repository not found: {repo_id}") from exc
    except Exception as exc:
        raise HuggingFaceError(f"Model download failed: {exc}") from exc

    logger.info("Model downloaded to: %s", path)
    return Path(path)

