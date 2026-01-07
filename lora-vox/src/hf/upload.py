"""HuggingFace Hub upload operations.

Creates repositories and uploads model artifacts to HuggingFace Hub.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from ..errors import HuggingFaceError


logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def create_repo_if_needed(
    repo_id: str,
    token: str,
    *,
    private: bool = False,
) -> bool:
    """Create HuggingFace model repository if it does not exist.

    Args:
        repo_id: Repository ID (e.g., "user/model-name").
        token: HuggingFace API token with write access.
        private: Whether to create a private repository.

    Returns:
        True if repository was created, False if it already existed.
    """
    api = HfApi(token=token)

    try:
        api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        logger.info("Repository already exists: %s", repo_id)
        return False
    except RepositoryNotFoundError:
        pass

    logger.info("Creating repository: %s", repo_id)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    return True


def upload_folder(
    folder_path: Path,
    repo_id: str,
    token: str,
    *,
    commit_message: str = "Upload model",
) -> str:
    """Upload folder contents to HuggingFace Hub repository.

    Args:
        folder_path: Local folder containing files to upload.
        repo_id: Target repository ID.
        token: HuggingFace API token with write access.
        commit_message: Git commit message for the upload.

    Returns:
        URL of the commit on HuggingFace Hub.

    Raises:
        HuggingFaceError: If upload fails.
    """
    logger.info("Uploading folder to: %s", repo_id)
    api = HfApi(token=token)

    try:
        commit_info = api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    except Exception as exc:
        raise HuggingFaceError(f"Upload failed: {exc}") from exc

    commit_url = getattr(commit_info, "commit_url", str(commit_info))
    logger.info("Upload complete: %s", commit_url)
    return commit_url

