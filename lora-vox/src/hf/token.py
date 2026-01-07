"""HuggingFace authentication token management.

Handles retrieval and validation of HuggingFace API tokens from environment.
"""

from __future__ import annotations

import os

from ..config import HF_TOKEN_ENV_VAR
from ..errors import HuggingFaceError


# ============================================================================
# Public API
# ============================================================================

def get_hf_token(env_var: str = HF_TOKEN_ENV_VAR) -> str | None:
    """Get HuggingFace token from environment.

    Args:
        env_var: Environment variable name containing the token.

    Returns:
        Token string if set, None otherwise.
    """
    return os.environ.get(env_var)


def require_hf_token(env_var: str = HF_TOKEN_ENV_VAR) -> str:
    """Get HuggingFace token, raising if not set.

    Args:
        env_var: Environment variable name containing the token.

    Returns:
        Token string.

    Raises:
        HuggingFaceError: If token environment variable is not set.
    """
    token = get_hf_token(env_var)
    if not token:
        raise HuggingFaceError(
            f"HuggingFace token not found. Set the {env_var} environment variable."
        )
    return token

