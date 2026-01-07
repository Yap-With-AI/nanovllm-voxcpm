#!/usr/bin/env python3
"""Download dataset and model from HuggingFace.

Usage:
    python -m src.scripts.download --dataset-repo REPO --model-repo REPO --cache-dir DIR
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from ..hf import download_dataset, download_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Download dataset and model, print paths for shell capture."""
    parser = argparse.ArgumentParser(description="Download HuggingFace resources")
    parser.add_argument("--dataset-repo", required=True, help="Dataset repository ID")
    parser.add_argument("--model-repo", required=True, help="Model repository ID")
    parser.add_argument("--cache-dir", required=True, help="Cache directory")
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN")

    try:
        logger.info("Downloading dataset: %s", args.dataset_repo)
        dataset_dir = download_dataset(
            repo_id=args.dataset_repo,
            cache_dir=args.cache_dir,
            token=token,
        )
        print(f"DATASET_DIR={dataset_dir}")

        logger.info("Downloading model: %s", args.model_repo)
        model_dir = download_model(
            repo_id=args.model_repo,
            cache_dir=args.cache_dir,
            token=token,
        )
        print(f"MODEL_DIR={model_dir}")

        return 0

    except Exception as exc:
        logger.exception("Download failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

