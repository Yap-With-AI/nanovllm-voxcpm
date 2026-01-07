#!/usr/bin/env python3
"""Publish per-voice LoRA model to HuggingFace.

Usage:
    python -m src.scripts.publish \
        --name NAME \
        --checkpoint-dir DIR \
        --model-dir DIR \
        --tokenizer-dir DIR \
        --output-repo REPO \
        --base-model-repo REPO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..hf import run_publish_job


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Publish LoRA model to HuggingFace Hub."""
    parser = argparse.ArgumentParser(description="Publish LoRA model to HuggingFace")
    parser.add_argument("--name", required=True, help="Run name")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    parser.add_argument("--model-dir", required=True, help="Base model directory")
    parser.add_argument("--tokenizer-dir", required=True, help="Tokenizer directory")
    parser.add_argument("--output-repo", required=True, help="Output repository")
    parser.add_argument("--base-model-repo", required=True, help="Base model repository")
    parser.add_argument("--samples-dir", help="Directory containing voice samples")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir) if args.samples_dir else None

    try:
        run_publish_job(
            run_name=args.name,
            checkpoint_dir=Path(args.checkpoint_dir),
            base_model_dir=Path(args.model_dir),
            tokenizer_dir=Path(args.tokenizer_dir),
            output_repo=args.output_repo,
            base_model_repo=args.base_model_repo,
            samples_dir=samples_dir,
        )
        logger.info("Published to: https://huggingface.co/%s", args.output_repo)
        return 0

    except Exception as exc:
        logger.exception("Publish failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

