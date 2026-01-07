#!/usr/bin/env python3
"""Prepare training manifest and copy base tokenizer.

Usage:
    python -m src.scripts.prepare \
        --name NAME \
        --dataset-repo REPO \
        --dataset-dir DIR \
        --model-dir DIR \
        --output-dir DIR \
        --val-split 0.05
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ..config import TOKENIZER_FILES
from ..data import generate_manifest, log_manifest_stats
from ..errors import TokenizerError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_manifest(
    dataset_dir: Path,
    output_dir: Path,
    val_split: float,
) -> tuple[Path, Path | None]:
    """Generate training manifest from dataset.

    Creates a single manifest with all voices. Each entry has a 'voice' field
    for filtering during training. Text is stored without voice tags.
    """
    manifests_dir = output_dir / "manifests"
    train_manifest, val_manifest, stats = generate_manifest(
        dataset_dir=dataset_dir,
        output_dir=manifests_dir,
        val_split=val_split,
    )

    log_manifest_stats(stats)
    return train_manifest, val_manifest


def prepare_tokenizer(model_dir: Path, output_dir: Path) -> Path:
    """Copy base model tokenizer.

    Since we use separate LoRAs per voice (no voice tags in text),
    we just copy the base tokenizer as-is.
    """
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    for filename in TOKENIZER_FILES:
        src = model_dir / filename
        if not src.exists():
            raise TokenizerError(f"Missing tokenizer file: {src}")
        shutil.copy2(src, tokenizer_dir / filename)

    logger.info("Copied base tokenizer to %s", tokenizer_dir)
    return tokenizer_dir


def main() -> int:
    """Prepare manifest and tokenizer, print paths for shell capture."""
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--name", required=True, help="Run name")
    parser.add_argument("--dataset-repo", required=True, help="Dataset repository ID")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split")
    args = parser.parse_args()

    try:
        train_manifest, val_manifest = prepare_manifest(
            dataset_dir=Path(args.dataset_dir),
            output_dir=Path(args.output_dir),
            val_split=args.val_split,
        )
        print(f"TRAIN_MANIFEST={train_manifest}")
        if val_manifest:
            print(f"VAL_MANIFEST={val_manifest}")

        tokenizer_dir = prepare_tokenizer(
            model_dir=Path(args.model_dir),
            output_dir=Path(args.output_dir),
        )
        print(f"TOKENIZER_DIR={tokenizer_dir}")

        return 0

    except Exception as exc:
        logger.exception("Preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
