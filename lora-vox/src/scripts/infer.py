# CLI entry point for generating sample audio files.
#
# Usage:
#   python -m src.scripts.infer \
#       --name my-run \
#       --checkpoint-dir artifacts/my-run/checkpoints/latest \
#       --model-dir artifacts/my-run/model \
#       --output-dir artifacts/my-run/samples

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Imports
# ============================================================================

from ..inference import generate_voice_samples, SampleConfig
from ..inference.generator import GeneratorConfig, GeneratorError


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate sample audio files from fine-tuned model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Name of the training run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to base model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated samples",
    )

    # Generation parameters
    parser.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="CFG scale (default: 2.0)",
    )
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="Diffusion inference steps (default: 10)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=600,
        help="Max generation steps (default: 600)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("Generating samples for run: %s", args.name)
    logger.info("Checkpoint: %s", args.checkpoint_dir)
    logger.info("Base model: %s", args.model_dir)
    logger.info("Output: %s", args.output_dir)

    if not args.checkpoint_dir.exists():
        logger.error("Checkpoint directory not found: %s", args.checkpoint_dir)
        return 1

    if not args.model_dir.exists():
        logger.error("Model directory not found: %s", args.model_dir)
        return 1

    generator_config = GeneratorConfig(
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        max_len=args.max_len,
    )

    sample_config = SampleConfig(
        generator_config=generator_config,
    )

    try:
        generated = generate_voice_samples(
            checkpoint_dir=args.checkpoint_dir,
            base_model_dir=args.model_dir,
            output_dir=args.output_dir,
            config=sample_config,
        )
        logger.info("Successfully generated %d samples", len(generated))
        for path in generated:
            logger.info("  - %s", path.name)
        return 0

    except GeneratorError as e:
        logger.error("Sample generation failed: %s", e)
        return 1

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

