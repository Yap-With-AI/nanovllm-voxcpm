#!/usr/bin/env python3
"""Run LoRA training job with optional publish.

Usage:
    python -m src.scripts.train --args-file PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..hf import run_publish_job
from ..training import run_training_job


def setup_logging() -> None:
    """Configure logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


logger = logging.getLogger(__name__)


def main() -> int:
    """Run LoRA training and optionally publish."""
    parser = argparse.ArgumentParser(description="Run VoxCPM LoRA training")
    parser.add_argument("--args-file", required=True, help="JSON args file")
    cli_args = parser.parse_args()

    setup_logging()

    with open(cli_args.args_file) as f:
        args = json.load(f)

    logger.info("Starting LoRA training with args: %s", args)

    try:
        # Run LoRA training
        checkpoint_dir = run_training_job(
            run_name=args["run_name"],
            base_model_dir=Path(args["base_model_dir"]),
            train_manifest=Path(args["train_manifest"]),
            val_manifest=Path(args["val_manifest"]) if args["val_manifest"] else None,
            checkpoints_dir=Path(args["checkpoints_dir"]),
            logs_dir=Path(args["logs_dir"]),
            batch_size=args["batch_size"],
            learning_rate=args["learning_rate"],
            num_iters=args["num_iters"],
            voice=args["voice"],
            lora_rank=args["lora_rank"],
            lora_alpha=args["lora_alpha"],
            hf_model_id=args.get("hf_model_id", args["base_model_repo"]),
        )

        logger.info("Training complete. Checkpoint: %s", checkpoint_dir)

        # Publish if requested
        if args["upload"] and args["output_repo"]:
            logger.info("Publishing to %s...", args["output_repo"])

            training_params = {
                "batch_size": args["batch_size"],
                "learning_rate": args["learning_rate"],
                "num_iters": args["num_iters"],
                "lora_rank": args["lora_rank"],
                "lora_alpha": args["lora_alpha"],
            }

            run_publish_job(
                run_name=args["run_name"],
                checkpoint_dir=checkpoint_dir,
                base_model_dir=Path(args["base_model_dir"]),
                tokenizer_dir=Path(args["tokenizer_dir"]),
                output_repo=args["output_repo"],
                base_model_repo=args["base_model_repo"],
                training_params=training_params,
            )
            logger.info("Published successfully")

        logger.info("Pipeline complete!")
        return 0

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

