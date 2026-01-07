"""Model card generation for HuggingFace Hub.

Reads markdown templates and substitutes values to generate model READMEs.
Only supports per-voice LoRA models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import SAMPLE_RATE, VOICES, VOICE_MAP

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _build_voice_table() -> str:
    """Build voice list markdown table.

    For per-voice LoRA mode, voices are in separate subdirectories.
    """
    rows = ["| Voice | Dataset Name | LoRA Path |", "|-------|--------------|-----------|"]
    for voice in VOICES:
        dataset_voice = VOICE_MAP.get(voice, voice)
        rows.append(f"| {voice.title()} | {dataset_voice} | `lora/{voice}/` |")
    return "\n".join(rows)


def _build_params_section(training_params: dict[str, Any]) -> str:
    """Build training parameters section."""
    if not training_params:
        return ""
    rows = [
        "",
        "## Training Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for key, value in training_params.items():
        rows.append(f"| {key} | {value} |")
    return "\n".join(rows)


def generate_model_readme(
    run_name: str,
    base_model_repo: str,
    training_params: dict[str, Any] | None = None,
) -> str:
    """Generate README.md content from template.

    Args:
        run_name: Name of the training run.
        base_model_repo: Base model repository ID.
        training_params: Optional training parameters to document.

    Returns:
        Complete README.md content.
    """
    template_path = TEMPLATES_DIR / "model_card_lora.md"
    template = template_path.read_text(encoding="utf-8")

    substitutions = {
        "{{run_name}}": run_name,
        "{{base_model_repo}}": base_model_repo,
        "{{sample_rate}}": str(SAMPLE_RATE),
        "{{voice_table}}": _build_voice_table(),
        "{{params_section}}": _build_params_section(training_params or {}),
        "{{timestamp}}": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    content = template
    for placeholder, value in substitutions.items():
        content = content.replace(placeholder, value)

    return content


def write_model_readme(
    output_path: Path,
    run_name: str,
    base_model_repo: str,
    training_params: dict[str, Any] | None = None,
) -> Path:
    """Generate and write README.md file.

    Args:
        output_path: Path to write README.md.
        run_name: Name of the training run.
        base_model_repo: Base model repository ID.
        training_params: Optional training parameters.

    Returns:
        Path to the written README file.
    """
    content = generate_model_readme(run_name, base_model_repo, training_params)
    output_path.write_text(content, encoding="utf-8")
    return output_path
