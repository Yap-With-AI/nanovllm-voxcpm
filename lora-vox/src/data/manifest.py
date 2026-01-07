"""Dataset manifest generation for VoxCPM training.

Transforms WebDataset shards into Vox-style JSONL manifests.
Each manifest entry includes a 'voice' field for filtering at training time.
"""

from __future__ import annotations

import json
import logging
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ..config import VOICE_MAP, get_public_voice
from ..errors import DatasetError


logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ManifestEntry:
    """Single manifest entry."""

    audio: str
    text: str
    duration: float
    voice: str  # Dataset voice name (tara/zac)

    def to_dict(self) -> dict:
        """Convert to Vox format dict with voice field for filtering."""
        return {
            "audio": self.audio,
            "text": self.text,
            "duration": self.duration,
            "voice": self.voice,
        }


@dataclass
class ManifestStats:
    """Statistics about generated manifest."""

    total_samples: int
    total_duration_hours: float
    samples_per_voice: dict[str, int]
    duration_per_voice_hours: dict[str, float]


# ============================================================================
# Private: Shard Processing
# ============================================================================

def _find_shards(dataset_dir: Path) -> list[Path]:
    """Find WebDataset shards in directory."""
    webdataset_dir = dataset_dir / "webdataset"
    search_dir = webdataset_dir if webdataset_dir.exists() else dataset_dir
    shards = sorted(search_dir.glob("shard-*.tar"))
    return shards if shards else sorted(search_dir.glob("*.tar"))


def _extract_samples(shard_path: Path, extract_dir: Path) -> Iterator[tuple[Path, dict]]:
    """Extract samples from a shard."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(shard_path, "r") as tar:
        members = tar.getnames()
        samples: dict[str, dict[str, str]] = {}

        for name in members:
            if "/" in name:
                name = name.split("/")[-1]
            if name.startswith("."):
                continue
            base = name.rsplit(".", 1)[0]
            ext = name.rsplit(".", 1)[1] if "." in name else ""
            if base not in samples:
                samples[base] = {}
            samples[base][ext] = name

        for key, files in samples.items():
            if "wav" not in files or "json" not in files:
                continue

            audio_path = extract_dir / f"{key}.wav"
            try:
                member = tar.getmember(files["wav"])
                with tar.extractfile(member) as src:
                    if src:
                        audio_path.write_bytes(src.read())
            except KeyError:
                continue

            try:
                member = tar.getmember(files["json"])
                with tar.extractfile(member) as src:
                    if src:
                        metadata = json.loads(src.read().decode("utf-8"))
                        yield audio_path, metadata
            except (KeyError, json.JSONDecodeError):
                continue


def _get_known_dataset_voices() -> set[str]:
    """Get set of known dataset voice names (tara, zac)."""
    return set(VOICE_MAP.values())


def _compute_stats(entries: list[ManifestEntry]) -> ManifestStats:
    """Compute statistics from entries."""
    samples_per_voice: dict[str, int] = {}
    duration_per_voice: dict[str, float] = {}

    for entry in entries:
        # Use public voice name in stats
        try:
            public_voice = get_public_voice(entry.voice)
        except ValueError:
            public_voice = entry.voice
        samples_per_voice[public_voice] = samples_per_voice.get(public_voice, 0) + 1
        duration_per_voice[public_voice] = duration_per_voice.get(public_voice, 0) + entry.duration

    return ManifestStats(
        total_samples=len(entries),
        total_duration_hours=sum(duration_per_voice.values()) / 3600.0,
        samples_per_voice=samples_per_voice,
        duration_per_voice_hours={v: d / 3600.0 for v, d in duration_per_voice.items()},
    )


def _write_manifest(entries: list[ManifestEntry], path: Path) -> None:
    """Write entries to JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d entries to %s", len(entries), path)


# ============================================================================
# Public: Manifest Generation
# ============================================================================

def generate_manifest(
    dataset_dir: Path,
    output_dir: Path,
    val_split: float = 0.0,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> tuple[Path, Optional[Path], ManifestStats]:
    """Generate training manifest from WebDataset shards.

    Creates a single manifest with all voices. Each entry includes a 'voice'
    field that can be used to filter at training time. Text is stored without
    any voice tags - just the plain transcript.

    Args:
        dataset_dir: Directory containing shards.
        output_dir: Output directory for manifests.
        val_split: Validation split fraction (0 = no validation).
        min_duration: Minimum audio duration.
        max_duration: Maximum audio duration.

    Returns:
        Tuple of (train_manifest, val_manifest, stats).

    Raises:
        DatasetError: If no valid samples found.
    """
    shards = _find_shards(dataset_dir)
    if not shards:
        raise DatasetError(f"No shards found in {dataset_dir}")

    logger.info("Found %d shards", len(shards))

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"

    known_voices = _get_known_dataset_voices()
    entries: list[ManifestEntry] = []

    for shard_path in shards:
        logger.info("Processing %s", shard_path.name)
        shard_audio_dir = audio_dir / shard_path.stem

        for audio_path, metadata in _extract_samples(shard_path, shard_audio_dir):
            voice = metadata.get("voice")
            if voice not in known_voices:
                audio_path.unlink(missing_ok=True)
                continue

            duration = metadata.get("duration", 0.0)
            if duration < min_duration or duration > max_duration:
                audio_path.unlink(missing_ok=True)
                continue

            text = metadata.get("text", "").strip()
            if not text:
                audio_path.unlink(missing_ok=True)
                continue

            # Plain text - NO voice tags
            entry = ManifestEntry(
                audio=str(audio_path.absolute()),
                text=text,
                duration=duration,
                voice=voice,
            )
            entries.append(entry)

    if not entries:
        raise DatasetError("No valid samples found")

    logger.info("Collected %d samples", len(entries))
    stats = _compute_stats(entries)

    train_entries = entries
    val_entries: list[ManifestEntry] = []

    if val_split > 0:
        random.seed(42)
        random.shuffle(entries)
        split_idx = int(len(entries) * (1 - val_split))
        train_entries = entries[:split_idx]
        val_entries = entries[split_idx:]

    train_manifest = output_dir / "train.jsonl"
    _write_manifest(train_entries, train_manifest)

    val_manifest: Optional[Path] = None
    if val_entries:
        val_manifest = output_dir / "val.jsonl"
        _write_manifest(val_entries, val_manifest)

    return train_manifest, val_manifest, stats


def log_manifest_stats(stats: ManifestStats) -> None:
    """Log manifest statistics using public voice names."""
    logger.info("Total: %d samples, %.2f hours", stats.total_samples, stats.total_duration_hours)
    for voice in sorted(stats.samples_per_voice.keys()):
        logger.info(
            "  %s: %d samples, %.2f hours",
            voice,
            stats.samples_per_voice[voice],
            stats.duration_per_voice_hours[voice],
        )
