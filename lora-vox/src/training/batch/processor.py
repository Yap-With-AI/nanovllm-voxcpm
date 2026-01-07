"""Batch processing for VoxCPM training.

Transforms raw audio and text into the packed format expected by VoxCPMModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from ...config import (
    AUDIO_END_TOKEN_ID,
    AUDIO_PROMPT_END_TOKEN_ID,
    AUDIO_PROMPT_START_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    AUDIO_VAE_FPS,
    DEFAULT_FEAT_DIM,
    DEFAULT_MAX_LENGTH,
    DEFAULT_PATCH_SIZE,
)
from .utils import (
    build_position_ids,
    create_empty_batch,
    pad_1d,
    pad_3d,
    remove_padding,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class BatchProcessorConfig:
    """Configuration for batch processing."""

    max_length: int = DEFAULT_MAX_LENGTH
    patch_size: int = DEFAULT_PATCH_SIZE
    feat_dim: int = DEFAULT_FEAT_DIM


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor:
    """Transforms raw batches into VoxCPM training format.

    Encodes audio through AudioVAE and packs text/audio into the
    interleaved sequence format expected by the model.
    """

    def __init__(
        self,
        config: BatchProcessorConfig,
        audio_vae: nn.Module,
        voice_count: int,
        device: torch.device,
    ) -> None:
        """Initialize batch processor.

        Args:
            config: Processing configuration.
            audio_vae: AudioVAE model for encoding audio.
            voice_count: Number of distinct voices/datasets.
            device: Target device for tensors.
        """
        self._max_length = config.max_length
        self._patch_size = config.patch_size
        self._feat_dim = config.feat_dim
        self._voice_count = max(voice_count, 1)
        self._device = device

        self._audio_vae = audio_vae.to(device)
        self._patch_len = audio_vae.hop_length * self._patch_size

    def __call__(
        self,
        batch: dict[str, torch.Tensor | list[bool]],
    ) -> dict[str, torch.Tensor]:
        """Process batch for training.

        Args:
            batch: Collated batch from DataLoader containing:
                - audio_tokens: [B, T_audio] raw audio waveforms
                - text_tokens: [B, T_text] tokenized text
                - task_ids: [B] task type indicators
                - dataset_ids: [B] voice/dataset indicators
                - is_prompts: [B] prompt flags

        Returns:
            Dict with packed tensors for model forward pass.
        """
        audio_tokens = batch["audio_tokens"].to(self._device)
        text_tokens = batch["text_tokens"].to(self._device)
        dataset_ids = batch["dataset_ids"].to(self._device)
        is_prompts = batch["is_prompts"]

        # Track per-voice statistics
        max_voice_id = int(dataset_ids.max().item()) if dataset_ids.numel() > 0 else -1
        voice_count = max(self._voice_count, max_voice_id + 1)

        audio_duration = torch.zeros(voice_count, dtype=torch.float32, device=self._device)
        text_count = torch.zeros(voice_count, dtype=torch.float32, device=self._device)

        # Process each sample
        packed_samples = []
        for audio, text, voice_id, is_prompt in zip(
            audio_tokens, text_tokens, dataset_ids.tolist(), is_prompts
        ):
            audio_unpad = remove_padding(audio).float()
            text_unpad = remove_padding(text)

            sample = self._process_single_sample(audio_unpad, text_unpad, is_prompt)
            sample["voice_id"] = voice_id

            audio_duration[voice_id] += sample["duration"]
            text_count[voice_id] += sample["text_length"]
            packed_samples.append(sample)

        return self._stack_samples(packed_samples, audio_duration, text_count)

    # ------------------------------------------------------------------ #
    # Sample Processing
    # ------------------------------------------------------------------ #

    def _process_single_sample(
        self,
        audio: torch.Tensor,
        text: torch.Tensor,
        is_prompt: bool,
    ) -> dict[str, torch.Tensor | float | int]:
        """Process a single TTS sample into packed format."""
        start_id = AUDIO_PROMPT_START_TOKEN_ID if is_prompt else AUDIO_START_TOKEN_ID
        end_id = AUDIO_PROMPT_END_TOKEN_ID if is_prompt else AUDIO_END_TOKEN_ID

        # Prepend text with start token
        text_with_start = torch.cat([
            text,
            torch.tensor([start_id], dtype=torch.int32, device=text.device),
        ])
        text_length = len(text)
        prefix_length = text_with_start.shape[0]

        # Encode audio
        audio_feats, duration = self._encode_audio(audio)
        audio_feats = audio_feats.squeeze(0)
        audio_length = audio_feats.shape[0]

        # Build packed sequences
        packed_text = self._build_packed_text(text_with_start, audio_length, end_id)
        packed_audio = self._build_packed_audio(audio_feats, prefix_length)
        text_mask, audio_mask = self._build_masks(prefix_length, audio_length, text.device)
        loss_mask = self._build_loss_mask(prefix_length, audio_length, is_prompt, text.device)
        labels = self._build_labels(prefix_length + audio_length + 1, text.device)

        return {
            "text_tokens": packed_text,
            "audio_feats": packed_audio,
            "text_mask": text_mask,
            "audio_mask": audio_mask,
            "loss_mask": loss_mask,
            "labels": labels,
            "duration": duration,
            "text_length": text_length,
        }

    def _build_packed_text(
        self,
        text_with_start: torch.Tensor,
        audio_length: int,
        end_id: int,
    ) -> torch.Tensor:
        """Build packed text sequence: [text, start, zeros..., end]."""
        text_padding = torch.zeros(audio_length, dtype=torch.int32, device=text_with_start.device)
        return torch.cat([
            text_with_start,
            text_padding,
            torch.tensor([end_id], dtype=torch.int32, device=text_with_start.device),
        ])

    def _build_packed_audio(
        self,
        audio_feats: torch.Tensor,
        prefix_length: int,
    ) -> torch.Tensor:
        """Build packed audio features: [zeros..., audio, zeros]."""
        zero_feats = torch.zeros(
            (prefix_length, self._patch_size, audio_feats.size(-1)),
            dtype=torch.float32,
            device=audio_feats.device,
        )
        return torch.cat([zero_feats, audio_feats, zero_feats[0:1, ...]], dim=0)

    def _build_masks(
        self,
        prefix_length: int,
        audio_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build text and audio attention masks."""
        text_mask = torch.cat([
            torch.ones(prefix_length, dtype=torch.int32),
            torch.zeros(audio_length, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        ]).to(device)

        audio_mask = torch.cat([
            torch.zeros(prefix_length, dtype=torch.int32),
            torch.ones(audio_length, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        ]).to(device)

        return text_mask, audio_mask

    def _build_loss_mask(
        self,
        prefix_length: int,
        audio_length: int,
        is_prompt: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Build loss mask (compute loss only on non-prompt audio)."""
        audio_loss = torch.zeros(audio_length) if is_prompt else torch.ones(audio_length)
        return torch.cat([
            torch.zeros(prefix_length),
            audio_loss,
            torch.zeros(1),
        ]).int().to(device)

    def _build_labels(self, total_length: int, device: torch.device) -> torch.Tensor:
        """Build stop prediction labels."""
        labels = torch.zeros(total_length, dtype=torch.int32, device=device)
        labels[-2] = 1  # Mark second-to-last position as stop
        return labels

    # ------------------------------------------------------------------ #
    # Audio Encoding
    # ------------------------------------------------------------------ #

    def _encode_audio(self, waveform: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Encode waveform through AudioVAE.

        Args:
            waveform: 1D audio waveform tensor.

        Returns:
            Tuple of (patched_features, duration_seconds).
        """
        wav = waveform.unsqueeze(0).unsqueeze(1)
        wav_length = wav.size(-1)

        # Pad to patch boundary
        if wav_length % self._patch_len != 0:
            padding = self._patch_len - wav_length % self._patch_len
            wav = torch.nn.functional.pad(wav, (0, padding))

        # Encode
        with torch.no_grad():
            latents = self._audio_vae.encode(wav, self._audio_vae.sample_rate)
            features = latents.transpose(1, 2)

        # Pad to patch_size boundary
        if features.size(1) % self._patch_size != 0:
            features_t = features.transpose(1, 2)
            padding = self._patch_size - features.size(1) % self._patch_size
            features_t = nn.functional.pad(features_t, (0, padding))
            features = features_t.transpose(1, 2)

        duration = features.size(1) / AUDIO_VAE_FPS
        features = rearrange(features, "b (t p) c -> b t p c", p=self._patch_size)

        return features, duration

    # ------------------------------------------------------------------ #
    # Batch Assembly
    # ------------------------------------------------------------------ #

    def _stack_samples(
        self,
        samples: list[dict[str, Any]],
        audio_duration: torch.Tensor,
        text_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stack processed samples with padding."""
        if not samples:
            return create_empty_batch(
                self._max_length, self._patch_size, self._feat_dim, self._device
            )

        lengths = [s["text_tokens"].shape[0] for s in samples]
        max_len = min(self._max_length, max(lengths))

        return {
            "text_tokens": torch.stack([pad_1d(s["text_tokens"], max_len) for s in samples]),
            "audio_feats": torch.stack([pad_3d(s["audio_feats"], max_len) for s in samples]),
            "text_mask": torch.stack([pad_1d(s["text_mask"], max_len) for s in samples]),
            "audio_mask": torch.stack([pad_1d(s["audio_mask"], max_len) for s in samples]),
            "loss_mask": torch.stack([pad_1d(s["loss_mask"], max_len) for s in samples]),
            "position_ids": build_position_ids(lengths, max_len, self._device),
            "labels": torch.stack([pad_1d(s["labels"], max_len) for s in samples]),
            "audio_duration_consumed": audio_duration.long(),
            "text_token_consumed": text_count.long(),
        }

