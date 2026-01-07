"""Configuration constants for VoxCPM 1.5 fine-tuning.

Centralizes all magic values, default parameters, and configuration structures.
"""

from __future__ import annotations


# ============================================================================
# Voice Configuration
# ============================================================================

# Map public voice names to internal dataset voice names
# Public API uses "female"/"male", dataset uses "tara"/"zac"
VOICE_MAP: dict[str, str] = {
    "female": "tara",
    "male": "zac",
}

# All available voices (public names)
VOICES: tuple[str, ...] = ("female", "male")


def get_dataset_voice(voice: str) -> str:
    """Map public voice name to dataset voice name.

    Args:
        voice: Public voice name ("female" or "male").

    Returns:
        Dataset voice name ("tara" or "zac").

    Raises:
        ValueError: If voice is not recognized.
    """
    if voice not in VOICE_MAP:
        raise ValueError(f"Unknown voice: {voice}. Must be one of: {list(VOICE_MAP.keys())}")
    return VOICE_MAP[voice]


def get_public_voice(dataset_voice: str) -> str:
    """Map dataset voice name to public voice name.

    Args:
        dataset_voice: Dataset voice name ("tara" or "zac").

    Returns:
        Public voice name ("female" or "male").

    Raises:
        ValueError: If dataset_voice is not recognized.
    """
    for public, internal in VOICE_MAP.items():
        if internal == dataset_voice:
            return public
    raise ValueError(f"Unknown dataset voice: {dataset_voice}")


# ============================================================================
# HuggingFace Repositories
# ============================================================================

BASE_MODEL_REPO = "openbmb/VoxCPM1.5"
DEFAULT_DATASET_REPO = "yapwithai/orpheus-distillation-audio-dataset"
DEFAULT_OUTPUT_REPO = "yapwithai/vox-1.5-orpheus-distil"


# ============================================================================
# Audio Configuration
# ============================================================================

SAMPLE_RATE = 44100
AUDIO_VAE_FPS = 25.0  # Frames per second for VAE output


# ============================================================================
# Token IDs (VoxCPM protocol)
# ============================================================================

AUDIO_START_TOKEN_ID = 101
AUDIO_END_TOKEN_ID = 102
AUDIO_PROMPT_START_TOKEN_ID = 103
AUDIO_PROMPT_END_TOKEN_ID = 104
TEXT_EOS_TOKEN_ID = 2
PAD_TOKEN_ID = -100


# ============================================================================
# Dataset Column Names
# ============================================================================

TEXT_COLUMN = "text"
AUDIO_COLUMN = "audio"
DATASET_ID_COLUMN = "dataset_id"
DURATION_COLUMN = "duration"


# ============================================================================
# Training Defaults
# ============================================================================

# Batch and iteration settings
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_ITERS = 2000
DEFAULT_WARMUP_STEPS = 100
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_BATCH_TOKENS = 8192
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_VAL_SPLIT = 0.05
DEFAULT_NUM_WORKERS = 2

# Logging intervals
DEFAULT_LOG_INTERVAL = 10
DEFAULT_VALID_INTERVAL = 1000
DEFAULT_SAVE_INTERVAL = 1000

# Loss weights
DEFAULT_LOSS_DIFF_WEIGHT = 1.0
DEFAULT_LOSS_STOP_WEIGHT = 1.0

# Validation
MAX_VALIDATION_BATCHES = 10

# Learning rates by mode
LEARNING_RATE_LORA = 1e-4
LEARNING_RATE_FULL = 1e-5


# ============================================================================
# LoRA Configuration
# ============================================================================

DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")


# ============================================================================
# Model Architecture Constants
# ============================================================================

DEFAULT_MAX_LENGTH = 4096
DEFAULT_PATCH_SIZE = 4
DEFAULT_FEAT_DIM = 64




# ============================================================================
# File Names
# ============================================================================

# Tokenizer files
TOKENIZER_JSON = "tokenizer.json"
TOKENIZER_CONFIG_JSON = "tokenizer_config.json"
SPECIAL_TOKENS_MAP_JSON = "special_tokens_map.json"
TOKENIZER_FILES = (TOKENIZER_JSON, TOKENIZER_CONFIG_JSON, SPECIAL_TOKENS_MAP_JSON)

# Model files
AUDIOVAE_FILE = "audiovae.pth"
CONFIG_JSON = "config.json"
LORA_WEIGHTS_FILE = "lora_weights.safetensors"
LORA_WEIGHTS_FILE_ALT = "lora_weights.bin"
LORA_CONFIG_FILE = "lora_config.json"
MODEL_WEIGHTS_FILE = "model.safetensors"
MODEL_WEIGHTS_FILE_ALT = "model.bin"

# Training state files
OPTIMIZER_STATE_FILE = "optimizer.pth"
SCHEDULER_STATE_FILE = "scheduler.pth"
TRAINING_STATE_FILE = "training_state.json"


# ============================================================================
# Environment Variables
# ============================================================================

HF_TOKEN_ENV_VAR = "HUGGINGFACE_TOKEN"
