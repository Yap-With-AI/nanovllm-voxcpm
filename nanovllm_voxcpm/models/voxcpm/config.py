from pydantic import BaseModel
from typing import List, Optional

class RopeScalingConfig(BaseModel):
    type: str
    long_factor: List[float]
    short_factor: List[float]
    original_max_position_embeddings: int


class MiniCPM4Config(BaseModel):
    bos_token_id: int
    eos_token_id: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: RopeScalingConfig
    vocab_size: int
    use_mup: bool = True
    scale_emb: float
    dim_model_base: int
    scale_depth: float
    rope_theta: float
    kv_channels: int = None


class CfmConfig(BaseModel):
    sigma_min: float = 1e-06
    solver: str = "euler"
    t_scheduler: str = "log-norm"


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None


class VoxCPMDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None

    cfm_config: CfmConfig
    
class AudioVAEConfig(BaseModel):
    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_dim: int = 1536
    decoder_rates: List[int] = [8, 8, 5, 2]
    depthwise: bool = True
    sample_rate: int = 16000
    use_noise_block: bool = False


class LoRAConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""

    enable_lm: bool = False  # Apply LoRA to base_lm + residual_lm
    enable_dit: bool = False  # Apply LoRA to VoxCPMLocDiT
    enable_proj: bool = False  # Apply LoRA to projection Linear layers

    r: int = 32
    alpha: int = 16
    dropout: float = 0.0

    # Target linear layer names for LM & DiT (matched by attribute name)
    target_modules_lm: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    target_modules_dit: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    # Projection layer attribute names to find on VoxCPMModel
    target_proj_modules: List[str] = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]


class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 2
    feat_dim: int = 64
    residual_lm_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: Optional[AudioVAEConfig] = None
    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"

    inference_timesteps: int = 20
    
    # LoRA configuration (optional)
    lora_config: Optional[LoRAConfig] = None
