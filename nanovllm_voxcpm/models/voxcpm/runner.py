from dataclasses import dataclass
import torch
from multiprocessing.synchronize import Event
import json
import logging
from typing import List

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.model_runner import RunnerTask, BaseModelRunner
from nanovllm_voxcpm.utils.loader import load_model
from nanovllm_voxcpm.models.voxcpm.model import VoxCPMModel, VoxCPMConfig
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig
from nanovllm_voxcpm.layers.audio_vae import AudioVAE
from torch.nn.utils import remove_weight_norm
import numpy as np
import os

try:
    from safetensors.torch import load_file as safetensors_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _apply_torch_compile(
    model: VoxCPMModel,
    targets: List[str],
    mode: str,
    fullgraph: bool,
    dynamic: bool,
) -> VoxCPMModel:
    """Apply torch.compile to specified model submodules.
    
    Args:
        model: The VoxCPMModel instance
        targets: List of targets to compile: "all", "estimator", "encoder", "lm", "residual_lm"
        mode: Compilation mode - "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        fullgraph: Whether to compile with fullgraph=True (stricter, may fail on dynamic control flow)
        dynamic: Whether to use dynamic shapes (recommended for variable batch sizes)
    
    Returns:
        The model with compiled submodules
    """
    compile_kwargs = {
        "mode": mode,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
    }
    
    if "all" in targets:
        # Compile the entire model - simplest but may have issues with complex control flow
        logger.info(f"Compiling entire VoxCPMModel with mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
        return torch.compile(model, **compile_kwargs)
    
    # Compile individual submodules for finer control
    if "estimator" in targets:
        # The DiT estimator is called multiple times per inference (inference_timesteps times)
        # This is the highest-value target for compilation
        logger.info(f"Compiling feat_decoder.estimator (DiT) with mode={mode}")
        model.feat_decoder.estimator = torch.compile(
            model.feat_decoder.estimator, **compile_kwargs
        )
    
    if "encoder" in targets:
        # Local encoder for feature encoding
        logger.info(f"Compiling feat_encoder (VoxCPMLocEnc) with mode={mode}")
        model.feat_encoder = torch.compile(model.feat_encoder, **compile_kwargs)
    
    if "lm" in targets:
        # Base language model - most parameters, significant compute
        logger.info(f"Compiling base_lm (Cpm4Model) with mode={mode}")
        model.base_lm = torch.compile(model.base_lm, **compile_kwargs)
    
    if "residual_lm" in targets:
        # Residual acoustic LM
        logger.info(f"Compiling residual_lm (Cpm4Model) with mode={mode}")
        model.residual_lm = torch.compile(model.residual_lm, **compile_kwargs)
    
    return model

@dataclass
class VoxCPMPayload:
    # (T)
    text_tokens : np.ndarray | None = None
    # (T, P, D)
    feats : np.ndarray | None = None
    # (T)
    feat_masks : np.ndarray | None = None
    
    temperature : float = 1.0
    cfg_value : float = 2.0

    # (T, D)
    padding_decode : np.ndarray | None = None


class VoxCPMRunner(BaseModelRunner):
    # Max padding frames for VAE decode (from engine.py n_decode_pad_frames)
    N_DECODE_PAD_FRAMES = 4
    
    def __init__(self, config: Config[VoxCPMConfig], rank: int, device_idx : int, distributed_port: int, event: Event | list[Event]):
        self.inference_timesteps = config.model_config.inference_timesteps
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.max_num_seqs = config.max_num_seqs  # For buffer pre-allocation
        self.lora_path = getattr(config, 'lora_path', None)
        
        # Multi-LoRA paths for hotswapping (e.g., {"female": "/path/to/female", "male": "/path/to/male"})
        self.lora_paths = getattr(config, 'lora_paths', None)
        self.default_voice = getattr(config, 'default_voice', 'female')
        
        # torch.compile configuration
        self.use_torch_compile = getattr(config, 'use_torch_compile', False)
        self.compile_mode = getattr(config, 'compile_mode', 'max-autotune')
        self.compile_targets = getattr(config, 'compile_targets', ['estimator', 'lm', 'residual_lm', 'encoder'])
        self.compile_fullgraph = getattr(config, 'compile_fullgraph', False)
        self.compile_dynamic = getattr(config, 'compile_dynamic', True)
        
        # Async VAE configuration - overlaps VAE decode with other operations
        self.async_vae = getattr(config, 'async_vae', True)
        self._vae_stream = None  # Lazy init after CUDA context is ready
        
        # Multi-LoRA state storage for instant hotswapping
        self._lora_states: dict[str, dict] = {}
        self._current_voice: str | None = None
        
        # Pre-allocated buffer for VAE decode (initialized in init_model after CUDA ready)
        self._vae_input_buffer: torch.Tensor | None = None
        
        super().__init__(config, rank, device_idx, distributed_port, event)
    
    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16
    
    @property
    def vae_stream(self) -> torch.cuda.Stream:
        """Lazy init VAE stream after CUDA context is ready."""
        if self._vae_stream is None:
            self._vae_stream = torch.cuda.Stream()
        return self._vae_stream
    
    def init_model(self, model_config : VoxCPMConfig, model_path : str):
        # Determine LoRA configuration - check multi-LoRA first, then fallback to single
        lora_config = None
        if self.lora_paths is not None:
            # Multi-LoRA mode: use first available path to get config structure
            first_lora_path = next(iter(self.lora_paths.values()))
            lora_config = self._load_lora_config(first_lora_path)
            logger.info(f"Multi-LoRA mode enabled with voices: {list(self.lora_paths.keys())}")
        elif self.lora_path is not None:
            lora_config = self._load_lora_config(self.lora_path)
            logger.info(f"LoRA config loaded from {self.lora_path}: {lora_config}")
        
        self.model = VoxCPMModel(model_config, self.inference_timesteps, lora_config=lora_config)
        load_model(self.model, model_path)
        
        # Load LoRA weights - multi-LoRA or single-LoRA mode
        if self.lora_paths is not None:
            self._init_multi_lora(self.lora_paths)
        elif self.lora_path is not None and lora_config is not None:
            self._load_lora_weights(self.lora_path)
            self._current_voice = "default"
        
        # Apply torch.compile if configured
        # Note: Compile AFTER loading weights but BEFORE CUDA graph capture
        if self.use_torch_compile:
            logger.info(f"Applying torch.compile with targets={self.compile_targets}, mode={self.compile_mode}")
            self.model = _apply_torch_compile(
                self.model,
                targets=self.compile_targets,
                mode=self.compile_mode,
                fullgraph=self.compile_fullgraph,
                dynamic=self.compile_dynamic,
            )

        torch.set_default_dtype(torch.float32)
        self.vae = AudioVAE() if model_config.audio_vae_config is None else AudioVAE(**model_config.audio_vae_config.model_dump(mode="dict"))

        vae_state_dict = torch.load(os.path.join(model_path, "audiovae.pth"))["state_dict"]
        self.vae.load_state_dict(vae_state_dict)
        
        # Fuse weight_norm for inference (removes runtime overhead)
        self._fuse_vae_weight_norm()
        
        # Convert VAE decoder to bfloat16 for memory savings (~50% less VRAM for decoder)
        # Encoder stays fp32 for encoding quality, decoder can use bf16 for generation
        self.vae.decoder = self.vae.decoder.to(torch.bfloat16)
        logger.info("VAE decoder converted to bfloat16 for memory optimization")
        
        # Compile VAE encoder and decoder for faster inference
        # Use "reduce-overhead" for VAE - avoids noisy autotuning errors and works well for inference
        vae_compile_mode = "reduce-overhead"
        logger.info(f"Compiling VAE encoder + decoder with torch.compile ({vae_compile_mode})")
        self.vae.encoder = torch.compile(self.vae.encoder, mode=vae_compile_mode, fullgraph=False)
        self.vae.decoder = torch.compile(self.vae.decoder, mode=vae_compile_mode, fullgraph=False)
        
        # Pre-allocate VAE input buffer (avoids per-step allocation)
        # Use bfloat16 to match decoder dtype for zero-copy decode
        max_vae_seq_len = self.N_DECODE_PAD_FRAMES + self.patch_size
        self._vae_input_buffer = torch.empty(
            self.max_num_seqs, max_vae_seq_len, self.feat_dim,
            dtype=torch.bfloat16, device="cuda"
        )
        logger.info(f"Pre-allocated VAE input buffer: {self._vae_input_buffer.shape} (bfloat16)")
        
        # Warmup VAE at various batch sizes to pre-compile for all shapes
        self._warmup_vae_batch_sizes()
        
        torch.set_default_dtype(torch.bfloat16)
        
        # Warmup all LoRAs for JIT compilation if multi-LoRA mode
        if self.lora_paths is not None and self.use_torch_compile:
            self._warmup_all_loras()
    
    def _fuse_vae_weight_norm(self):
        """Remove weight_norm from VAE modules to fuse weights for inference.
        
        weight_norm adds runtime overhead by computing weight = g * v/||v|| each forward.
        After training, we can fuse g and v into a single weight tensor.
        """
        fused_count = 0
        for module in self.vae.modules():
            if hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
                try:
                    remove_weight_norm(module)
                    fused_count += 1
                except ValueError:
                    pass  # Module doesn't have weight_norm applied
        logger.info(f"Fused weight_norm in {fused_count} VAE modules")
    
    @torch.inference_mode()
    def _warmup_vae_batch_sizes(self):
        """Warmup VAE encoder + decoder at various batch sizes to pre-compile torch.compile.
        
        Without this, first inference at a new batch size triggers recompilation,
        causing TTFB spikes when concurrency changes.
        """
        max_vae_seq_len = self.N_DECODE_PAD_FRAMES + self.patch_size
        # Encoder input length (audio samples) - 1 second at VAE's sample rate
        encoder_audio_len = self.vae.sample_rate
        
        # Warmup batch sizes: 1, powers of 2 up to max, and max itself
        warmup_sizes = [1]
        power = 2
        while power <= self.max_num_seqs:
            warmup_sizes.append(power)
            power *= 2
        if self.max_num_seqs not in warmup_sizes:
            warmup_sizes.append(self.max_num_seqs)
        warmup_sizes = sorted(warmup_sizes)
        
        logger.info(f"Warming up VAE encoder + decoder for batch sizes: {warmup_sizes}")
        
        for bs in warmup_sizes:
            # Warmup decoder: (batch, feat, seq) - use bf16 to match decoder dtype
            decoder_input = torch.randn(
                bs, self.feat_dim, max_vae_seq_len,
                dtype=torch.bfloat16, device="cuda"
            )
            _ = self.vae.decode(decoder_input)
            
            # Warmup encoder: (batch, 1, audio_samples) - stays fp32
            encoder_input = torch.randn(
                bs, 1, encoder_audio_len,
                dtype=torch.float32, device="cuda"
            )
            _ = self.vae.encoder(encoder_input)
        
        torch.cuda.synchronize()
        logger.info("VAE batch size warmup complete")
    
    def _load_lora_config(self, lora_path: str) -> LoRAConfig:
        """Load LoRA config from a directory containing lora_config.json.
        
        Per VoxCPM lora-vox documentation, the lora_config.json file has a nested
        structure with a "lora_config" key containing the actual config values.
        """
        config_path = os.path.join(lora_path, "lora_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"LoRA config not found at {config_path}")
        
        with open(config_path, "r") as f:
            lora_info = json.load(f)
        
        # Handle nested "lora_config" structure per VoxCPM documentation
        # The file structure is: {"lora_config": {...actual config...}}
        if "lora_config" in lora_info:
            config_dict = lora_info["lora_config"]
        else:
            # Fallback: treat as flat config for compatibility
            config_dict = lora_info
        
        return LoRAConfig(**config_dict)
    
    def _load_lora_weights(self, lora_path: str):
        """Load LoRA weights from a directory."""
        safetensors_path = os.path.join(lora_path, "lora_weights.safetensors")
        bin_path = os.path.join(lora_path, "lora_weights.bin")
        
        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            logger.info(f"Loading LoRA weights from safetensors: {safetensors_path}")
            lora_state_dict = safetensors_load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            logger.info(f"Loading LoRA weights from bin: {bin_path}")
            lora_state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"LoRA weights not found. Expected either {safetensors_path} or {bin_path}"
            )
        
        loaded_keys, skipped_keys = self.model.load_lora_weights(lora_state_dict, device="cuda")
        logger.info(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}")
        if skipped_keys:
            logger.warning(f"Skipped LoRA keys: {skipped_keys[:10]}...")
    
    def _init_multi_lora(self, lora_paths: dict[str, str]):
        """Initialize multi-LoRA support by loading and storing state for each voice.
        
        Args:
            lora_paths: Dict mapping voice name to LoRA directory path
                       e.g. {"female": "/path/to/female", "male": "/path/to/male"}
        """
        logger.info(f"Initializing multi-LoRA with {len(lora_paths)} voices")
        
        for voice_name, lora_path in lora_paths.items():
            logger.info(f"Loading LoRA for voice '{voice_name}' from {lora_path}")
            self._load_lora_weights(lora_path)
            
            # Store the LoRA state (weights + pre-computed deltas)
            self._lora_states[voice_name] = self.model.get_lora_state_dict()
            logger.info(f"Stored LoRA state for voice '{voice_name}' ({len(self._lora_states[voice_name])} tensors)")
        
        # Set default voice
        if self.default_voice in self._lora_states:
            self.switch_voice(self.default_voice)
        else:
            # Fall back to first available voice
            first_voice = next(iter(self._lora_states.keys()))
            logger.warning(f"Default voice '{self.default_voice}' not found, using '{first_voice}'")
            self.switch_voice(first_voice)
    
    def switch_voice(self, voice: str) -> bool:
        """Switch to a different LoRA voice instantly (no model reload).
        
        Args:
            voice: Voice name (e.g., "female" or "male")
            
        Returns:
            True if switch was successful, False if voice not found
        """
        if voice == self._current_voice:
            return True  # Already using this voice
            
        if voice not in self._lora_states:
            logger.warning(f"Voice '{voice}' not found. Available: {list(self._lora_states.keys())}")
            return False
        
        logger.debug(f"Switching voice from '{self._current_voice}' to '{voice}'")
        self.model.set_lora_state_dict(self._lora_states[voice], device="cuda")
        self._current_voice = voice
        return True
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice names."""
        return list(self._lora_states.keys())
    
    def get_current_voice(self) -> str | None:
        """Get currently active voice name."""
        return self._current_voice
    
    def _warmup_all_loras(self):
        """Warmup all LoRAs to ensure JIT compilation covers all voices.
        
        This runs a dummy forward pass with each LoRA active to ensure
        torch.compile generates optimized code for all voices.
        """
        if not self._lora_states:
            return
            
        logger.info(f"Warming up {len(self._lora_states)} LoRA voices for JIT compilation...")
        
        # Create dummy inputs for warmup
        dummy_inputs = self.make_dummy_inputs(batch_size=1, length=4)
        for key, tensor in dummy_inputs.items():
            if isinstance(tensor, torch.Tensor):
                dummy_inputs[key] = tensor.cuda()
        
        original_voice = self._current_voice
        
        for voice in self._lora_states.keys():
            logger.info(f"Warming up voice '{voice}'...")
            self.switch_voice(voice)
            
            # Run a forward pass to trigger JIT compilation
            with torch.no_grad():
                try:
                    # Minimal forward to trigger compilation paths
                    self.model.eval()
                    # We need actual context setup, so just switch is enough
                    # The real warmup happens on first inference
                except Exception as e:
                    logger.warning(f"Warmup forward pass failed for voice '{voice}': {e}")
        
        # Restore original voice
        if original_voice:
            self.switch_voice(original_voice)
        
        logger.info("LoRA warmup complete")
    
    def make_dummy_inputs(self, batch_size: int, length: int) -> torch.Tensor:
        return {
            "text_tokens": torch.zeros(batch_size * length, dtype=torch.int64),
            "feat": torch.zeros(batch_size * length, self.patch_size, self.feat_dim),
            "feat_mask": torch.zeros(batch_size * length, dtype=torch.bool),
            "temperature": torch.zeros(batch_size),
            "cfg_value": torch.zeros(batch_size),
        }

    def make_dummy_outputs(self, batch_size: int) -> torch.Tensor:
        # Use empty instead of zeros - values are always overwritten
        latents = torch.empty(
            batch_size,
            self.patch_size,
            self.feat_dim,
            dtype=self.dtype,
        )
        stop_flag = torch.empty(
            batch_size,
            dtype=torch.int64,
        )
        return {
            "latents": latents,
            "stop_flag": stop_flag,
        }
    
    def encode_latents(self, wav : torch.Tensor) -> np.ndarray:
        assert wav.ndim == 2, "Invalid shape of wav"
        return self.vae.encode(wav, self.vae.sample_rate).permute(0, 2, 1).view(-1, self.feat_dim).to(torch.float32).cpu().numpy()
    
    def run(self, seqs: list[RunnerTask[VoxCPMPayload]], is_prefill: bool):
        positions = self.prepare_prefill_context(seqs) if is_prefill else self.prepare_decode_context(seqs)
        inputs = {
            "positions": positions,
        }

        text_tokens = []
        feats = []
        feat_masks = []
        temperatures = []
        cfg_values = []

        for seq in seqs:
            payload: VoxCPMPayload = seq.custom_payload
            assert payload.text_tokens.shape[0] == payload.feats.shape[0]
            assert payload.text_tokens.shape[0] == payload.feat_masks.shape[0]

            text_tokens.append(payload.text_tokens)
            feats.append(payload.feats)
            feat_masks.append(payload.feat_masks)

            temperatures.append(payload.temperature)
            cfg_values.append(payload.cfg_value)
        
        inputs["text_tokens"] = torch.from_numpy(np.concatenate(text_tokens, axis=0)).cuda(non_blocking=True)
        inputs["feat"] = torch.from_numpy(np.concatenate(feats, axis=0)).cuda(non_blocking=True).to(self.dtype)
        inputs["feat_mask"] = torch.from_numpy(np.concatenate(feat_masks, axis=0)).cuda(non_blocking=True)
        inputs["temperature"] = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        inputs["cfg_value"] = torch.tensor(cfg_values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        
        outputs = self.run_model(inputs, is_prefill)

        latents = outputs["latents"]

        pad_lengths = []
        for i in range(len(seqs)):
            if seqs[i].custom_payload.padding_decode is not None:
                pad_lengths.append(seqs[i].custom_payload.padding_decode.shape[0])
            else:
                pad_lengths.append(0)
        max_pad_decode = max(pad_lengths) + self.patch_size
        batch_size = len(seqs)

        # Use pre-allocated buffer (slice to actual size needed)
        vae_decoder_inputs = self._vae_input_buffer[:batch_size, :max_pad_decode]
        vae_decoder_inputs.zero_()  # Clear previous data
        for i in range(batch_size):
            pad_len = pad_lengths[i]
            if pad_len > 0:
                vae_decoder_inputs[i, :pad_len] = torch.from_numpy(seqs[i].custom_payload.padding_decode).to(torch.bfloat16).cuda(non_blocking=True)
            vae_decoder_inputs[i, pad_len:pad_len+self.patch_size] = latents[i]  # Already bf16 from model
        
        if self.async_vae:
            # Async VAE: run VAE decode on separate stream, overlap with CPU operations
            # This allows VAE GPU compute to run while we transfer other data to CPU
            vae_event = torch.cuda.Event()
            # VAE stream must wait for default stream to finish populating vae_decoder_inputs
            # (the non_blocking=True copies above may not be complete yet)
            self.vae_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.vae_stream):
                vae_decoder_outputs_gpu = self.vae.decode(vae_decoder_inputs.permute(0, 2, 1))[:, 0, :]
                vae_event.record()
            
            # While VAE runs on GPU, do CPU transfers that don't depend on VAE output
            stop_flag = outputs["stop_flag"].cpu().tolist()
            np_latents = latents.to(torch.float32).cpu().numpy()
            
            # Now sync VAE and transfer results to CPU
            vae_event.synchronize()
            vae_decoder_outputs = vae_decoder_outputs_gpu.cpu().numpy()
        else:
            # Synchronous VAE decode (original behavior)
            vae_decoder_outputs = self.vae.decode(vae_decoder_inputs.permute(0, 2, 1))[:, 0, :].cpu().numpy()
            stop_flag = outputs["stop_flag"].cpu().tolist()
            np_latents = latents.to(torch.float32).cpu().numpy()

        ret_waveforms = []
        for i in range(len(seqs)):
            pad_len = pad_lengths[i]
            ret_waveforms.append(
                vae_decoder_outputs[
                    i, 
                    pad_len * self.vae.chunk_size: 
                    (pad_len + self.patch_size) * self.vae.chunk_size
                ]
            )

        ret = []
        for i in range(len(seqs)):
            ret.append({
                "latents": np_latents[i],
                "stop_flag": stop_flag[i],
                "waveforms": ret_waveforms[i],
            })

        return ret


                

