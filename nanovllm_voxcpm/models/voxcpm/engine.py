from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMRunner, RunnerTask, VoxCPMPayload
from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.models.voxcpm.config import VoxCPMConfig
from nanovllm_voxcpm.engine.sequence import Sequence
from dataclasses import dataclass
import numpy as np
from transformers import LlamaTokenizerFast
from nanovllm_voxcpm.models.voxcpm.utils import mask_multichar_chinese_tokens
import torch


@dataclass
class VoxCPMSeqPayload:
    # [(T, P, D)]
    feats : list[np.ndarray]

    text_tokens : list[int]
    feat_masks : list[bool]
    
    generated_waveforms : list[np.ndarray]

    temperature : float
    cfg_value : float

    decode_pad : np.ndarray | None = None
    max_generate_length : int | None = None
    

class VoxCPMEngine(LLMEngineBase):
    def __init__(self, config: Config[VoxCPMConfig]):
        self.n_decode_pad_frames = 4
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.audio_start_token = 101
        self.block_size = config.kvcache_block_size

        self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))

        super().__init__(VoxCPMRunner, config, config.tensor_parallel_size)
    
    def preprocess_seq(self, seq: Sequence[VoxCPMSeqPayload], is_prefill: bool) -> RunnerTask[VoxCPMPayload]:
        if is_prefill:
            # Concatenate features if needed
            if len(seq.custom_payload.feats) > 1:
                feats = np.concatenate(seq.custom_payload.feats, axis=0)
                seq.custom_payload.feats = [feats]

            # Calculate chunk boundaries
            start_idx = seq.num_cached_tokens
            remaining_tokens = len(seq) - start_idx
            
            # Apply chunk size limit
            if self.prefill_chunk_size > 0:
                chunk_tokens = min(remaining_tokens, self.prefill_chunk_size)
            else:
                chunk_tokens = remaining_tokens
            
            end_idx = start_idx + chunk_tokens
            
            # Check if this is the FINAL prefill chunk
            is_final_chunk = (end_idx >= len(seq))

            return RunnerTask(
                seq.block_table,
                end_idx,  # seq_length = up to end of this chunk
                start_idx,  # num_cached_tokens = start of this chunk
                seq.block_size,
                VoxCPMPayload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[start_idx:end_idx], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][start_idx:end_idx],
                    feat_masks=np.array(seq.custom_payload.feat_masks[start_idx:end_idx], dtype=np.bool_),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    # Only include decode_pad on final prefill chunk
                    padding_decode=seq.custom_payload.decode_pad if is_final_chunk else None,
                )
            )
        else:
            # Decode step - unchanged
            return RunnerTask(
                seq.block_table,
                len(seq),
                len(seq) - 1,
                seq.block_size,
                VoxCPMPayload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[-1:], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][-1:],
                    feat_masks=np.array(seq.custom_payload.feat_masks[-1:], dtype=np.bool_),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    padding_decode=seq.custom_payload.decode_pad,
                )
            )


    def postprocess_seq(self, seq: Sequence[VoxCPMSeqPayload], outputs: dict, is_prefill: bool):
        """
        Process outputs after a step.
        
        For prefill: Only called on the FINAL chunk, produces first audio token.
        For decode: Called every step, produces subsequent audio tokens.
        """
        stop_flag = outputs["stop_flag"]
        latents = outputs["latents"]
        waveforms = outputs["waveforms"]

        seq.append_token(latents.tobytes())

        seq.custom_payload.feats.append(latents[None])
        seq.custom_payload.text_tokens.append(0)
        seq.custom_payload.feat_masks.append(True)

        seq.custom_payload.generated_waveforms.append(waveforms)

        latents = latents.reshape(-1, self.feat_dim)
        if seq.custom_payload.decode_pad is not None:
            seq.custom_payload.decode_pad = np.concatenate([seq.custom_payload.decode_pad, latents], axis=0)[-self.n_decode_pad_frames:]
        else:
            seq.custom_payload.decode_pad = latents[-self.n_decode_pad_frames:]

        if stop_flag == 1:
            seq.stoped = True
        elif seq.custom_payload.max_generate_length is not None and len(seq.custom_payload.generated_waveforms) >= seq.custom_payload.max_generate_length:
            seq.stoped = True

    def add_request(
            self,
            seq_id : str,
            target_text : str,
            prompt_text : str = "",
            prompt_latents : np.ndarray = None,
            max_generate_length : int = 2000,
            temperature : float = 1.0,
            cfg_value : float = 2.0,
        ):
        text_tokens = self.tokenizer(prompt_text + target_text) + [self.audio_start_token]
        audio_feat = np.zeros((len(text_tokens), self.patch_size, self.feat_dim), dtype=np.float32)
        feat_masks = [False for _ in range(len(text_tokens))]
        hash_tokens = []
        for t in text_tokens:
            hash_tokens.append(t)

        decode_pad = None

        if prompt_latents is not None:
            wav_latents = prompt_latents
            decode_pad = wav_latents[-self.n_decode_pad_frames:]
            
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)
            audio_feat = np.concatenate([audio_feat, wav_latents], axis=0)
            text_tokens.extend([0 for _ in range(wav_latents.shape[0])])
            feat_masks.extend([True for _ in range(wav_latents.shape[0])])

            for i in range(wav_latents.shape[0]):
                hash_tokens.append(wav_latents[i].tobytes())

        seq = Sequence(
            seq_id,
            hash_tokens,
            self.block_size,
            VoxCPMSeqPayload(
                feats=[audio_feat],
                text_tokens=text_tokens,
                feat_masks=feat_masks,
                decode_pad=decode_pad,
                temperature=temperature,
                cfg_value=cfg_value,
                max_generate_length=max_generate_length,
                generated_waveforms=[],
            )
        )

        self.add_sequence(seq)

    def encode_latents(self, wav : torch.Tensor, align_size : int = -1) -> np.ndarray:
        """ Encode wav to latents
        This function will pad the wav to the nearest multiple of the chunk size.
        Args:
            wav: (1, T)
        Returns:
            latents: (n_latents, dim_feat)
        """
        if align_size == -1:
            align_size = self.patch_size * self.model_runner.vae.chunk_size
        if wav.size(1) % align_size != 0:
            remained = align_size - wav.size(1) % align_size
            wav = torch.nn.functional.pad(wav, (remained, 0))
        return self.model_runner.encode_latents(wav)
