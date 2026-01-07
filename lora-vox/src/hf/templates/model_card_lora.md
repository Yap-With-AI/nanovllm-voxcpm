---
language: en
license: apache-2.0
base_model: {{base_model_repo}}
pipeline_tag: text-to-speech
tags:
  - voxcpm
  - tts
  - text-to-speech
  - voice-synthesis
  - lora
  - orpheus-distillation
---

# {{run_name}}

LoRA adapters for [VoxCPM 1.5](https://huggingface.co/{{base_model_repo}}), trained on the Orpheus distillation dataset for high-quality text-to-speech synthesis.

## Model Description

This repository contains per-voice LoRA adapters for [{{base_model_repo}}](https://huggingface.co/{{base_model_repo}}), distilled from the Orpheus TTS system. Each voice has its own LoRA adapter that can be hot-swapped at inference time.

## Voices

{{voice_table}}

## Usage

### Load a specific voice

```python
import json
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# Load LoRA configuration from any voice
with open("lora/female/lora_config.json") as f:
    lora_info = json.load(f)

# Initialize model with LoRA architecture
model = VoxCPM.from_pretrained(
    hf_model_id=lora_info["base_model"],
    lora_config=LoRAConfig(**lora_info["lora_config"]),
)

# Load female voice LoRA weights
model.load_lora_weights("lora/female/lora_weights.safetensors")
model.set_lora_enabled(True)

# Generate speech (no voice tag needed - voice is determined by LoRA)
wav = model.generate(text="Hello, how are you today?")
```

### Hot-swap voices at runtime

```python
# ... after initial setup above ...

# Switch to male voice (base model stays in memory)
model.load_lora_weights("lora/male/lora_weights.safetensors")
wav = model.generate(text="Hello, this is a different voice.")
```
{{params_section}}

## Technical Details

| Specification | Value |
|---------------|-------|
| Base Model | [{{base_model_repo}}](https://huggingface.co/{{base_model_repo}}) |
| Model Type | Per-Voice LoRA Adapters |
| Sample Rate | {{sample_rate}} Hz |
| Training Framework | VoxCPM |

## Files

| File | Description |
|------|-------------|
| `lora/<voice>/lora_weights.safetensors` | LoRA adapter weights for each voice |
| `lora/<voice>/lora_config.json` | LoRA configuration and base model reference |
| `model.safetensors` | Base model weights |
| `config.json` | Model configuration |
| `audiovae.pth` | Audio VAE weights |
| `tokenizer.json` | Tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `samples/` | Voice samples for each voice |

## License

This model inherits the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) from the base model.

## Citation

If you use this model, please cite the original VoxCPM paper:

```bibtex
@article{voxcpm2025,
  title        = {VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning},
  author       = {Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and Li, Xiang and Yu, Renjie and Wang, Ziyang and Ye, Runchuan and Sun, Weiyue and Gui, Jiancheng and Li, Kehan and Wu, Zhiyong and Liu, Zhiyuan},
  journal      = {arXiv preprint arXiv:2509.24650},
  year         = {2025},
}
```

---
*Generated: {{timestamp}}*
