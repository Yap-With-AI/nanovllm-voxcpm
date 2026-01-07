"""LoRA (Low-Rank Adaptation) layers for VoxCPM inference.

Provides LoRA implementations compatible with nanovllm's optimized parallel linear layers.
The LoRA weights are trained with separate q_proj/k_proj/v_proj/o_proj, but nanovllm
uses merged QKVParallelLinear. This module bridges that gap.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA linear layer that maintains nn.Linear state_dict key structure.

    State dict structure:
        - weight: Original weight (same as nn.Linear)
        - bias: Original bias (same as nn.Linear)
        - lora_A: LoRA low-rank matrix A
        - lora_B: LoRA low-rank matrix B

    This design allows loading pretrained weights without key conversion.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear), "LoRALinear only supports wrapping nn.Linear."

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self._base_scaling = alpha / r if r > 0 else 0.0

        # Use buffer for scaling to avoid torch.compile recompilation
        self.register_buffer("scaling", torch.tensor(self._base_scaling), persistent=False)

        # Hold weight and bias directly (transferred from original Linear)
        self.weight = base.weight
        self.bias = base.bias  # May be None

        # LoRA parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r <= 0 or self.lora_A is None:
            return result
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return result + self.dropout(lora_out) * self.scaling

    def reset_lora_parameters(self) -> None:
        if self.r > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def set_enabled(self, enabled: bool) -> None:
        self.scaling.fill_(self._base_scaling if enabled else 0.0)

    @property
    def enabled(self) -> bool:
        return self.scaling.item() != 0.0


class QKVLoRAAdapter(nn.Module):
    """LoRA adapter for merged QKV projections.
    
    The base model uses merged QKVParallelLinear, but LoRA weights are trained
    with separate q_proj, k_proj, v_proj. This adapter stores the separate LoRA
    weights and computes the LoRA delta to add to the merged QKV output.
    
    State dict structure (matching trained weights):
        - q_proj.lora_A, q_proj.lora_B
        - k_proj.lora_A, k_proj.lora_B  
        - v_proj.lora_A, v_proj.lora_B
    """
    
    def __init__(
        self,
        hidden_size: int,
        q_output_size: int,
        kv_output_size: int,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_output_size = q_output_size
        self.kv_output_size = kv_output_size
        self.r = r
        self.alpha = alpha
        self._base_scaling = alpha / r if r > 0 else 0.0
        
        self.register_buffer("scaling", torch.tensor(self._base_scaling), persistent=False)
        
        if r > 0:
            # Q projection LoRA
            self.q_proj_lora_A = nn.Parameter(torch.zeros(r, hidden_size))
            self.q_proj_lora_B = nn.Parameter(torch.zeros(q_output_size, r))
            
            # K projection LoRA
            self.k_proj_lora_A = nn.Parameter(torch.zeros(r, hidden_size))
            self.k_proj_lora_B = nn.Parameter(torch.zeros(kv_output_size, r))
            
            # V projection LoRA
            self.v_proj_lora_A = nn.Parameter(torch.zeros(r, hidden_size))
            self.v_proj_lora_B = nn.Parameter(torch.zeros(kv_output_size, r))
            
            self._init_lora_weights()
        else:
            self.q_proj_lora_A = None
            self.q_proj_lora_B = None
            self.k_proj_lora_A = None
            self.k_proj_lora_B = None
            self.v_proj_lora_A = None
            self.v_proj_lora_B = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _init_lora_weights(self):
        for name in ['q_proj_lora_A', 'k_proj_lora_A', 'v_proj_lora_A']:
            nn.init.kaiming_uniform_(getattr(self, name), a=math.sqrt(5))
        for name in ['q_proj_lora_B', 'k_proj_lora_B', 'v_proj_lora_B']:
            nn.init.zeros_(getattr(self, name))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LoRA deltas for Q, K, V.
        
        Args:
            x: Input tensor [batch, seq, hidden] or [seq, hidden]
            
        Returns:
            Tuple of (delta_q, delta_k, delta_v) to add to base Q, K, V outputs
        """
        if self.r <= 0 or self.scaling.item() == 0.0:
            # Return zeros with correct shapes
            if x.dim() == 2:
                seq_len = x.size(0)
            else:
                seq_len = x.size(0) * x.size(1)
            device, dtype = x.device, x.dtype
            return (
                torch.zeros(seq_len, self.q_output_size, device=device, dtype=dtype),
                torch.zeros(seq_len, self.kv_output_size, device=device, dtype=dtype),
                torch.zeros(seq_len, self.kv_output_size, device=device, dtype=dtype),
            )
        
        # Compute LoRA contributions
        delta_q = F.linear(F.linear(x, self.q_proj_lora_A), self.q_proj_lora_B)
        delta_k = F.linear(F.linear(x, self.k_proj_lora_A), self.k_proj_lora_B)
        delta_v = F.linear(F.linear(x, self.v_proj_lora_A), self.v_proj_lora_B)
        
        scale = self.scaling
        return (
            self.dropout(delta_q) * scale,
            self.dropout(delta_k) * scale,
            self.dropout(delta_v) * scale,
        )
    
    def set_enabled(self, enabled: bool) -> None:
        self.scaling.fill_(self._base_scaling if enabled else 0.0)


class OutputLoRAAdapter(nn.Module):
    """LoRA adapter for output projection (o_proj).
    
    State dict structure:
        - lora_A, lora_B (or o_proj.lora_A, o_proj.lora_B)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.r = r
        self.alpha = alpha
        self._base_scaling = alpha / r if r > 0 else 0.0
        
        self.register_buffer("scaling", torch.tensor(self._base_scaling), persistent=False)
        
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, input_size))
            self.lora_B = nn.Parameter(torch.zeros(output_size, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta for output projection."""
        if self.r <= 0 or self.lora_A is None or self.scaling.item() == 0.0:
            return torch.zeros_like(x[..., :self.output_size])
        
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.dropout(delta) * self.scaling
    
    def set_enabled(self, enabled: bool) -> None:
        self.scaling.fill_(self._base_scaling if enabled else 0.0)


def _get_parent_module(root: nn.Module, name: str) -> Optional[nn.Module]:
    """Get parent module from full name like 'layers.0.self_attn.q_proj'."""
    parts = name.split(".")
    if len(parts) == 1:
        return root
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None
        parent = getattr(parent, p)
    return parent


def apply_lora_to_named_linear_modules(
    root: nn.Module,
    *,
    target_submodule_names: list[str],
    r: int,
    alpha: float,
    dropout: float,
) -> None:
    """Inject LoRA into Linear layers matching target names.

    For example, target_submodule_names=["q_proj", "v_proj"] will replace
    all *.q_proj / *.v_proj nn.Linear modules with LoRALinear.
    """
    for full_name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        short_name = full_name.split(".")[-1]
        if short_name not in target_submodule_names:
            continue

        parent = _get_parent_module(root, full_name)
        if parent is None:
            continue

        # Replace original Linear with LoRALinear
        lora_layer = LoRALinear(
            base=module,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, short_name, lora_layer)
