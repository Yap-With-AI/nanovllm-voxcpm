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
    """LoRA adapter for merged QKV projections with pre-computed deltas.
    
    The base model uses merged QKVParallelLinear, but LoRA weights are trained
    with separate q_proj, k_proj, v_proj. This adapter stores the separate LoRA
    weights and pre-computes delta = lora_B @ lora_A * scale for fast inference.
    
    State dict structure (matching trained weights):
        - q_proj.lora_A, q_proj.lora_B
        - k_proj.lora_A, k_proj.lora_B  
        - v_proj.lora_A, v_proj.lora_B
    
    Call compute_deltas() after loading weights to pre-compute for fast inference.
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
            
            # Pre-computed deltas (lora_B @ lora_A * scale) for fast inference
            # These are buffers, not parameters - computed once after loading
            self.register_buffer("_delta_q", torch.zeros(q_output_size, hidden_size), persistent=False)
            self.register_buffer("_delta_k", torch.zeros(kv_output_size, hidden_size), persistent=False)
            self.register_buffer("_delta_v", torch.zeros(kv_output_size, hidden_size), persistent=False)
            self._deltas_computed = False
        else:
            self.q_proj_lora_A = None
            self.q_proj_lora_B = None
            self.k_proj_lora_A = None
            self.k_proj_lora_B = None
            self.v_proj_lora_A = None
            self.v_proj_lora_B = None
            self._delta_q = None
            self._delta_k = None
            self._delta_v = None
            self._deltas_computed = True
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _init_lora_weights(self):
        for name in ['q_proj_lora_A', 'k_proj_lora_A', 'v_proj_lora_A']:
            nn.init.kaiming_uniform_(getattr(self, name), a=math.sqrt(5))
        for name in ['q_proj_lora_B', 'k_proj_lora_B', 'v_proj_lora_B']:
            nn.init.zeros_(getattr(self, name))
    
    @torch.no_grad()
    def compute_deltas(self) -> None:
        """Pre-compute delta matrices for fast inference.
        
        Computes delta = lora_B @ lora_A * scale for each projection.
        Call this after loading LoRA weights.
        """
        if self.r <= 0:
            return
        scale = self._base_scaling
        self._delta_q.copy_(self.q_proj_lora_B @ self.q_proj_lora_A * scale)
        self._delta_k.copy_(self.k_proj_lora_B @ self.k_proj_lora_A * scale)
        self._delta_v.copy_(self.v_proj_lora_B @ self.v_proj_lora_A * scale)
        self._deltas_computed = True
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LoRA deltas for Q, K, V using pre-computed delta matrices.
        
        Args:
            x: Input tensor [batch, seq, hidden] or [seq, hidden]
            
        Returns:
            Tuple of (delta_q, delta_k, delta_v) to add to base Q, K, V outputs
        """
        # Single matmul per projection using pre-computed deltas
        delta_q = F.linear(x, self._delta_q)
        delta_k = F.linear(x, self._delta_k)
        delta_v = F.linear(x, self._delta_v)
        
        return (
            self.dropout(delta_q),
            self.dropout(delta_k),
            self.dropout(delta_v),
        )
    
    def set_enabled(self, enabled: bool) -> None:
        self.scaling.fill_(self._base_scaling if enabled else 0.0)
        # Recompute deltas with new scaling
        self.compute_deltas()


class OutputLoRAAdapter(nn.Module):
    """LoRA adapter for output projection (o_proj) with pre-computed delta.
    
    State dict structure:
        - lora_A, lora_B (or o_proj.lora_A, o_proj.lora_B)
    
    Call compute_deltas() after loading weights to pre-compute for fast inference.
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
            
            # Pre-computed delta for fast inference
            self.register_buffer("_delta", torch.zeros(output_size, input_size), persistent=False)
            self._delta_computed = False
        else:
            self.lora_A = None
            self.lora_B = None
            self._delta = None
            self._delta_computed = True
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    @torch.no_grad()
    def compute_deltas(self) -> None:
        """Pre-compute delta matrix for fast inference.
        
        Computes delta = lora_B @ lora_A * scale.
        Call this after loading LoRA weights.
        """
        if self.r <= 0:
            return
        scale = self._base_scaling
        self._delta.copy_(self.lora_B @ self.lora_A * scale)
        self._delta_computed = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA delta for output projection using pre-computed delta."""
        # Single matmul using pre-computed delta
        delta = F.linear(x, self._delta)
        return self.dropout(delta)
    
    def set_enabled(self, enabled: bool) -> None:
        self.scaling.fill_(self._base_scaling if enabled else 0.0)
        # Recompute delta with new scaling
        self.compute_deltas()


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
