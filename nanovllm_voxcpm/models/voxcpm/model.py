import torch
from torch import nn
import torch.distributed as dist

from nanovllm_voxcpm.layers.activation import SiluAndMul
from nanovllm_voxcpm.layers.attention import Attention
from nanovllm_voxcpm.layers.layernorm import RMSNorm
from nanovllm_voxcpm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm_voxcpm.layers.embed_head import VocabParallelEmbedding
import math

from nanovllm_voxcpm.models.voxcpm.config import MiniCPM4Config, CfmConfig, VoxCPMConfig, LoRAConfig
from nanovllm_voxcpm.utils.context import get_context
from nanovllm_voxcpm.layers.lora import LoRALinear, QKVLoRAAdapter, OutputLoRAAdapter

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    
    This is equivalent to the MiniCPM modeling implementation.
    """
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


class MiniCPMLongRoPE(nn.Module):
    """MiniCPM LongRoPE implementation equivalent to modeling_minicpm.py"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ) -> None:
        super().__init__()
        self.dim = head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.short_factor = short_factor or [1.0] * (head_size // 2)
        self.long_factor = long_factor or [1.0] * (head_size // 2)
        self.original_max_position_embeddings = original_max_position_embeddings or max_position_embeddings
        
        # Calculate scaling factor (kept for compatibility, but not used to scale cos/sin amplitude)
        scale = (max_position_embeddings / self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        
        # Create base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Pre-compute cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # Do NOT scale cos/sin amplitude; only frequency is scaled by ext_factors
        # Store in bfloat16 to avoid dtype conversions during forward pass (torch.compile friendly)
        self.register_buffer('cos_cached', (emb.cos() * self.scaling_factor).to(torch.bfloat16), persistent=False)
        self.register_buffer('sin_cached', (emb.sin() * self.scaling_factor).to(torch.bfloat16), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position: [t]
        # query: [t, h, d]
        # key: [t, h, d]
        num_tokens = positions.size(0)
        
        # Get cos/sin for the positions
        cos = self.cos_cached[positions]  # [num_tokens, head_dim]
        sin = self.sin_cached[positions]  # [num_tokens, head_dim]
        
        # Apply rotary embedding using the original nano-vllm method but with corrected math
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.dim)
        query = self._apply_rotary_emb(query, cos, sin).view(query_shape)
        
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.dim)
        key = self._apply_rotary_emb(key, cos, sin).view(key_shape)
        
        return query, key
    
    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding in native dtype (bf16) for better performance.
        
        Staying in bf16 avoids dtype conversion overhead and is more torch.compile friendly.
        Modern GPUs handle bf16 RoPE with sufficient precision.
        
        Args:
            x: [num_tokens, num_heads, head_dim]
            cos/sin: [num_tokens, head_dim] from _set_cos_sin_cache (already repeated)
        """
        cos = cos.unsqueeze(1)  # [num_tokens, 1, head_dim] to broadcast over heads
        sin = sin.unsqueeze(1)  # [num_tokens, 1, head_dim] to broadcast over heads
        
        # Apply standard RoPE: (x * cos) + (rotate_half(x) * sin)
        # Stay in native dtype (bf16) - no fp32 conversion needed
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rotate_half_x = torch.cat((-x2, x1), dim=-1)
        
        return x * cos + rotate_half_x * sin


def get_cpm4_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """Get CPM4 LongRoPE implementation"""
    rotary_emb = MiniCPMLongRoPE(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        short_factor=rope_scaling.short_factor if rope_scaling else None,
        long_factor=rope_scaling.long_factor if rope_scaling else None,
        original_max_position_embeddings=rope_scaling.original_max_position_embeddings if rope_scaling else None,
    )
    return rotary_emb


class Cpm4Attention(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        is_causal: bool = True,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        apply_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position = max_position
        self.apply_qk_norm = apply_qk_norm
        self.is_causal = is_causal
        

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
        )
        self.rotary_emb = get_cpm4_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            is_causal=self.is_causal,
        )
        if self.apply_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        
        # LoRA adapters (initialized as None, set via add_lora_adapters)
        self.qkv_lora: QKVLoRAAdapter | None = None
        self.o_lora: OutputLoRAAdapter | None = None
    
    def add_lora_adapters(self, r: int, alpha: float, dropout: float = 0.0) -> None:
        """Add LoRA adapters to this attention layer.
        
        The trained LoRA weights use separate q_proj, k_proj, v_proj, o_proj naming.
        This method creates adapters that match that structure while working with
        our merged QKV projection.
        """
        # QKV LoRA adapter - stores separate q/k/v LoRA weights
        self.qkv_lora = QKVLoRAAdapter(
            hidden_size=self.hidden_size,
            q_output_size=self.q_size,
            kv_output_size=self.kv_size,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Output projection LoRA adapter
        self.o_lora = OutputLoRAAdapter(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply QKV LoRA if present
        if self.qkv_lora is not None:
            delta_q, delta_k, delta_v = self.qkv_lora(hidden_states)
            q = q + delta_q
            k = k + delta_k
            v = v + delta_v

        if self.is_causal:
            # Apply Q/K normalization only if enabled
            assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2, "q, k, v must be 2D tensors"
            if self.q_norm is not None:
                q_by_head = q.view(-1, self.num_heads, self.head_dim)
                q_by_head = self.q_norm(q_by_head)
                q = q_by_head.view(q.shape)
                
                k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
                k = k_by_head.view(k.shape)
            
            # Apply rotary embedding using nano-vllm interface
            q, k = self.rotary_emb(positions, q, k)

            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3, "q, k, v must be 3D tensors"
            B = q.size(0)

            if self.q_norm is not None:
                q_by_head = q.view(-1, self.num_heads, self.head_dim)
                q_by_head = self.q_norm(q_by_head)
                q = q_by_head.view(q.shape)
                
                k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
                k = k_by_head.view(k.shape)
            
            # Apply rotary embedding using nano-vllm interface
            q, k = self.rotary_emb(positions.repeat(B), q, k)
            q = q.view(B, -1, self.num_heads, self.head_dim)
            k = k.view(B, -1, self.num_kv_heads, self.head_dim)
            v = v.view(B, -1, self.num_kv_heads, self.head_dim)

        o = self.attn(q, k, v)

        if self.is_causal:
            o = o.view(-1, self.num_heads * self.head_dim)
        else:
            o = o.view(B, -1, self.num_heads * self.head_dim)

        output = self.o_proj(o)
        
        # Apply output LoRA if present
        if self.o_lora is not None:
            output = output + self.o_lora(o)
        
        return output


class Cpm4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Cpm4DecoderLayer(nn.Module):

    def __init__(
        self,
        config : MiniCPM4Config,
        is_causal: bool = True, 
    ) -> None:
        super().__init__()
        self.self_attn = Cpm4Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            is_causal=is_causal,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            apply_qk_norm=getattr(config, 'apply_qk_norm', False),
        )
        self.mlp = Cpm4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # depth scaling like MiniCPM
        self.scale_depth = getattr(config, 'scale_depth', 1.0)
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PreNorm + Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(positions, hidden_states)
        hidden_states = residual + attn_out

        # PreNorm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out
        return hidden_states, residual


class Cpm4Model(nn.Module):

    def __init__(
        self,
        config: MiniCPM4Config,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

        if config.vocab_size > 0:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Identity()
        
        self.layers = nn.ModuleList([Cpm4DecoderLayer(config, is_causal) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class VoxCPMLocDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        config: MiniCPM4Config,
        in_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.config = config

        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)

        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )
        self.delta_time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )

        assert config.vocab_size == 0, "vocab_size must be 0 for local DiT"
        self.decoder = Cpm4Model(config, is_causal=False)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of inputs
        mu: (N, C) tensor of hidden embedding
        t: (N,) tensor of diffusion timesteps
        cond: (N, C, T') tensor of prefix conditions
        dt: (N,) used for mean velocity (may be supported in the future...)
        """
        x = self.in_proj(x.transpose(1, 2).contiguous())

        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)

        t = self.time_embeddings(t).to(x.dtype)
        t = self.time_mlp(t)
        dt = self.time_embeddings(dt).to(x.dtype)
        dt = self.delta_time_mlp(dt)
        t = t + dt

        x = torch.cat([(mu + t).unsqueeze(1), cond, x], dim=1)

        position_ids = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        hidden = self.decoder(x, position_ids)
        hidden = hidden[:, prefix + 1 :, :]
        hidden = self.out_proj(hidden)

        return hidden.transpose(1, 2).contiguous()

class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size : int,
        inference_timesteps : int,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.in_channels = in_channels
        self.mean_mode = mean_mode
        self.patch_size = patch_size
        self.inference_timesteps = inference_timesteps

        # Just change the architecture of the estimator here
        self.estimator = estimator
        
        # Pre-compute t_span with sway sampling (constant across all calls)
        # Will be moved to correct device/dtype on first forward
        t_span = torch.linspace(1, 0, inference_timesteps + 1)
        t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        self.register_buffer('_t_span_base', t_span, persistent=False)
        
        # Pre-compute all t and dt values for each step (deterministic)
        # These are used to avoid scalar operations in the CUDA graph
        t_values = []
        dt_values = []
        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t - t_span[step]
            t_values.append(t.item())
            dt_values.append(dt.item())
            t = t - dt
        self.register_buffer('_t_values', torch.tensor(t_values), persistent=False)
        self.register_buffer('_dt_values', torch.tensor(dt_values), persistent=False)
        
        # CUDA graph state (initialized by capture_cuda_graphs)
        self._euler_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._euler_graph_pool = None
        self._graph_vars: dict[int, dict] = {}
        self._graphs_captured = False

    @torch.inference_mode()
    def capture_cuda_graphs(self, max_batch_size: int, mu_dim: int, dtype: torch.dtype = torch.bfloat16):
        """Capture CUDA graphs for the Euler solver loop at various batch sizes.
        
        This eliminates kernel launch overhead between diffusion timesteps.
        Call this during model warmup, after torch.compile has warmed up.
        
        Args:
            max_batch_size: Maximum batch size to capture graphs for
            mu_dim: Dimension of mu (hidden dim from LM)
            dtype: Data type for tensors
        """
        import logging
        logger = logging.getLogger(__name__)
        
        device = next(self.parameters()).device
        seq_len = self.patch_size
        
        # Batch sizes to capture: 1, powers of 2, and max
        batch_sizes = sorted(set([1, 2, 4, 8, 16, 32] + [max_batch_size]))
        batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]
        
        logger.info(f"Capturing CUDA graphs for Euler solver at batch sizes: {batch_sizes}")
        
        # Move pre-computed values to device
        t_values = self._t_values.to(device=device, dtype=dtype)
        dt_values = self._dt_values.to(device=device, dtype=dtype)
        
        for bs in reversed(batch_sizes):  # Capture largest first for memory pool
            logger.info(f"  Capturing Euler graph for batch_size={bs}...")
            
            # Allocate graph input buffers
            graph_vars = {
                # Inputs (copied before replay)
                'x': torch.zeros(bs, self.in_channels, seq_len, device=device, dtype=dtype),
                'mu': torch.zeros(bs, mu_dim, device=device, dtype=dtype),
                'cond': torch.zeros(bs, self.in_channels, seq_len, device=device, dtype=dtype),
                'cfg_value': torch.zeros(bs, device=device, dtype=dtype),
                # Internal buffers (reused during graph execution)
                'x_in': torch.zeros(2 * bs, self.in_channels, seq_len, device=device, dtype=dtype),
                'mu_in': torch.zeros(2 * bs, mu_dim, device=device, dtype=dtype),
                't_in': torch.zeros(2 * bs, device=device, dtype=dtype),
                'dt_in': torch.zeros(2 * bs, device=device, dtype=dtype),
                'cond_in': torch.zeros(2 * bs, self.in_channels, seq_len, device=device, dtype=dtype),
                # Output
                'output': torch.zeros(bs, self.in_channels, seq_len, device=device, dtype=dtype),
            }
            
            # Warmup run (required before capture)
            self._run_euler_loop_for_capture(
                graph_vars, bs, t_values, dt_values
            )
            torch.cuda.synchronize()
            
            # Capture the graph
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._euler_graph_pool):
                self._run_euler_loop_for_capture(
                    graph_vars, bs, t_values, dt_values
                )
            
            if self._euler_graph_pool is None:
                self._euler_graph_pool = graph.pool()
            
            self._euler_graphs[bs] = graph
            self._graph_vars[bs] = graph_vars
            torch.cuda.synchronize()
        
        self._graphs_captured = True
        logger.info(f"Euler CUDA graph capture complete for {len(batch_sizes)} batch sizes")
    
    def _run_euler_loop_for_capture(
        self,
        graph_vars: dict,
        bs: int,
        t_values: torch.Tensor,
        dt_values: torch.Tensor,
    ):
        """Run the Euler loop for CUDA graph capture.
        
        Uses in-place operations where possible for graph compatibility.
        """
        x = graph_vars['x']
        mu = graph_vars['mu']
        cond = graph_vars['cond']
        cfg_value = graph_vars['cfg_value']
        x_in = graph_vars['x_in']
        mu_in = graph_vars['mu_in']
        t_in = graph_vars['t_in']
        dt_in = graph_vars['dt_in']
        cond_in = graph_vars['cond_in']
        output = graph_vars['output']
        
        # Copy x to working buffer (will be modified in-place)
        output.copy_(x)
        
        # Setup static parts of CFG inputs
        mu_in[:bs].copy_(mu)
        mu_in[bs:].zero_()  # Unconditional
        cond_in[:bs].copy_(cond)
        cond_in[bs:].copy_(cond)
        
        # Run all timesteps
        for step in range(self.inference_timesteps):
            t = t_values[step]
            dt = dt_values[step]
            
            # Fill x_in with current x state
            x_in[:bs].copy_(output)
            x_in[bs:].copy_(output)
            t_in.fill_(t)
            if self.mean_mode:
                dt_in.fill_(dt)
            
            # Run estimator
            dphi_dt_full = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
            dphi_dt = dphi_dt_full[:bs]
            cfg_dphi_dt = dphi_dt_full[bs:]
            
            # Compute CFG guidance scale
            positive_flat = dphi_dt.view(bs, -1)
            negative_flat = cfg_dphi_dt.view(bs, -1)
            dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
            squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
            st_star = (dot_product / squared_norm).view(bs, 1, 1)
            
            # Apply CFG: result = uncond * st_star + cfg * (cond - uncond * st_star)
            # output = output - dt * (cfg_dphi_dt * st_star + cfg_value[:, None, None] * (dphi_dt - cfg_dphi_dt * st_star))
            guided = cfg_dphi_dt * st_star + cfg_value[:, None, None] * (dphi_dt - cfg_dphi_dt * st_star)
            output.sub_(guided * dt)
    
    def forward(
        self,
        mu: torch.Tensor,
        cond: torch.Tensor,
        temperature: torch.Tensor,
        cfg_value: torch.Tensor,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            n_timesteps (int): number of diffusion steps
            cond: Not used but kept for future purposes
            temperature (torch.Tensor): temperature for scaling noise. (batch_size,)
            cfg_value (torch.Tensor): cfg value for guidance. (batch_size,)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, c = mu.shape
        t = self.patch_size
        z = torch.randn((b, self.in_channels, t), device=mu.device, dtype=mu.dtype) * temperature[:, None, None]

        # Use CUDA graph if available for this batch size
        if self._graphs_captured and b in self._euler_graphs:
            return self._solve_euler_with_graph(z, mu, cond, cfg_value, b)
        
        # Fallback to eager execution
        t_span = self._t_span_base.to(dtype=mu.dtype)
        return self.solve_euler(z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value)
    
    def _solve_euler_with_graph(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: torch.Tensor,
        bs: int,
    ) -> torch.Tensor:
        """Run Euler solver using captured CUDA graph."""
        graph_vars = self._graph_vars[bs]
        
        # Copy inputs into graph buffers
        graph_vars['x'].copy_(z)
        graph_vars['mu'].copy_(mu)
        graph_vars['cond'].copy_(cond)
        graph_vars['cfg_value'].copy_(cfg_value)
        
        # Replay the captured graph
        self._euler_graphs[bs].replay()
        
        # Return output (clone to avoid aliasing issues)
        return graph_vars['output'].clone()

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        
        st_star = dot_product / squared_norm
        return st_star

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            cond: condition -- prefix prompt
            cfg_value (float, optional): cfg value for guidance. Defaults to 1.0.
        """
        b = x.size(0)
        seq_len = x.size(2)
        mu_dim = mu.size(1)
        device = x.device
        dtype = x.dtype
        
        # Pre-allocate CFG tensors once (reused across all timesteps)
        x_in = torch.empty([2 * b, self.in_channels, seq_len], device=device, dtype=dtype)
        mu_in = torch.zeros([2 * b, mu_dim], device=device, dtype=dtype)  # zeros for unconditional half
        t_in = torch.empty([2 * b], device=device, dtype=dtype)
        dt_in = torch.zeros([2 * b], device=device, dtype=dtype)
        cond_in = torch.empty([2 * b, self.in_channels, seq_len], device=device, dtype=dtype)
        
        # Pre-fill static parts (unconditional mu stays zero, cond is same for both halves)
        mu_in[:b] = mu
        cond_in[:b] = cond
        cond_in[b:] = cond
        
        t, _, dt = t_span[0], t_span[-1], t_span[0] - t_span[1]

        for step in range(1, len(t_span)):
            # Fill pre-allocated tensors (much faster than allocation)
            # Use copy_ and expand to stay GPU-side (no .item() - breaks CUDA graph capture)
            x_in[:b] = x
            x_in[b:] = x
            t_in[:] = t  # Broadcast scalar tensor to all elements
            if self.mean_mode:
                dt_in[:] = dt

            dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
            dphi_dt, cfg_dphi_dt = dphi_dt[:b], dphi_dt[b:]
            
            positive_flat = dphi_dt.view(b, -1)
            negative_flat = cfg_dphi_dt.view(b, -1)
            st_star = self.optimized_scale(positive_flat, negative_flat)
            st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))
            
            dphi_dt = cfg_dphi_dt * st_star + cfg_value[:, None, None] * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return x


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.empty(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)

        assert config.vocab_size == 0, "vocab_size must be 0 for local encoder"
        self.encoder = Cpm4Model(config, is_causal=False)

    def forward(self, x):
        """
        x: [T, P, D]
        """
        T, P, D = x.size()

        x = self.in_proj(x)
        special_tokens = self.special_token[0].expand(T, 1, -1)
        x = torch.cat([special_tokens, x], dim=1)
        position_ids = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        outputs = self.encoder(x, position_ids)
        cls_output = outputs[:, 0, :]

        return cls_output.view(T, -1)


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.scale = scale

        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)
    
    def forward(self, hidden):
        hidden = self.in_proj(hidden)
        hidden = torch.tanh(hidden)
        hidden = torch.round(hidden * self.scale) / self.scale

        return self.out_proj(hidden)


class VoxCPMModel(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: VoxCPMConfig,
        inference_timesteps: int,
        lora_config: LoRAConfig = None,
    ):
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.lora_config = lora_config

        assert not self.config.lm_config.use_mup, "mup inference is not supported now"

        # Text-Semantic LM
        self.base_lm = Cpm4Model(config.lm_config)

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        self.residual_lm = Cpm4Model(residual_lm_config)

        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            patch_size=config.patch_size,
            inference_timesteps=inference_timesteps,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiT(decoder_config, in_channels=config.feat_dim),
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size, 
            config.lm_config.hidden_size, 
            config.scalar_quantization_latent_dim, 
            config.scalar_quantization_scale
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
        
        # Apply LoRA if configured
        if self.lora_config is not None:
            self._apply_lora()
    
    def _apply_lora(self) -> None:
        """Inject LoRA into LM / DiT / projection layers.
        
        For attention layers, we add LoRA adapters that store separate q/k/v/o LoRA
        weights (matching the trained checkpoint structure) while working with our
        merged QKVParallelLinear architecture.
        """
        cfg = self.lora_config
        lora_kwargs = dict(r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)

        # LM: base_lm + residual_lm - Add LoRA adapters to attention layers
        if cfg.enable_lm:
            for lm in [self.base_lm, self.residual_lm]:
                for module in lm.modules():
                    if isinstance(module, Cpm4Attention):
                        module.add_lora_adapters(**lora_kwargs)

        # DiT: feat_decoder.estimator.decoder - Add LoRA adapters to attention layers
        if cfg.enable_dit:
            dit_decoder = self.feat_decoder.estimator.decoder
            for module in dit_decoder.modules():
                if isinstance(module, Cpm4Attention):
                    module.add_lora_adapters(**lora_kwargs)

        # Projection layers - These are nn.Linear so wrap with LoRALinear
        if cfg.enable_proj:
            for attr_name in cfg.target_proj_modules:
                module = getattr(self, attr_name, None)
                if isinstance(module, nn.Linear):
                    setattr(self, attr_name, LoRALinear(base=module, **lora_kwargs))
    
    # ------------------------------------------------------------------ #
    # LoRA Weight Management
    # ------------------------------------------------------------------ #
    
    # Mapping from trained LoRA weight names to our adapter structure
    # Trained: base_lm.layers.0.self_attn.q_proj.lora_A
    # Ours:    base_lm.layers.0.self_attn.qkv_lora.q_proj_lora_A
    LORA_KEY_MAPPING = {
        "q_proj.lora_A": "qkv_lora.q_proj_lora_A",
        "q_proj.lora_B": "qkv_lora.q_proj_lora_B",
        "k_proj.lora_A": "qkv_lora.k_proj_lora_A",
        "k_proj.lora_B": "qkv_lora.k_proj_lora_B",
        "v_proj.lora_A": "qkv_lora.v_proj_lora_A",
        "v_proj.lora_B": "qkv_lora.v_proj_lora_B",
        "o_proj.lora_A": "o_lora.lora_A",
        "o_proj.lora_B": "o_lora.lora_B",
    }
    
    def _iter_lora_modules(self):
        """Iterate over all LoRA modules."""
        for module in self.modules():
            if isinstance(module, (LoRALinear, QKVLoRAAdapter, OutputLoRAAdapter)):
                yield module

    def load_lora_weights(self, lora_state_dict: dict, device: str = "cuda"):
        """Load LoRA weights from a trained checkpoint.
        
        Maps trained LoRA weight keys (q_proj.lora_A, etc.) to our adapter
        structure (qkv_lora.q_proj_lora_A, etc.).

        Args:
            lora_state_dict: State dict containing lora_A and lora_B weights
            device: Target device

        Returns:
            tuple: (loaded_keys, skipped_keys)
        """
        # Build param mapping
        model_params = dict(self.named_parameters())

        loaded_keys, skipped_keys = [], []
        for key, value in lora_state_dict.items():
            # Try direct key first
            if key in model_params:
                model_params[key].data.copy_(value.to(device))
                loaded_keys.append(key)
                continue
            
            # Try mapping trained keys to our adapter structure
            mapped_key = None
            for old_suffix, new_suffix in self.LORA_KEY_MAPPING.items():
                if key.endswith(old_suffix):
                    mapped_key = key.replace(old_suffix, new_suffix)
                    break
            
            if mapped_key and mapped_key in model_params:
                model_params[mapped_key].data.copy_(value.to(device))
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        # Pre-compute delta matrices for fast inference (1 matmul instead of 2)
        self._compute_lora_deltas()

        return loaded_keys, skipped_keys

    def _compute_lora_deltas(self) -> None:
        """Pre-compute LoRA delta matrices for all adapters."""
        for module in self._iter_lora_modules():
            if hasattr(module, 'compute_deltas'):
                module.compute_deltas()

    def set_lora_enabled(self, enabled: bool) -> None:
        """Enable/disable all LoRA layers."""
        for module in self._iter_lora_modules():
            if hasattr(module, 'set_enabled'):
                module.set_enabled(enabled)

    def reset_lora_weights(self) -> None:
        """Reset all LoRA weights (A: kaiming, B: zeros), effectively unloading LoRA."""
        for module in self._iter_lora_modules():
            if hasattr(module, 'reset_lora_parameters'):
                module.reset_lora_parameters()

    def get_lora_state_dict(self) -> dict:
        """Get all LoRA parameters (lora_A/lora_B) and their pre-computed deltas."""
        state = {}
        for name, param in self.named_parameters():
            if "lora_" in name:
                state[name] = param.data.clone()
        # Also save pre-computed deltas
        for name, buffer in self.named_buffers():
            if "_delta" in name:
                state[name] = buffer.data.clone()
        return state
    
    def set_lora_state_dict(self, state_dict: dict, device: str = "cuda") -> None:
        """Set LoRA parameters and their pre-computed deltas from a state dict.
        
        This enables instant hotswapping between different LoRA weights.
        """
        model_params = dict(self.named_parameters())
        model_buffers = dict(self.named_buffers())
        
        for key, value in state_dict.items():
            if key in model_params:
                model_params[key].data.copy_(value.to(device))
            elif key in model_buffers:
                model_buffers[key].data.copy_(value.to(device))
    
    def forward(
            self,
            positions : torch.Tensor,
            text_tokens : torch.Tensor,
            feat : torch.Tensor,
            feat_mask : torch.Tensor,
            temperature : torch.Tensor,
            cfg_value : torch.Tensor,
        ):
        """
        text_tokens: [T]
        feat: [T, P, D]
        feat_mask: [T]
        temperature: [B]
        cfg_value: [B]
        """
        feat_embeds = self.feat_encoder(feat)
        feat_embeds = self.enc_to_lm_proj(feat_embeds)
        feat_embeds = torch.masked_fill(feat_embeds, feat_mask.unsqueeze(-1).logical_not(), 0)

        text_embeds = self.base_lm.embed_tokens(text_tokens)
        combined_embeds = torch.where(
            feat_mask.unsqueeze(-1),
            feat_embeds,
            text_embeds,
        )

        enc_outputs = self.base_lm(combined_embeds, positions)
        enc_outputs = torch.where(
            feat_mask.unsqueeze(-1),
            self.fsq_layer(enc_outputs),
            enc_outputs,
        )

        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            lm_hidden = enc_outputs[last_indices].contiguous()
        else:
            lm_hidden = enc_outputs
        

        ralm_outputs = self.residual_lm(
            enc_outputs + feat_embeds, 
            positions,
        )

        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            ralm_hidden = ralm_outputs[last_indices].contiguous()
            # (b, P, D)
            prefix_feat_cond = feat[last_indices].contiguous()
        else:
            ralm_hidden = ralm_outputs
            # (b, P, D)
            prefix_feat_cond = feat
        
        dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
        dit_hidden_2 = self.res_to_dit_proj(ralm_hidden)  # [b, h_dit]
        dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit]

        # (b, P, D)
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            temperature=temperature,
            cfg_value=cfg_value,
        ).transpose(1, 2)

        stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)

        return {
            "latents":  pred_feat, 
            "stop_flag": stop_flag,
        }
