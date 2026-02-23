import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from attn_arena.attention.base import AttentionModule, KVCache
from attn_arena.models.llama.config import LlamaConfig

class RMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm * self.weight).type_as(x)

class FeedForward(nn.Module):
    pass

class TransformerBlock(nn.Module):
    pass

class LlamaBackbone(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.attention = AttentionModule | None = None
        self._rope_cache_len = 0
        self._rope_cos_cache: torch.Tensor | None = None
        self._rope_sin_cache: torch.Tensor | None = None
        self._rope_cache_device: torch.device | None = None
        

    def set_attention(self, attention: AttentionModule) -> None:
        if self.attention is not None:
            raise ValueError("Attention already set on LlamaBackbone.")
        self.attention = attention

    def get_rope_cos_sin(
        self,
        position_ids: torch.LongTensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RoPE coefficients as `(cos, sin)` for the provided positions."""
        max_pos = int(position_ids.max().item()) + 1
        required_len = max(seq_len, max_pos)
        device = position_ids.device

        if (
            self._rope_cos_cache is None
            or self._rope_sin_cache is None
            or self._rope_cache_device != device
            or self._rope_cache_len < required_len
        ):
            dim = self.config.head_dim
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
            )
            t = torch.arange(required_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._rope_cos_cache = emb.cos()
            self._rope_sin_cache = emb.sin()
            self._rope_cache_len = required_len
            self._rope_cache_device = device

        cos = self._rope_cos_cache[position_ids]
        sin = self._rope_sin_cache[position_ids]
        return cos.unsqueeze(2), sin.unsqueeze(2)
        
    def shard(self, tp_rank: int, tp_world_size: int) -> "LlamaBackbone":
      """Return a new model instance configured for tensor-parallel rank."""

    def forward(self, x: torch.Tensor):
        pass
