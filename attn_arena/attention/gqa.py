from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attn_arena.attention.base import (
    AttentionCapabilities,
    AttentionModule,
    AttentionOutput,
    KVCache,
)
from attn_arena.attention.mha import FullKVCache, apply_rotary_pos_emb
from attn_arena.attention.registry import register_attention
from attn_arena.models.llama.config import LlamaConfig


def _repeat_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand grouped KV heads to match query-head count for attention compute."""

    if n_groups <= 0:
        raise ValueError("n_groups must be > 0.")

    batch_size, seq_len, n_kv_heads, head_dim = k.shape
    k_expanded = (
        k.unsqueeze(3)
        .expand(batch_size, seq_len, n_kv_heads, n_groups, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_groups, head_dim)
    )
    v_expanded = (
        v.unsqueeze(3)
        .expand(batch_size, seq_len, n_kv_heads, n_groups, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_groups, head_dim)
    )
    return k_expanded, v_expanded


def _sdpa_with_grouped_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_groups: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run SDPA for GQA using grouped-KV fast path with fallback.

    Shapes:
    - q: [B, Hq, Tq, D]
    - k/v: [B, Hkv, Tk, D]
    """

    try:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )
    except TypeError:
        k_expanded, v_expanded = _repeat_kv(
            k.transpose(1, 2),
            v.transpose(1, 2),
            n_groups=n_groups,
        )
        return F.scaled_dot_product_attention(
            q,
            k_expanded.transpose(1, 2),
            v_expanded.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )


@register_attention("gqa")
class GroupedQueryAttention(nn.Module):
    capabilities = AttentionCapabilities(
        supports_prefill=True,
        supports_decode=True,
        supports_tensor_parallel=False,
        cache_type="full_kv",
        kv_compression_ratio=1.0,
        requires_rope_dim=False,
        max_tested_seq_len=0,
    )

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.effective_num_kv_heads
        self.head_dim = config.head_dim

        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0 in GQA.")
        if self.n_kv_heads >= self.n_heads:
            raise ValueError("n_kv_heads must be < n_heads in GQA.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads in GQA.")
        if self.n_heads * self.head_dim != self.dim:
            raise ValueError("dim must equal n_heads * head_dim.")

        self.n_groups = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.wq(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            if cache_position is None:
                raise ValueError(
                    "cache_position must be provided when kv_cache is used in prefill."
                )
            new_kv = torch.stack([k, v], dim=0)
            kv_cache.update(new_kv, layer_idx=0, position=cache_position)
            kv = kv_cache.get(layer_idx=0)
            cache_len = kv_cache.current_seq_len()
            k = kv[0, :, :cache_len, :, :]
            v = kv[1, :, :cache_len, :, :]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = _sdpa_with_grouped_kv(
            q,
            k,
            v,
            n_groups=self.n_groups,
            attention_mask=attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return AttentionOutput(self.wo(attn_output))

    def decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCache,
        cache_position: torch.LongTensor,
    ) -> AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise ValueError("decode expects seq_len == 1.")

        q = self.wq(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        new_kv = torch.stack([k, v], dim=0)
        kv_cache.update(new_kv, layer_idx=0, position=cache_position)
        kv = kv_cache.get(layer_idx=0)
        cache_len = kv_cache.current_seq_len()
        k = kv[0, :, :cache_len, :, :]
        v = kv[1, :, :cache_len, :, :]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = _sdpa_with_grouped_kv(
            q,
            k,
            v,
            n_groups=self.n_groups,
            attention_mask=None,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return AttentionOutput(self.wo(attn_output))

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        return FullKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

    def shard(self, tp_rank: int, tp_world_size: int) -> AttentionModule:
        _ = tp_rank
        if tp_world_size == 1:
            return self
        raise NotImplementedError("Tensor parallel sharding is not implemented yet.")


class GQAFactory:
    """Create a fresh GQA module per layer to avoid parameter sharing."""

    def __init__(self, config: LlamaConfig) -> None:
        self.config = config

    def create(self, layer_idx: int) -> AttentionModule:
        _ = layer_idx
        return GroupedQueryAttention(self.config)
