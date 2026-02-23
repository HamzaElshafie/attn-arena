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
from attn_arena.attention.registry import register_attention
from attn_arena.models.llama.config import LlamaConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        cos = cos.unsqueeze(2)
    if sin.dim() == 3:
        sin = sin.unsqueeze(2)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class FullKVCache:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._cache = torch.zeros(
            (2, batch_size, max_seq_len, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._current_seq_len = 0

    def update(
        self,
        new_kv: torch.Tensor,
        layer_idx: int,
        position: torch.LongTensor,
    ) -> None:
        _ = layer_idx
        if new_kv.dim() != 5:
            raise ValueError("new_kv must have shape (2, B, T, H, D).")
        if new_kv.shape[0] != 2:
            raise ValueError("new_kv must have leading dimension size 2 for (K, V).")

        seq_len = new_kv.shape[2]
        if position.numel() == 1:
            start = int(position.item())
            end = start + seq_len
            if end > self.max_seq_len:
                raise ValueError("cache update would exceed max_seq_len.")
            self._cache[:, :, start:end, :, :] = new_kv
            self._current_seq_len = max(self._current_seq_len, end)
            return

        if position.dim() != 1:
            raise ValueError("position must be a scalar or 1D tensor.")
        if position.numel() != seq_len:
            raise ValueError("position length must match new_kv seq_len.")

        if int(position.max().item()) >= self.max_seq_len:
            raise ValueError("cache update would exceed max_seq_len.")

        self._cache[:, :, position, :, :] = new_kv
        self._current_seq_len = max(
            self._current_seq_len,
            int(position.max().item()) + 1,
        )

    def get(self, layer_idx: int) -> torch.Tensor:
        _ = layer_idx
        return self._cache

    def size_bytes(self) -> int:
        return self._cache.numel() * self._cache.element_size()

    def current_seq_len(self) -> int:
        return self._current_seq_len

    def clear(self) -> None:
        self._cache.zero_()
        self._current_seq_len = 0


@register_attention("mha")
class MultiHeadAttention(nn.Module):
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

        if self.n_kv_heads != self.n_heads:
            raise ValueError("n_kv_heads must be == n_heads in MHA")
        if self.n_heads * self.head_dim != self.dim:
            raise ValueError("dim must equal n_heads * head_dim")

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
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
        k = self.wk(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.wv(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            if cache_position is None:
                raise ValueError("cache_position must be provided when kv_cache is used in prefill.")
            new_kv = torch.stack([k, v], dim=0)
            kv_cache.update(new_kv, layer_idx=0, position=cache_position)
            kv = kv_cache.get(layer_idx=0)
            cache_len = kv_cache.current_seq_len()
            k = kv[0, :, :cache_len, :, :]
            v = kv[1, :, :cache_len, :, :]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
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
            raise ValueError("decode expects seq_len == 1")

        q = self.wq(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.wv(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim)

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

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
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
            num_kv_heads=self.n_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

    def shard(self, tp_rank: int, tp_world_size: int) -> AttentionModule:
        _ = tp_rank
        if tp_world_size == 1:
            return self
        raise NotImplementedError("Tensor parallel sharding is not implemented yet.")
