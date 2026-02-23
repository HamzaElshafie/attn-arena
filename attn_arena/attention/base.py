from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch

CacheType = Literal["full_kv", "latent", "tied"]


@dataclass(frozen=True)
class AttentionCapabilities:
    """Static feature flags describing an attention variant."""
    
    supports_prefill: bool
    supports_decode: bool
    supports_tensor_parallel: bool
    cache_type: CacheType
    kv_compression_ratio: float
    requires_rope_dim: bool
    max_tested_seq_len: int


@dataclass(frozen=True)
class AttentionOutput:
    """Attention output tensor for a single forward call."""

    output: torch.Tensor


@runtime_checkable
class KVCache(Protocol):
    """Variant-owned KV cache interface.

    The cache tensor representation is intentionally variant-defined.
    Convention for `cache_type="full_kv"` variants: `new_kv` / `get()` return a
    stacked tensor where the leading dimension indexes (K, V) with size 2.
    """

    def update(
        self,
        new_kv: torch.Tensor,
        layer_idx: int,
        position: torch.LongTensor,
    ) -> None:
        """Update cache state for `layer_idx` at `position` (shape depends on mode)."""

    def get(self, layer_idx: int) -> torch.Tensor:
        """Return the cache tensor for `layer_idx` (representation is variant-defined)."""

    def size_bytes(self) -> int:
        """Return the cache payload size in bytes (used for memory benchmarking)."""

    def current_seq_len(self) -> int:
        """Return the current cached sequence length."""

    def clear(self) -> None:
        """Clear cached state."""


@runtime_checkable
class AttentionModule(Protocol):
    """Attention variant contract (no inheritance; structural typing via Protocol)."""

    capabilities: AttentionCapabilities

    def prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> AttentionOutput: ...

    def decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCache,
        cache_position: torch.LongTensor,
    ) -> AttentionOutput: ...

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache: ...

    def shard(self, tp_rank: int, tp_world_size: int) -> AttentionModule: ...
