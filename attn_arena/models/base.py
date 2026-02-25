from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from attn_arena.attention.base import AttentionFactory, KVCache


@runtime_checkable
class ModelBackbone(Protocol):
    """Backbone contract for decoder-only model families."""

    def set_attention(self, attention_factory: AttentionFactory) -> None:
        """Inject the attention implementation used by this backbone."""

    def get_rope_cos_sin(
        self,
        position_ids: torch.LongTensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RoPE coefficients as `(cos, sin)` for the provided positions."""

    def shard(self, tp_rank: int, tp_world_size: int) -> ModelBackbone:
        """Return a new model instance configured for tensor-parallel rank."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Run model forward and return logits."""
