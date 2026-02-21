from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import torch

from attn_arena.attention.base import AttentionModule, KVCache


@runtime_checkable
class ModelBackbone(Protocol):
    """Backbone contract for decoder-only model families."""

    def set_attention(self, attention: AttentionModule) -> None:
        """Inject the attention implementation used by this backbone."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        kv_caches: Optional[list[KVCache]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Run model forward and return logits."""

    def get_rope_cos_sin(
        self,
        position_ids: torch.LongTensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return RoPE coefficients as `(cos, sin)` for the provided positions."""

    def shard(self, tp_rank: int, tp_world_size: int) -> "ModelBackbone":
        """Return a new model instance configured for tensor-parallel rank."""
