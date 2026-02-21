from __future__ import annotations

import torch

from attn_arena.attention.base import (
    AttentionCapabilities,
    AttentionModule,
    AttentionOutput,
    KVCache,
)
from attn_arena.models.base import ModelBackbone


class DummyKVCache:
    def update(self, new_kv: torch.Tensor, layer_idx: int, position: torch.LongTensor) -> None:
        return None

    def get(self, layer_idx: int) -> torch.Tensor:
        return torch.zeros(2, 1, 1, 1, 1)

    def size_bytes(self) -> int:
        return 0

    def current_seq_len(self) -> int:
        return 0

    def clear(self) -> None:
        return None


class DummyAttention:
    capabilities = AttentionCapabilities(
        supports_prefill=True,
        supports_decode=True,
        supports_tensor_parallel=True,
        cache_type="full_kv",
        kv_compression_ratio=1.0,
        requires_rope_dim=False,
        max_tested_seq_len=256,
    )

    def prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        kv_cache: KVCache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> AttentionOutput:
        return AttentionOutput(output=hidden_states)

    def decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCache,
        cache_position: torch.LongTensor,
    ) -> AttentionOutput:
        return AttentionOutput(output=hidden_states)

    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        return DummyKVCache()

    def shard(self, tp_rank: int, tp_world_size: int) -> DummyAttention:
        return self


class DummyModel:
    def set_attention(self, attention: AttentionModule) -> None:
        self.attention = attention

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.zeros(batch_size, seq_len, 8)

    def get_rope_cos_sin(
        self,
        position_ids: torch.LongTensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(seq_len, 4), torch.zeros(seq_len, 4)

    def shard(self, tp_rank: int, tp_world_size: int) -> DummyModel:
        return self


def test_runtime_checkable_attention_contract() -> None:
    assert isinstance(DummyAttention(), AttentionModule)


def test_runtime_checkable_kv_cache_contract() -> None:
    assert isinstance(DummyKVCache(), KVCache)


def test_runtime_checkable_model_contract() -> None:
    assert isinstance(DummyModel(), ModelBackbone)
