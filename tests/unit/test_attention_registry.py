from __future__ import annotations

import pytest
import torch

import attn_arena.attention.registry as attention_registry
from attn_arena.attention.base import AttentionCapabilities, AttentionOutput, KVCache


@pytest.fixture(autouse=True)
def clean_attention_registry() -> None:
    snapshot = dict(attention_registry._ATTENTION_REGISTRY)
    attention_registry._ATTENTION_REGISTRY.clear()
    yield
    attention_registry._ATTENTION_REGISTRY.clear()
    attention_registry._ATTENTION_REGISTRY.update(snapshot)


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
        supports_tensor_parallel=False,
        cache_type="full_kv",
        kv_compression_ratio=1.0,
        requires_rope_dim=False,
        max_tested_seq_len=128,
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


def test_register_and_get_attention() -> None:
    attention_registry.register_attention("dummy_attention")(DummyAttention)
    instance = attention_registry.get_attention("dummy_attention")
    assert isinstance(instance, DummyAttention)
    assert attention_registry.has_attention("dummy_attention")
    assert attention_registry.list_attentions() == ("dummy_attention",)


def test_register_duplicate_attention_raises() -> None:
    attention_registry.register_attention("dummy_attention")(DummyAttention)
    with pytest.raises(KeyError, match="already registered"):
        attention_registry.register_attention("dummy_attention")(DummyAttention)


def test_get_unknown_attention_raises() -> None:
    with pytest.raises(KeyError, match="unknown attention"):
        attention_registry.get_attention("missing_attention")


def test_register_empty_attention_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        attention_registry.register_attention("   ")
