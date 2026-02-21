from __future__ import annotations

import pytest
import torch

import attn_arena.models.registry as model_registry
from attn_arena.attention.base import AttentionModule, KVCache


@pytest.fixture(autouse=True)
def clean_model_registry() -> None:
    snapshot = dict(model_registry._MODEL_REGISTRY)
    model_registry._MODEL_REGISTRY.clear()
    yield
    model_registry._MODEL_REGISTRY.clear()
    model_registry._MODEL_REGISTRY.update(snapshot)


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
        cos = torch.ones(seq_len, 4)
        sin = torch.zeros(seq_len, 4)
        return cos, sin

    def shard(self, tp_rank: int, tp_world_size: int) -> DummyModel:
        return self


def test_register_and_get_model() -> None:
    model_registry.register_model("dummy_model")(DummyModel)
    instance = model_registry.get_model("dummy_model")
    assert isinstance(instance, DummyModel)
    assert model_registry.has_model("dummy_model")
    assert model_registry.list_models() == ("dummy_model",)


def test_register_duplicate_model_raises() -> None:
    model_registry.register_model("dummy_model")(DummyModel)
    with pytest.raises(KeyError, match="already registered"):
        model_registry.register_model("dummy_model")(DummyModel)


def test_get_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        model_registry.get_model("missing_model")


def test_register_empty_model_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        model_registry.register_model(" ")
