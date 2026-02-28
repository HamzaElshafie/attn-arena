from __future__ import annotations

import pytest
import torch

from attn_arena.attention.gqa import GQAFactory, GroupedQueryAttention
from attn_arena.models.llama.config import LlamaConfig
from attn_arena.models.llama.model import LlamaBackbone


def _tiny_gqa_config() -> LlamaConfig:
    return LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=64,
        max_seq_len=16,
        multiple_of=8,
    )


def _dummy_position_embeddings(
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = torch.ones(batch_size, seq_len, head_dim, device=device)
    sin = torch.zeros(batch_size, seq_len, head_dim, device=device)
    return cos, sin


def test_gqa_init_rejects_invalid_head_configuration() -> None:
    config = LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=64,
        max_seq_len=16,
        multiple_of=8,
    )
    with pytest.raises(ValueError, match="must be < n_heads"):
        GroupedQueryAttention(config)


def test_gqa_prefill_and_decode_smoke() -> None:
    config = _tiny_gqa_config()
    gqa = GroupedQueryAttention(config)

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.dim)
    position_embeddings = _dummy_position_embeddings(
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=config.head_dim,
        device=hidden_states.device,
    )

    output_no_cache = gqa.prefill(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        kv_cache=None,
        cache_position=None,
    )
    assert output_no_cache.output.shape == (batch_size, seq_len, config.dim)

    kv_cache = gqa.init_kv_cache(
        batch_size=batch_size,
        max_seq_len=config.max_seq_len,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache_position = torch.arange(seq_len, dtype=torch.long)
    output_with_cache = gqa.prefill(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        kv_cache=kv_cache,
        cache_position=cache_position,
    )
    assert output_with_cache.output.shape == (batch_size, seq_len, config.dim)
    assert kv_cache.current_seq_len() == seq_len

    decode_hidden_states = torch.randn(batch_size, 1, config.dim)
    decode_position_embeddings = _dummy_position_embeddings(
        batch_size=batch_size,
        seq_len=1,
        head_dim=config.head_dim,
        device=decode_hidden_states.device,
    )
    decode_output = gqa.decode(
        hidden_states=decode_hidden_states,
        position_embeddings=decode_position_embeddings,
        kv_cache=kv_cache,
        cache_position=torch.tensor([seq_len], dtype=torch.long),
    )
    assert decode_output.output.shape == (batch_size, 1, config.dim)
    assert kv_cache.current_seq_len() == seq_len + 1


def test_llama_backbone_with_gqa_factory_smoke() -> None:
    config = _tiny_gqa_config()
    model = LlamaBackbone(config)
    model.set_attention(GQAFactory(config))

    input_ids = torch.randint(0, config.vocab_size, (2, 5), dtype=torch.long)
    position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0).expand(2, -1)

    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        kv_caches=None,
        attention_mask=None,
        cache_position=None,
    )
    assert logits.shape == (2, 5, config.vocab_size)
