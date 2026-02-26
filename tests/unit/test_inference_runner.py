from __future__ import annotations

import torch

from attn_arena.attention.mha import MHAFactory
from attn_arena.inference.runner import (
    BenchmarkRunConfig,
    BenchmarkWorkload,
    SyntheticTokenSpec,
    make_synthetic_input_ids,
    run_prefill_decode_benchmark,
)
from attn_arena.models.llama.config import LlamaConfig
from attn_arena.models.llama.model import LlamaBackbone


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=64,
        max_seq_len=32,
        multiple_of=8,
    )


def test_make_synthetic_input_ids_is_deterministic() -> None:
    device = torch.device("cpu")
    spec = SyntheticTokenSpec(seed=123, offset=7)
    tokens_a = make_synthetic_input_ids(
        batch_size=2,
        seq_len=5,
        vocab_size=64,
        spec=spec,
        device=device,
    )
    tokens_b = make_synthetic_input_ids(
        batch_size=2,
        seq_len=5,
        vocab_size=64,
        spec=spec,
        device=device,
    )
    tokens_c = make_synthetic_input_ids(
        batch_size=2,
        seq_len=5,
        vocab_size=64,
        spec=SyntheticTokenSpec(seed=124, offset=7),
        device=device,
    )

    assert torch.equal(tokens_a, tokens_b)
    assert not torch.equal(tokens_a, tokens_c)


def test_run_prefill_decode_benchmark_smoke() -> None:
    config = _tiny_llama_config()
    model = LlamaBackbone(config)
    model.set_attention(MHAFactory(config))

    result = run_prefill_decode_benchmark(
        model=model,
        workload=BenchmarkWorkload(
            batch_size=2,
            prefill_len=4,
            decode_len=3,
            max_seq_len=16,
            vocab_size=config.vocab_size,
        ),
        run_config=BenchmarkRunConfig(
            warmup_iters=0,
            timed_iters=1,
            dtype=torch.float32,
            device="cpu",
        ),
        token_spec=SyntheticTokenSpec(seed=0),
    )

    assert result.prefill.total_tokens == 8
    assert result.decode.total_tokens == 6
    assert result.total_tokens == 14
    assert result.prefill.elapsed_seconds >= 0.0
    assert result.decode.elapsed_seconds >= 0.0
    assert result.total_elapsed_seconds >= 0.0
    assert result.kv_cache_bytes > 0
