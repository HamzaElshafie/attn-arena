from __future__ import annotations

import torch

from attn_arena.attention.mha import MHAFactory
from attn_arena.inference.runner import (
    BenchmarkRunConfig,
    BenchmarkWorkload,
    SyntheticInitConfig,
    SyntheticTokenSpec,
    initialize_model_weights_for_synthetic_mode,
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


def test_initialize_synthetic_weights_is_deterministic() -> None:
    config = _tiny_llama_config()

    model_a = LlamaBackbone(config)
    model_a.set_attention(MHAFactory(config))
    initialize_model_weights_for_synthetic_mode(
        model_a,
        config=SyntheticInitConfig(policy="xavier_uniform", seed=11),
    )

    model_b = LlamaBackbone(config)
    model_b.set_attention(MHAFactory(config))
    initialize_model_weights_for_synthetic_mode(
        model_b,
        config=SyntheticInitConfig(policy="xavier_uniform", seed=11),
    )

    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    assert params_a.keys() == params_b.keys()
    for name in params_a:
        assert torch.equal(params_a[name], params_b[name]), name


def test_initialize_synthetic_weights_changes_with_seed() -> None:
    config = _tiny_llama_config()

    model_a = LlamaBackbone(config)
    model_a.set_attention(MHAFactory(config))
    initialize_model_weights_for_synthetic_mode(
        model_a,
        config=SyntheticInitConfig(policy="xavier_uniform", seed=11),
    )

    model_b = LlamaBackbone(config)
    model_b.set_attention(MHAFactory(config))
    initialize_model_weights_for_synthetic_mode(
        model_b,
        config=SyntheticInitConfig(policy="xavier_uniform", seed=12),
    )

    any_difference = any(
        not torch.equal(param_a, param_b)
        for (_, param_a), (_, param_b) in zip(
            model_a.named_parameters(),
            model_b.named_parameters(),
            strict=True,
        )
    )
    assert any_difference


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
    assert result.metadata.weights_mode == "synthetic"
    assert result.metadata.model_name == "LlamaBackbone"
    assert result.metadata.attention_name == "MultiHeadAttention"
    assert result.metadata.attention_backend == "sdpa"
    assert result.metadata.synthetic_init_policy == "xavier_uniform"
    assert result.metadata.synthetic_init_seed == 0


def test_run_prefill_decode_benchmark_native_metadata() -> None:
    config = _tiny_llama_config()
    model = LlamaBackbone(config)
    model.set_attention(MHAFactory(config))

    result = run_prefill_decode_benchmark(
        model=model,
        workload=BenchmarkWorkload(
            batch_size=1,
            prefill_len=2,
            decode_len=1,
            max_seq_len=8,
            vocab_size=config.vocab_size,
        ),
        run_config=BenchmarkRunConfig(
            warmup_iters=0,
            timed_iters=1,
            dtype=torch.float32,
            device="cpu",
            weights_mode="native",
            checkpoint_source="meta-llama/Meta-Llama-3-8B",
        ),
        token_spec=SyntheticTokenSpec(seed=0),
    )

    assert result.metadata.weights_mode == "native"
    assert result.metadata.attention_backend == "sdpa"
    assert result.metadata.synthetic_init_policy is None
    assert result.metadata.synthetic_init_seed is None
    assert result.metadata.checkpoint_source == "meta-llama/Meta-Llama-3-8B"
