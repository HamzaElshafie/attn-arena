"""Inference orchestration package.

This package will host reusable prefill/decode runners and benchmark-oriented
execution utilities. It is kept separate from model backbones and attention
variants so orchestration concerns (cache lifecycle, decoding loops, request
state) do not leak into math modules.
"""

from attn_arena.inference.runner import (
    BenchmarkRunConfig,
    BenchmarkWorkload,
    InferenceBenchmarkResult,
    StageMetrics,
    SyntheticTokenSpec,
    build_position_ids,
    init_kv_caches_for_model,
    make_synthetic_input_ids,
    run_prefill_decode_benchmark,
    total_kv_cache_bytes,
)

__all__ = [
    "BenchmarkRunConfig",
    "BenchmarkWorkload",
    "InferenceBenchmarkResult",
    "StageMetrics",
    "SyntheticTokenSpec",
    "build_position_ids",
    "init_kv_caches_for_model",
    "make_synthetic_input_ids",
    "run_prefill_decode_benchmark",
    "total_kv_cache_bytes",
]
