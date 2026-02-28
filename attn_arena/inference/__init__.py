"""Inference orchestration package.

This package will host reusable prefill/decode runners and benchmark-oriented
execution utilities. It is kept separate from model backbones and attention
variants so orchestration concerns (cache lifecycle, decoding loops, request
state) do not leak into math modules.
"""

from attn_arena.inference.reporting import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    benchmark_result_to_dict,
    write_benchmark_report_csv,
    write_benchmark_report_json,
)
from attn_arena.inference.runner import (
    AttentionBackend,
    BenchmarkMetadata,
    BenchmarkRunConfig,
    BenchmarkWorkload,
    InferenceBenchmarkResult,
    StageMetrics,
    SyntheticInitConfig,
    SyntheticInitPolicy,
    SyntheticTokenSpec,
    WeightsMode,
    build_position_ids,
    init_kv_caches_for_model,
    initialize_model_weights_for_synthetic_mode,
    make_synthetic_input_ids,
    run_prefill_decode_benchmark,
    total_kv_cache_bytes,
)

__all__ = [
    "BENCHMARK_REPORT_SCHEMA_VERSION",
    "AttentionBackend",
    "BenchmarkMetadata",
    "BenchmarkRunConfig",
    "BenchmarkWorkload",
    "InferenceBenchmarkResult",
    "StageMetrics",
    "SyntheticInitConfig",
    "SyntheticInitPolicy",
    "SyntheticTokenSpec",
    "WeightsMode",
    "benchmark_result_to_dict",
    "build_position_ids",
    "init_kv_caches_for_model",
    "initialize_model_weights_for_synthetic_mode",
    "make_synthetic_input_ids",
    "run_prefill_decode_benchmark",
    "total_kv_cache_bytes",
    "write_benchmark_report_csv",
    "write_benchmark_report_json",
]
