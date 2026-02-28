from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from attn_arena.attention.mha import MHAFactory
from attn_arena.inference.reporting import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    benchmark_result_to_dict,
    write_benchmark_report_csv,
    write_benchmark_report_json,
)
from attn_arena.inference.runner import (
    BenchmarkRunConfig,
    BenchmarkWorkload,
    InferenceBenchmarkResult,
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


def _run_tiny_benchmark() -> InferenceBenchmarkResult:
    config = _tiny_llama_config()
    model = LlamaBackbone(config)
    model.set_attention(MHAFactory(config))
    return run_prefill_decode_benchmark(
        model=model,
        workload=BenchmarkWorkload(
            batch_size=2,
            prefill_len=4,
            decode_len=2,
            max_seq_len=12,
            vocab_size=config.vocab_size,
        ),
        run_config=BenchmarkRunConfig(
            warmup_iters=0,
            timed_iters=1,
            dtype=torch.float32,
            device="cpu",
            weights_mode="synthetic",
        ),
    )


def test_benchmark_result_to_dict_contains_schema_and_metadata() -> None:
    result = _run_tiny_benchmark()
    payload = benchmark_result_to_dict(result)

    assert payload["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert payload["metadata"]["weights_mode"] == "synthetic"
    assert payload["metadata"]["model_name"] == "LlamaBackbone"
    assert payload["metadata"]["attention_name"] == "MultiHeadAttention"
    assert payload["metadata"]["attention_backend"] == "sdpa"
    assert payload["metrics"]["total_tokens"] == result.total_tokens
    assert payload["metrics"]["persistent_kv_cache_bytes"] == result.kv_cache_bytes


def test_write_benchmark_report_json_writes_schema_payload(tmp_path: Path) -> None:
    result = _run_tiny_benchmark()
    output = tmp_path / "report.json"
    path = write_benchmark_report_json(result, output)
    assert path == output

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert payload["metadata"]["weights_mode"] == "synthetic"
    assert payload["metadata"]["attention_backend"] == "sdpa"


def test_write_benchmark_report_csv_writes_flat_rows(tmp_path: Path) -> None:
    result = _run_tiny_benchmark()
    output = tmp_path / "report.csv"
    path = write_benchmark_report_csv([result], output)
    assert path == output

    with output.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == BENCHMARK_REPORT_SCHEMA_VERSION
    assert row["weights_mode"] == "synthetic"
    assert row["model_name"] == "LlamaBackbone"
    assert row["attention_backend"] == "sdpa"
