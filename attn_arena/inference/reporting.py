from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from attn_arena.inference.runner import InferenceBenchmarkResult

BENCHMARK_REPORT_SCHEMA_VERSION = "attn_arena.benchmark.v1"


def benchmark_result_to_dict(result: InferenceBenchmarkResult) -> dict[str, Any]:
    """Convert a benchmark result to a schema-versioned dictionary."""

    return {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "metadata": {
            "model_name": result.metadata.model_name,
            "attention_name": result.metadata.attention_name,
            "attention_backend": result.metadata.attention_backend,
            "weights_mode": result.metadata.weights_mode,
            "device": result.metadata.device,
            "dtype": result.metadata.dtype,
            "synthetic_init_policy": result.metadata.synthetic_init_policy,
            "synthetic_init_seed": result.metadata.synthetic_init_seed,
            "checkpoint_source": result.metadata.checkpoint_source,
        },
        "workload": {
            "batch_size": result.workload.batch_size,
            "prefill_len": result.workload.prefill_len,
            "decode_len": result.workload.decode_len,
            "max_seq_len": result.workload.max_seq_len,
            "vocab_size": result.workload.vocab_size,
        },
        "metrics": {
            "prefill_elapsed_seconds": result.prefill.elapsed_seconds,
            "prefill_total_tokens": result.prefill.total_tokens,
            "prefill_tokens_per_second": result.prefill.tokens_per_second,
            "decode_elapsed_seconds": result.decode.elapsed_seconds,
            "decode_total_tokens": result.decode.total_tokens,
            "decode_tokens_per_second": result.decode.tokens_per_second,
            "total_elapsed_seconds": result.total_elapsed_seconds,
            "total_tokens": result.total_tokens,
            "total_tokens_per_second": result.total_tokens_per_second,
            "kv_cache_bytes": result.kv_cache_bytes,
            "persistent_kv_cache_bytes": result.persistent_kv_cache_bytes,
        },
    }


def write_benchmark_report_json(
    result: InferenceBenchmarkResult,
    output_path: str | Path,
) -> Path:
    """Write a single benchmark result report as JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = benchmark_result_to_dict(result)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


def write_benchmark_report_csv(
    results: Iterable[InferenceBenchmarkResult],
    output_path: str | Path,
) -> Path:
    """Write benchmark results as flattened rows in CSV format."""

    rows = [benchmark_result_to_dict(result) for result in results]
    if not rows:
        raise ValueError("At least one result is required to write CSV output.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "schema_version",
        "model_name",
        "attention_name",
        "attention_backend",
        "weights_mode",
        "device",
        "dtype",
        "synthetic_init_policy",
        "synthetic_init_seed",
        "checkpoint_source",
        "batch_size",
        "prefill_len",
        "decode_len",
        "max_seq_len",
        "vocab_size",
        "prefill_elapsed_seconds",
        "prefill_total_tokens",
        "prefill_tokens_per_second",
        "decode_elapsed_seconds",
        "decode_total_tokens",
        "decode_tokens_per_second",
        "total_elapsed_seconds",
        "total_tokens",
        "total_tokens_per_second",
        "kv_cache_bytes",
        "persistent_kv_cache_bytes",
    ]

    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metadata = row["metadata"]
            workload = row["workload"]
            metrics = row["metrics"]
            writer.writerow(
                {
                    "schema_version": row["schema_version"],
                    "model_name": metadata["model_name"],
                    "attention_name": metadata["attention_name"],
                    "attention_backend": metadata["attention_backend"],
                    "weights_mode": metadata["weights_mode"],
                    "device": metadata["device"],
                    "dtype": metadata["dtype"],
                    "synthetic_init_policy": metadata["synthetic_init_policy"],
                    "synthetic_init_seed": metadata["synthetic_init_seed"],
                    "checkpoint_source": metadata["checkpoint_source"],
                    "batch_size": workload["batch_size"],
                    "prefill_len": workload["prefill_len"],
                    "decode_len": workload["decode_len"],
                    "max_seq_len": workload["max_seq_len"],
                    "vocab_size": workload["vocab_size"],
                    "prefill_elapsed_seconds": metrics["prefill_elapsed_seconds"],
                    "prefill_total_tokens": metrics["prefill_total_tokens"],
                    "prefill_tokens_per_second": metrics["prefill_tokens_per_second"],
                    "decode_elapsed_seconds": metrics["decode_elapsed_seconds"],
                    "decode_total_tokens": metrics["decode_total_tokens"],
                    "decode_tokens_per_second": metrics["decode_tokens_per_second"],
                    "total_elapsed_seconds": metrics["total_elapsed_seconds"],
                    "total_tokens": metrics["total_tokens"],
                    "total_tokens_per_second": metrics["total_tokens_per_second"],
                    "kv_cache_bytes": metrics["kv_cache_bytes"],
                    "persistent_kv_cache_bytes": metrics["persistent_kv_cache_bytes"],
                }
            )

    return destination
