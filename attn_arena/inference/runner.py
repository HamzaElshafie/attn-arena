from __future__ import annotations

import time
from dataclasses import dataclass
import torch

from attn_arena.attention.base import KVCache
from attn_arena.models.base import ModelBackbone

@dataclass(frozen=True)
class SyntheticTokenSpec:
    """Deterministic synthetic token generation settings."""

    seed: int = 0
    offset: int = 0


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Benchmark workload definition for prefill+decode inference."""

    batch_size: int
    prefill_len: int
    decode_len: int
    max_seq_len: int
    vocab_size: int


@dataclass(frozen=True)
class BenchmarkRunConfig:
    """Execution controls for stable timing measurements."""

    warmup_iters: int = 1
    timed_iters: int = 3
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


@dataclass(frozen=True)
class StageMetrics:
    """Timing and throughput for a single stage."""

    elapsed_seconds: float
    total_tokens: int

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_tokens / self.elapsed_seconds


@dataclass(frozen=True)
class InferenceBenchmarkResult:
    """Aggregated metrics for a prefill+decode benchmark run."""

    workload: BenchmarkWorkload
    prefill: StageMetrics
    decode: StageMetrics
    total_elapsed_seconds: float
    kv_cache_bytes: int

    @property
    def total_tokens(self) -> int:
        return self.prefill.total_tokens + self.decode.total_tokens

    @property
    def total_tokens_per_second(self) -> float:
        if self.total_elapsed_seconds <= 0:
            return 0.0
        return self.total_tokens / self.total_elapsed_seconds


def make_synthetic_input_ids(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    spec: SyntheticTokenSpec,
    device: torch.device,
) -> torch.LongTensor:
    """Create deterministic synthetic token IDs for benchmarking.

    Generation happens on CPU using a fixed generator for reproducibility, then
    moves to the target device. This keeps behavior stable across CPU/GPU runs.
    """

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be > 0.")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        generator=generator,
    )
    if spec.offset:
        tokens = (tokens + spec.offset) % vocab_size
    return tokens.to(device=device)


def build_position_ids(
    *,
    batch_size: int,
    seq_len: int,
    start_position: int,
    device: torch.device,
) -> torch.LongTensor:
    """Create batched position IDs for prefill or decode."""

    if seq_len <= 0:
        raise ValueError("seq_len must be > 0.")
    positions = torch.arange(
        start_position,
        start_position + seq_len,
        dtype=torch.long,
        device=device,
    )
    return positions.unsqueeze(0).expand(batch_size, -1)


def init_kv_caches_for_model(
    model: ModelBackbone,
    *,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[KVCache]:
    """Initialize one KV cache per layer using the injected attention modules.

    Current implementation assumes a Llama-like backbone with a `layers`
    collection where each layer owns an `attention` module exposing
    `init_kv_cache(...)`. This is intentionally isolated here so future model
    backbones can add a native cache-init API without changing benchmark code.
    """

    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            "Model does not expose `layers`; cannot initialize KV caches. "
            "Add a model-specific cache initializer or extend the backbone contract."
        )

    caches: list[KVCache] = []
    for layer in layers:
        attention = getattr(layer, "attention", None)
        if attention is None:
            raise ValueError("All layers must have attention set before initializing KV caches.")
        init_kv_cache = getattr(attention, "init_kv_cache", None)
        if init_kv_cache is None:
            raise TypeError("Layer attention does not implement init_kv_cache(...).")
        cache = init_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        caches.append(cache)
    return caches


def total_kv_cache_bytes(kv_caches: list[KVCache]) -> int:
    """Return the total payload size across all layer caches."""

    return sum(cache.size_bytes() for cache in kv_caches)


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_prefill_once(
    *,
    model: ModelBackbone,
    input_ids: torch.LongTensor,
    kv_caches: list[KVCache],
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    position_ids = build_position_ids(
        batch_size=batch_size,
        seq_len=seq_len,
        start_position=0,
        device=device,
    )
    cache_position = torch.arange(seq_len, dtype=torch.long, device=device)
    return model.forward(
        input_ids=input_ids,
        position_ids=position_ids,
        kv_caches=kv_caches,
        attention_mask=None,
        cache_position=cache_position,
    )


def _run_decode_once(
    *,
    model: ModelBackbone,
    batch_size: int,
    decode_len: int,
    vocab_size: int,
    kv_caches: list[KVCache],
    token_spec: SyntheticTokenSpec,
    device: torch.device,
) -> torch.Tensor:
    if decode_len <= 0:
        raise ValueError("decode_len must be > 0.")

    last_logits: torch.Tensor | None = None
    for step in range(decode_len):
        input_ids = make_synthetic_input_ids(
            batch_size=batch_size,
            seq_len=1,
            vocab_size=vocab_size,
            spec=SyntheticTokenSpec(seed=token_spec.seed + step, offset=token_spec.offset),
            device=device,
        )
        position_ids = build_position_ids(
            batch_size=batch_size,
            seq_len=1,
            start_position=kv_caches[0].current_seq_len(),
            device=device,
        )
        cache_position = torch.tensor(
            [kv_caches[0].current_seq_len()],
            dtype=torch.long,
            device=device,
        )
        last_logits = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_caches=kv_caches,
            attention_mask=None,
            cache_position=cache_position,
        )

    assert last_logits is not None
    return last_logits


def run_prefill_decode_benchmark(
    *,
    model: ModelBackbone,
    workload: BenchmarkWorkload,
    run_config: BenchmarkRunConfig = BenchmarkRunConfig(),
    token_spec: SyntheticTokenSpec = SyntheticTokenSpec(),
) -> InferenceBenchmarkResult:
    """Run deterministic prefill+decode benchmarking on a backbone.

    This function is benchmark-first and intentionally does not depend on a
    tokenizer. It uses synthetic tokens to exercise the inference path and KV
    cache behavior under controlled workloads.
    """

    if workload.prefill_len <= 0:
        raise ValueError("workload.prefill_len must be > 0.")
    if workload.decode_len < 0:
        raise ValueError("workload.decode_len must be >= 0.")
    if workload.max_seq_len < workload.prefill_len + workload.decode_len:
        raise ValueError("workload.max_seq_len must cover prefill_len + decode_len.")
    if run_config.timed_iters <= 0:
        raise ValueError("run_config.timed_iters must be > 0.")
    if run_config.warmup_iters < 0:
        raise ValueError("run_config.warmup_iters must be >= 0.")

    device = torch.device(run_config.device)
    if isinstance(model, torch.nn.Module):
        model = model.to(device=device, dtype=run_config.dtype)  # type: ignore[assignment]
        model.eval()

    prefill_input_ids = make_synthetic_input_ids(
        batch_size=workload.batch_size,
        seq_len=workload.prefill_len,
        vocab_size=workload.vocab_size,
        spec=token_spec,
        device=device,
    )

    def _single_iteration() -> tuple[float, float, int]:
        kv_caches = init_kv_caches_for_model(
            model,
            batch_size=workload.batch_size,
            max_seq_len=workload.max_seq_len,
            device=device,
            dtype=run_config.dtype,
        )

        _synchronize_if_needed(device)
        prefill_start = time.perf_counter()
        with torch.no_grad():
            _ = _run_prefill_once(model=model, input_ids=prefill_input_ids, kv_caches=kv_caches)
        _synchronize_if_needed(device)
        prefill_elapsed = time.perf_counter() - prefill_start

        decode_elapsed = 0.0
        if workload.decode_len > 0:
            _synchronize_if_needed(device)
            decode_start = time.perf_counter()
            with torch.no_grad():
                _ = _run_decode_once(
                    model=model,
                    batch_size=workload.batch_size,
                    decode_len=workload.decode_len,
                    vocab_size=workload.vocab_size,
                    kv_caches=kv_caches,
                    token_spec=token_spec,
                    device=device,
                )
            _synchronize_if_needed(device)
            decode_elapsed = time.perf_counter() - decode_start

        cache_bytes = total_kv_cache_bytes(kv_caches)
        return prefill_elapsed, decode_elapsed, cache_bytes

    for _ in range(run_config.warmup_iters):
        _single_iteration()

    prefill_elapsed_sum = 0.0
    decode_elapsed_sum = 0.0
    kv_cache_bytes = 0
    for _ in range(run_config.timed_iters):
        prefill_elapsed, decode_elapsed, cache_bytes = _single_iteration()
        prefill_elapsed_sum += prefill_elapsed
        decode_elapsed_sum += decode_elapsed
        kv_cache_bytes = cache_bytes

    timed_iters = run_config.timed_iters
    prefill_metrics = StageMetrics(
        elapsed_seconds=prefill_elapsed_sum / timed_iters,
        total_tokens=workload.batch_size * workload.prefill_len,
    )
    decode_metrics = StageMetrics(
        elapsed_seconds=decode_elapsed_sum / timed_iters,
        total_tokens=workload.batch_size * workload.decode_len,
    )
    total_elapsed = prefill_metrics.elapsed_seconds + decode_metrics.elapsed_seconds

    return InferenceBenchmarkResult(
        workload=workload,
        prefill=prefill_metrics,
        decode=decode_metrics,
        total_elapsed_seconds=total_elapsed,
        kv_cache_bytes=kv_cache_bytes,
    )
