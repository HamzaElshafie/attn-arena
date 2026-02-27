from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn as nn

from attn_arena.attention.base import KVCache
from attn_arena.models.base import ModelBackbone

WeightsMode = Literal["native", "adapted", "synthetic"]
SyntheticInitPolicy = Literal[
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "uniform",
    "normal",
]


@dataclass(frozen=True)
class SyntheticTokenSpec:
    """Deterministic synthetic token generation settings."""

    seed: int = 0
    offset: int = 0


@dataclass(frozen=True)
class SyntheticInitConfig:
    """Configuration for deterministic synthetic parameter initialization."""

    policy: SyntheticInitPolicy = "xavier_uniform"
    seed: int = 0
    uniform_bound: float = 0.02
    normal_std: float = 0.02


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
    """Execution controls and weight-mode semantics for benchmark runs."""

    warmup_iters: int = 1
    timed_iters: int = 3
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    weights_mode: WeightsMode = "synthetic"
    synthetic_init: SyntheticInitConfig = field(default_factory=SyntheticInitConfig)
    checkpoint_source: str | None = None


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Run metadata required for reproducible benchmark reporting."""

    model_name: str
    attention_name: str
    weights_mode: WeightsMode
    device: str
    dtype: str
    synthetic_init_policy: SyntheticInitPolicy | None
    synthetic_init_seed: int | None
    checkpoint_source: str | None


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

    metadata: BenchmarkMetadata
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


DEFAULT_BENCHMARK_RUN_CONFIG = BenchmarkRunConfig()
DEFAULT_SYNTHETIC_TOKEN_SPEC = SyntheticTokenSpec()


def _fan_in_fan_out(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        raise ValueError("Fan-in/fan-out requires tensor with at least 2 dimensions.")
    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = math.prod(tensor.shape[2:])
    fan_in = int(num_input_fmaps * receptive_field_size)
    fan_out = int(num_output_fmaps * receptive_field_size)
    return fan_in, fan_out


def _generator_key(device: torch.device) -> str:
    index = device.index if device.index is not None else -1
    return f"{device.type}:{index}"


def _generator_for_device(
    *,
    device: torch.device,
    seed: int,
    generators: dict[str, torch.Generator],
) -> torch.Generator:
    key = _generator_key(device)
    if key not in generators:
        generators[key] = torch.Generator(device=str(device))
        generators[key].manual_seed(seed)
    return generators[key]


def _initialize_matrix_parameter(
    parameter: torch.Tensor,
    *,
    config: SyntheticInitConfig,
    generator: torch.Generator,
) -> None:
    fan_in, fan_out = _fan_in_fan_out(parameter)
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan_in/fan_out must be positive for initialization.")

    if config.policy == "xavier_uniform":
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        parameter.uniform_(-bound, bound, generator=generator)
        return
    if config.policy == "xavier_normal":
        std = math.sqrt(2.0 / (fan_in + fan_out))
        parameter.normal_(0.0, std, generator=generator)
        return
    if config.policy == "kaiming_uniform":
        bound = math.sqrt(3.0 / fan_in)
        parameter.uniform_(-bound, bound, generator=generator)
        return
    if config.policy == "kaiming_normal":
        std = math.sqrt(2.0 / fan_in)
        parameter.normal_(0.0, std, generator=generator)
        return
    if config.policy == "uniform":
        parameter.uniform_(-config.uniform_bound, config.uniform_bound, generator=generator)
        return
    if config.policy == "normal":
        parameter.normal_(0.0, config.normal_std, generator=generator)
        return

    raise ValueError(f"Unsupported synthetic init policy: {config.policy}")


def initialize_model_weights_for_synthetic_mode(
    model: nn.Module,
    *,
    config: SyntheticInitConfig,
) -> None:
    """Initialize all model parameters explicitly for synthetic benchmark mode."""

    generators: dict[str, torch.Generator] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2:
                generator = _generator_for_device(
                    device=parameter.device,
                    seed=config.seed,
                    generators=generators,
                )
                _initialize_matrix_parameter(
                    parameter,
                    config=config,
                    generator=generator,
                )
                continue

            if name.endswith("bias"):
                parameter.zero_()
            elif "norm" in name and name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                parameter.zero_()


def make_synthetic_input_ids(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    spec: SyntheticTokenSpec,
    device: torch.device,
) -> torch.LongTensor:
    """Create deterministic synthetic token IDs for benchmarking."""

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
    return cast(torch.LongTensor, tokens.to(device=device))


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
    return cast(torch.LongTensor, positions.unsqueeze(0).expand(batch_size, -1))


def init_kv_caches_for_model(
    model: ModelBackbone,
    *,
    batch_size: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[KVCache]:
    """Initialize one KV cache per layer using injected attention modules."""

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


def _resolve_attention_name(model: ModelBackbone) -> str:
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        return "unknown"
    attention = getattr(layers[0], "attention", None)
    if attention is None:
        return "unset"
    return str(attention.__class__.__name__)


def _build_benchmark_metadata(
    *,
    model: ModelBackbone,
    run_config: BenchmarkRunConfig,
    device: torch.device,
) -> BenchmarkMetadata:
    synthetic_init_policy: SyntheticInitPolicy | None = None
    synthetic_init_seed: int | None = None
    if run_config.weights_mode == "synthetic":
        synthetic_init_policy = run_config.synthetic_init.policy
        synthetic_init_seed = run_config.synthetic_init.seed

    return BenchmarkMetadata(
        model_name=model.__class__.__name__,
        attention_name=_resolve_attention_name(model),
        weights_mode=run_config.weights_mode,
        device=str(device),
        dtype=str(run_config.dtype),
        synthetic_init_policy=synthetic_init_policy,
        synthetic_init_seed=synthetic_init_seed,
        checkpoint_source=run_config.checkpoint_source,
    )


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
    cache_position = cast(
        torch.LongTensor,
        torch.arange(seq_len, dtype=torch.long, device=device),
    )
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
        position = kv_caches[0].current_seq_len()
        position_ids = build_position_ids(
            batch_size=batch_size,
            seq_len=1,
            start_position=position,
            device=device,
        )
        cache_position = cast(
            torch.LongTensor,
            torch.tensor([position], dtype=torch.long, device=device),
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
    run_config: BenchmarkRunConfig | None = None,
    token_spec: SyntheticTokenSpec | None = None,
) -> InferenceBenchmarkResult:
    """Run deterministic prefill+decode benchmarking on a backbone."""

    if run_config is None:
        run_config = DEFAULT_BENCHMARK_RUN_CONFIG
    if token_spec is None:
        token_spec = DEFAULT_SYNTHETIC_TOKEN_SPEC

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
    if run_config.weights_mode == "synthetic" and not isinstance(model, nn.Module):
        raise TypeError(
            "Synthetic mode requires `model` to be a torch.nn.Module so parameters "
            "can be initialized explicitly."
        )

    if isinstance(model, nn.Module):
        model = model.to(device=device, dtype=run_config.dtype)  # type: ignore[assignment]
        if run_config.weights_mode == "synthetic":
            initialize_model_weights_for_synthetic_mode(model, config=run_config.synthetic_init)
        model.eval()

    prefill_input_ids = make_synthetic_input_ids(
        batch_size=workload.batch_size,
        seq_len=workload.prefill_len,
        vocab_size=workload.vocab_size,
        spec=token_spec,
        device=device,
    )

    metadata = _build_benchmark_metadata(model=model, run_config=run_config, device=device)

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

        return prefill_elapsed, decode_elapsed, total_kv_cache_bytes(kv_caches)

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
        metadata=metadata,
        workload=workload,
        prefill=prefill_metrics,
        decode=decode_metrics,
        total_elapsed_seconds=total_elapsed,
        kv_cache_bytes=kv_cache_bytes,
    )
