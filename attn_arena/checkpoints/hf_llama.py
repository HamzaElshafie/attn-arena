from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from attn_arena.models.llama.config import LlamaConfig


@dataclass(frozen=True)
class CheckpointLoadReport:
    """Structured result of checkpoint loading into a model."""

    source: str
    num_source_tensors: int
    num_remapped_tensors: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def hf_llama_config_from_dict(config_dict: dict[str, Any]) -> LlamaConfig:
    """Convert a Hugging Face Llama `config.json` payload to `LlamaConfig`.

    The local `LlamaConfig` does not currently store `intermediate_size`
    directly. This converter derives a representation that reproduces the HF
    `intermediate_size` exactly for the current backbone implementation.
    """

    required_keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    missing = [key for key in required_keys if key not in config_dict]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"HF Llama config is missing required keys: {missing_str}")

    hidden_size = int(config_dict["hidden_size"])
    intermediate_size = int(config_dict["intermediate_size"])
    num_layers = int(config_dict["num_hidden_layers"])
    num_heads = int(config_dict["num_attention_heads"])
    num_kv_heads = int(config_dict.get("num_key_value_heads", num_heads))
    vocab_size = int(config_dict["vocab_size"])
    norm_eps = float(config_dict.get("rms_norm_eps", 1e-5))
    rope_theta = float(config_dict.get("rope_theta", 10000.0))
    max_seq_len = int(config_dict.get("max_position_embeddings", 2048))

    # Preferred representation: keep multiple_of small and use an explicit
    # multiplier. We validate that the derived internal intermediate size matches
    # the HF config exactly, and fall back to an exact representation if needed.
    base_hidden_dim = int(4 * hidden_size / 3)
    if base_hidden_dim <= 0:
        raise ValueError("Invalid hidden_size; computed FFN base hidden dim must be > 0.")

    candidate = LlamaConfig(
        dim=hidden_size,
        n_layers=num_layers,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        multiple_of=1,
        ffn_dim_multiplier=intermediate_size / base_hidden_dim,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
    )
    if candidate.intermediate_size == intermediate_size:
        return candidate

    # Exact fallback for families where float rounding or a different hidden-dim
    # convention would otherwise produce an off-by-one mismatch.
    fallback = LlamaConfig(
        dim=hidden_size,
        n_layers=num_layers,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        multiple_of=intermediate_size,
        ffn_dim_multiplier=None,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        max_seq_len=max_seq_len,
    )
    if fallback.intermediate_size != intermediate_size:
        raise ValueError(
            "Could not represent HF intermediate_size exactly with current LlamaConfig. "
            f"hf={intermediate_size}, derived={fallback.intermediate_size}"
        )
    return fallback


def hf_llama_config_from_file(config_path: str | Path) -> LlamaConfig:
    """Load an HF Llama `config.json` from disk and convert to `LlamaConfig`."""

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"HF config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("HF config.json must contain a JSON object.")
    return hf_llama_config_from_dict(payload)


def hf_llama_config_from_pretrained_dir(pretrained_dir: str | Path) -> LlamaConfig:
    """Load `LlamaConfig` from a local HF model directory."""

    return hf_llama_config_from_file(Path(pretrained_dir) / "config.json")


def _load_safetensors_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "safetensors is required to load HF safetensors checkpoints. "
            "Install it (e.g. `pip install safetensors`) before using this loader."
        ) from exc

    tensors = load_file(str(path))
    return dict(tensors)


def read_hf_safetensors_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a local HF safetensors checkpoint (single-file or sharded directory).

    Supported inputs:
    - a `.safetensors` file
    - a directory with `model.safetensors`
    - a directory with `model.safetensors.index.json` and shard files
    """

    input_path = Path(path)
    if input_path.is_file():
        if input_path.suffix != ".safetensors":
            raise ValueError(f"Expected a .safetensors file, got: {input_path}")
        return _load_safetensors_file(input_path)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {input_path}")

    single_file = input_path / "model.safetensors"
    if single_file.exists():
        return _load_safetensors_file(single_file)

    index_path = input_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "Could not find `model.safetensors` or `model.safetensors.index.json` "
            f"in directory: {input_path}"
        )

    with index_path.open("r", encoding="utf-8") as file:
        index_payload = json.load(file)
    if not isinstance(index_payload, dict):
        raise ValueError("Invalid safetensors index JSON: expected an object.")

    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Safetensors index JSON is missing object field `weight_map`.")

    shard_names = sorted({str(shard_name) for shard_name in weight_map.values()})
    if not shard_names:
        raise ValueError("Safetensors index JSON `weight_map` is empty.")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = input_path / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Safetensors shard referenced in index not found: {shard_path}"
            )

        shard_tensors = _load_safetensors_file(shard_path)
        duplicate_keys = state_dict.keys() & shard_tensors.keys()
        if duplicate_keys:
            duplicates_str = ", ".join(sorted(duplicate_keys))
            raise ValueError(f"Duplicate tensor keys found across shards: {duplicates_str}")
        state_dict.update(shard_tensors)

    return state_dict


def remap_hf_llama_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HF Llama parameter names to the current attn_arena Llama backbone names.

    This remapper targets the current `LlamaBackbone` + MHA/GQA-style projection
    naming (`wq/wk/wv/wo`) and FFN naming (`w`, `v`, `w2`).
    """

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        mapped = _remap_hf_llama_key(key)
        if mapped is None:
            continue
        remapped[mapped] = value
    return remapped


def _remap_hf_llama_key(key: str) -> str | None:
    root_map = {
        "model.embed_tokens.weight": "embedding.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "unembedding.weight",
    }
    if key in root_map:
        return root_map[key]

    if not key.startswith("model.layers."):
        return None

    parts = key.split(".")
    if len(parts) < 5:
        return None

    layer_idx = parts[2]
    suffix = ".".join(parts[3:])
    layer_prefix = f"layers.{layer_idx}"

    layer_map = {
        "input_layernorm.weight": f"{layer_prefix}.attn_norm.weight",
        "post_attention_layernorm.weight": f"{layer_prefix}.ffn_norm.weight",
        "self_attn.q_proj.weight": f"{layer_prefix}.attention.wq.weight",
        "self_attn.k_proj.weight": f"{layer_prefix}.attention.wk.weight",
        "self_attn.v_proj.weight": f"{layer_prefix}.attention.wv.weight",
        "self_attn.o_proj.weight": f"{layer_prefix}.attention.wo.weight",
        "mlp.gate_proj.weight": f"{layer_prefix}.ffn.w.weight",
        "mlp.up_proj.weight": f"{layer_prefix}.ffn.v.weight",
        "mlp.down_proj.weight": f"{layer_prefix}.ffn.w2.weight",
    }
    return layer_map.get(suffix)


def transform_hf_llama_tensors_if_needed(
    remapped_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply tensor-layout transforms required by the current implementation.

    This is intentionally a separate step from key remapping so future variants
    (e.g., fused projections or adapter-driven shape transforms) can hook in
    cleanly. For the current Llama backbone with unfused projections, no
    transform is needed.
    """

    return remapped_state_dict


def load_hf_llama_safetensors(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> CheckpointLoadReport:
    """Load local HF Llama safetensors weights into an attn_arena model.

    This function currently performs:
    1. safetensors read (single-file or sharded)
    2. HF key remap -> attn_arena naming
    3. no-op tensor transform step (reserved for future layout transforms)
    4. `model.load_state_dict(...)`
    """

    source_state_dict = read_hf_safetensors_state_dict(checkpoint_path)
    remapped_state_dict = remap_hf_llama_state_dict(source_state_dict)
    transformed_state_dict = transform_hf_llama_tensors_if_needed(remapped_state_dict)

    incompatible = model.load_state_dict(transformed_state_dict, strict=strict)
    return CheckpointLoadReport(
        source=str(checkpoint_path),
        num_source_tensors=len(source_state_dict),
        num_remapped_tensors=len(transformed_state_dict),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
