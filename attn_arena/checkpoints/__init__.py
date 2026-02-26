"""Checkpoint loading and weight adaptation utilities for inference benchmarking.

This package intentionally isolates checkpoint format handling (HF safetensors,
key remapping, and later weight adaptation policies) from backbone/attention
implementations. The rest of the codebase should remain agnostic to external
checkpoint conventions.
"""

from attn_arena.checkpoints.hf_llama import (
    CheckpointLoadReport,
    hf_llama_config_from_dict,
    hf_llama_config_from_file,
    hf_llama_config_from_pretrained_dir,
    load_hf_llama_safetensors,
    read_hf_safetensors_state_dict,
    remap_hf_llama_state_dict,
)

__all__ = [
    "CheckpointLoadReport",
    "hf_llama_config_from_dict",
    "hf_llama_config_from_file",
    "hf_llama_config_from_pretrained_dir",
    "load_hf_llama_safetensors",
    "read_hf_safetensors_state_dict",
    "remap_hf_llama_state_dict",
]
