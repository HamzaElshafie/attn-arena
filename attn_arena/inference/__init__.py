"""Inference orchestration package.

This package will host reusable prefill/decode runners and benchmark-oriented
execution utilities. It is kept separate from model backbones and attention
variants so orchestration concerns (cache lifecycle, decoding loops, request
state) do not leak into math modules.
"""

__all__: list[str] = []
