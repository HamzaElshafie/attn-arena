from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

from attn_arena.attention.base import AttentionModule

TAttention = TypeVar("TAttention", bound=AttentionModule)

_ATTENTION_REGISTRY: dict[str, type[Any]] = {}


def register_attention(name: str) -> Callable[[type[TAttention]], type[TAttention]]:
    """Register an attention variant under a stable string key.

    Usage:
        @register_attention("gqa")
        class GroupedQueryAttention:
            ...
    """

    if not name or not name.strip():
        raise ValueError("attention name must be a non-empty string")

    key = name.strip()

    def decorator(attention_cls: type[TAttention]) -> type[TAttention]:
        if key in _ATTENTION_REGISTRY:
            existing = _ATTENTION_REGISTRY[key]
            raise KeyError(
                f"attention '{key}' is already registered as "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _ATTENTION_REGISTRY[key] = attention_cls
        return attention_cls

    return decorator


def get_attention(name: str, /, **kwargs: Any) -> AttentionModule:
    """Instantiate a registered attention variant by name."""

    try:
        attention_cls = _ATTENTION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_ATTENTION_REGISTRY.keys()))
        raise KeyError(f"unknown attention '{name}'. available: [{available}]") from exc

    return cast(AttentionModule, attention_cls(**kwargs))


def list_attentions() -> tuple[str, ...]:
    """Return registered attention names (sorted)."""

    return tuple(sorted(_ATTENTION_REGISTRY.keys()))


def has_attention(name: str) -> bool:
    """Return True if an attention variant is registered under `name`."""

    return name in _ATTENTION_REGISTRY
