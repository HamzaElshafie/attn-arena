from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from attn_arena.models.base import ModelBackbone

TModel = TypeVar("TModel", bound=ModelBackbone)

_MODEL_REGISTRY: dict[str, Callable[..., ModelBackbone]] = {}


def register_model(name: str) -> Callable[[type[TModel]], type[TModel]]:
    """Register a model backbone under a stable string key.

    Usage:
        @register_model("llama")
        class LlamaBackbone:
            ...
    """

    if not name or not name.strip():
        raise ValueError("model name must be a non-empty string")

    key = name.strip()

    def decorator(model_cls: type[TModel]) -> type[TModel]:
        if key in _MODEL_REGISTRY:
            existing = _MODEL_REGISTRY[key]
            raise KeyError(
                f"model '{key}' is already registered as "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _MODEL_REGISTRY[key] = model_cls
        return model_cls

    return decorator


def get_model(name: str, /, **kwargs: Any) -> ModelBackbone:
    """Instantiate a registered model backbone by name."""

    try:
        model_cls = _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"unknown model '{name}'. available: [{available}]") from exc

    return model_cls(**kwargs)


def list_models() -> tuple[str, ...]:
    """Return registered model names (sorted)."""

    return tuple(sorted(_MODEL_REGISTRY.keys()))


def has_model(name: str) -> bool:
    """Return True if a model backbone is registered under `name`."""

    return name in _MODEL_REGISTRY
