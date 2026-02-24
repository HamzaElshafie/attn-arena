from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 201088  # TODO: derive from tokenizer/checkpoint (current is dummy)

    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None

    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_seq_len: int = 2048

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.dim <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("dim, n_layers, n_heads must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be > 0")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")

        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")

        kv_heads = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        if kv_heads <= 0:
            raise ValueError("n_kv_heads must be > 0 when provided")
        if self.n_heads % kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def effective_num_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def intermediate_size(self) -> int:
        hidden_dim = int(4 * self.dim / 3)
        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * self.ffn_dim_multiplier)
        return self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
