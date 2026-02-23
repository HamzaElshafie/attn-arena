import torch
import torch.nn as nn
import torch.nn.functional as F

from attn_arena.attention.base import AttentionModule, KVCache
from attn_arena.models.llama.config import LlamaConfig

class RMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.eps = config.norm_eps
        self.weight = nn.Parameter(torch.ones(config.dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm * self.weight).type_as(x)

class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.intermediate_size = config.intermediate_size

        self.w = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.v = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w(x)) * self.v(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.attention: AttentionModule | None = None
        self.ffn = FeedForward(config)
        self.attn_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)

    def set_attention(self, attention: AttentionModule) -> None:
        self.attention = attention

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if self.attention is None:
            raise ValueError("Attention must be set on TransformerBlock before forward.")

        residual = x
        x_norm = self.attn_norm(x)

        if kv_cache is not None and x.shape[1] == 1 and cache_position is not None:
            attn_output = self.attention.decode(
                hidden_states=x_norm,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache,
                cache_position=cache_position,
            ).output
        else:
            attn_output = self.attention.prefill(
                hidden_states=x_norm,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                cache_position=cache_position,
            ).output

        x = residual + attn_output

        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.ffn(x_norm)
        return x


class LlamaBackbone(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.attention: AttentionModule | None = None
        self._rope_cache_len = 0
        self._rope_cos_cache: torch.Tensor | None = None
        self._rope_sin_cache: torch.Tensor | None = None
        self._rope_cache_device: torch.device | None = None
        
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        self.norm = RMSNorm(config)
        self.unembedding = nn.Linear(self.dim, self.vocab_size, bias=False)

        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config))

    def set_attention(self, attention: AttentionModule) -> None:
        if self.attention is not None:
            raise ValueError("Attention already set on LlamaBackbone.")
            
        self.attention = attention
        for layer in self.layers:
            layer.set_attention(attention)

    def get_rope_cos_sin(
        self,
        position_ids: torch.LongTensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return RoPE coefficients as `(cos, sin)` for the provided positions."""
        max_pos = int(position_ids.max().item()) + 1
        required_len = max(seq_len, max_pos)
        device = position_ids.device

        if (
            self._rope_cos_cache is None
            or self._rope_sin_cache is None
            or self._rope_cache_device != device
            or self._rope_cache_len < required_len
        ):
            dim = self.config.head_dim
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
            )
            t = torch.arange(required_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._rope_cos_cache = emb.cos()
            self._rope_sin_cache = emb.sin()
            self._rope_cache_len = required_len
            self._rope_cache_device = device

        cos = self._rope_cos_cache[position_ids]
        sin = self._rope_sin_cache[position_ids]
        return cos.unsqueeze(2), sin.unsqueeze(2)
        
    def shard(self, tp_rank: int, tp_world_size: int) -> "LlamaBackbone":
        """Return a new model instance configured for tensor-parallel rank."""
        _ = tp_rank
        if tp_world_size == 1:
            return self
        raise NotImplementedError("Tensor parallel sharding is not implemented yet.")

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if self.attention is None:
            raise ValueError("Attention must be set before calling forward.")

        _, seq_len = input_ids.shape
        if kv_caches is not None and len(kv_caches) != len(self.layers):
            raise ValueError("kv_caches length must match number of layers.")

        caches = kv_caches or [None] * len(self.layers)
        position_embeddings = self.get_rope_cos_sin(position_ids, seq_len)
        x = self.embedding(input_ids)

        for layer, cache in zip(self.layers, caches):
            x = layer(
                x=x,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                kv_cache=cache,
                cache_position=cache_position,
            )
            
        x = self.norm(x)
        out = self.unembedding(x)
        return out
