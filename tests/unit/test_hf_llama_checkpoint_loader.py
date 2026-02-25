from __future__ import annotations

import torch

from attn_arena.checkpoints.hf_llama import hf_llama_config_from_dict, remap_hf_llama_state_dict


def test_hf_llama_config_from_dict_maps_required_fields() -> None:
    config = hf_llama_config_from_dict(
        {
            "hidden_size": 32,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 128,
        }
    )

    assert config.dim == 32
    assert config.n_layers == 2
    assert config.n_heads == 4
    assert config.effective_num_kv_heads == 4
    assert config.vocab_size == 128
    assert config.max_seq_len == 128
    assert config.intermediate_size == 96


def test_remap_hf_llama_state_dict_maps_known_keys_and_ignores_unknown() -> None:
    tensor = torch.randn(2, 2)
    remapped = remap_hf_llama_state_dict(
        {
            "model.embed_tokens.weight": tensor,
            "model.layers.0.input_layernorm.weight": tensor,
            "model.layers.0.self_attn.q_proj.weight": tensor,
            "model.layers.0.self_attn.k_proj.weight": tensor,
            "model.layers.0.self_attn.v_proj.weight": tensor,
            "model.layers.0.self_attn.o_proj.weight": tensor,
            "model.layers.0.mlp.gate_proj.weight": tensor,
            "model.layers.0.mlp.up_proj.weight": tensor,
            "model.layers.0.mlp.down_proj.weight": tensor,
            "model.layers.0.post_attention_layernorm.weight": tensor,
            "model.norm.weight": tensor,
            "lm_head.weight": tensor,
            "model.rotary_emb.inv_freq": tensor,
        }
    )

    assert set(remapped.keys()) == {
        "embedding.weight",
        "layers.0.attn_norm.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.ffn.w.weight",
        "layers.0.ffn.v.weight",
        "layers.0.ffn.w2.weight",
        "layers.0.ffn_norm.weight",
        "norm.weight",
        "unembedding.weight",
    }
