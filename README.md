# attn-arena (In progress)

`attn-arena` is an inference-first benchmarking framework for attention mechanisms in decoder-only LLMs.

The project provides one controlled codebase for implementing and evaluating modern attention variants under consistent runtime conditions, with an emphasis on practical serving signals: KV-cache footprint, latency, throughput, and scaling behavior on consumer/prosumer hardware.

## Project Design

### Inference-first scope
`attn-arena` focuses on inference orchestration and systems characterization. The objective is to measure execution behavior across attention designs in a reproducible way, not to optimize for downstream quality metrics inside this repository.

### Attention scope
The benchmark scope covers the following attention families:
- **Multi-Head Attention (MHA)**
- **Multi-Query Attention (MQA)**
- **Grouped-Query Attention (GQA)**
- **Multi-head Latent Attention (MLA)**
- **Grouped Latent Attention (GLA)**
- **Grouped-Tied Attention (GTA)**

### Explicit separation of concerns
The codebase is structured so that each layer has a clear responsibility:
- **Attention variants** own attention math and cache representation.
- **Model backbones** own architecture-specific structure and wiring.
- **Inference orchestration** owns prefill/decode execution and benchmark loops.
- **Checkpoint utilities** own external format loading and remapping.
- **Reporting** owns schema-stable benchmark outputs.

This separation keeps variants and backbones pluggable while reducing coupling between model logic and benchmarking logic.

### Reproducibility by construction
Benchmark execution is designed to be deterministic and explicit:
- synthetic token generation is seed-controlled
- synthetic weight initialization is explicit and policy-based
- run metadata captures mode, device, dtype, and initialization context
- benchmark reports are exported in schema-versioned formats

### Weight modes for fair comparisons
`attn-arena` uses three explicit run modes:
- **native**: checkpoint and architecture/attention are directly compatible
- **adapted**: checkpoint is transformed via a declared adaptation policy
- **synthetic**: model parameters are deterministically initialized for systems benchmarking without checkpoint dependency

Treating these modes explicitly avoids ambiguous comparisons and keeps experiment semantics clear.

## Current Foundation

The current implementation establishes:
- a pluggable Llama backbone path
- a deterministic prefill/decode benchmark runner
- synthetic initialization policy support
- benchmark metadata capture
- JSON/CSV benchmark report export
- HF Llama safetensors loading scaffold (local checkpoint path)

As the project evolves, new attention variants and model backbones are added through the same interfaces so experiments remain comparable across implementations.

## In-scope experiments (v1)

The current benchmark scope focuses on three inference-system experiments:
- **Persistent KV cache bytes vs context length** (single GPU and multi-GPU tensor-parallel runs)
- **Decode throughput vs batch size**
- **Prefill throughput vs prompt length**


## Resources
- [Hardware-Efficient Attention for Fast Decoding](https://arxiv.org/abs/2505.21487)
- [DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [Chapter 12: Initialization Techniques for Deep Networks](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks)
