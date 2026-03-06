# vibe-cpu-llm

A low-latency, CPU-only LLM inference engine written in Rust, targeting small-batch autoregressive decoding. Designed to run quantized transformer models efficiently without a GPU, using hand-rolled SIMD kernels and Rayon-based parallelism.

Currently supports the **Cohere2** architecture (e.g. `tiny-aya-global`).

---

## Features

- **BF16 weight storage** — weights stay in bf16 in memory; converted to f32 on the fly during compute, halving memory bandwidth requirements
- **SIMD-accelerated kernels** — hand-written bf16→f32 + fused dot product kernels for:
  - `aarch64` NEON: `vshll`/`vmovl` + `vfmaq_f32`, 32 elements/iter, 4 independent accumulators
  - `x86_64` AVX2+FMA: `vpslld`/`vpmovzxwd` + `vfmadd231ps`, 32 elements/iter, 4×8 accumulators
  - Scalar fallback for all other architectures
- **SIMD attention** — f32·f32 QK dot product and weighted value accumulation also use NEON/AVX2+FMA kernels
- **Rayon parallelism** — multi-threaded across rows (matvec), across heads (attention), across projections (gate+up in MLP), and across blocks (attention+MLP run concurrently)
- **Sliding Window Attention** — per-layer hybrid: full causal attention every 4th layer, sliding window (4096 tokens) for all other layers, controlled by `layer_types` in `config.json`
- **Grouped Query Attention (GQA)** — supports asymmetric Q/KV head counts (e.g. 16Q / 4KV)
- **Parallel blocks** — attention and MLP can be fused in parallel when `use_parallel_block` is set in the config
- **Greedy decoding** — argmax token selection with stop on `<|END_OF_TURN_TOKEN|>` or EOS
- **HuggingFace-compatible** — loads `tokenizer.json` via the `tokenizers` crate and reads sharded safetensors via `memmap2` (zero-copy weight mapping)

---

## Performance

Measured on `tiny-aya-global` (Cohere2, 36 layers), single sequence, ARM64 Windows:

| Configuration | Decode speed |
|---|---|
| Scalar baseline | ~3 tok/s |
| SIMD (NEON) kernels | ~11.3 tok/s (3.7× speedup) |
| SIMD + attention kernels | ~13.6 tok/s (at 128+ token context) |

---

## Project Structure

```
src/
  main.rs              # CLI entry point (generate binary)
  lib.rs               # Crate root
  layers/
    attention.rs       # Multi-head / grouped-query attention + KV cache
    embedding.rs       # Token embedding lookup
    layernorm.rs       # RMS layer norm
    linear.rs          # BF16 linear projection (matvec)
    mlp.rs             # SwiGLU MLP (gate_proj, up_proj, down_proj)
    rope.rs            # Rotary positional embeddings (RoPE)
    simd.rs            # Platform SIMD kernels (NEON / AVX2+FMA / scalar)
    tensor.rs          # Tensor utilities and bf16 slice helpers
  model/
    config.rs          # Cohere2Config — deserialized from config.json
    forward.rs         # Cohere2Model — full forward pass and KV cache
    loading.rs         # Weight loading from sharded safetensors
models/
  tiny-aya-global/     # Example model weights (not included in repo)
```

---

## Requirements

- Rust (edition 2024, stable)
- A Cohere2-compatible model directory containing:
  - `config.json`
  - `tokenizer.json`
  - `model.safetensors.index.json`
  - Shard files (`model-00001-of-NNNNN.safetensors`, …)

---

## Building

```powershell
cargo build --release
```

For best performance, enable native CPU features:

```powershell
$env:RUSTFLAGS="-C target-cpu=native"
cargo build --release
```

---

## Running

```powershell
cargo run --release --bin generate -- `
  --model-dir models/tiny-aya-global `
  --prompt "Explain the Rust borrow checker in simple terms." `
  --max-tokens 256
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--model-dir` | `models/tiny-aya-global` | Path to model directory |
| `--prompt` | `"Hello, how are you?"` | User prompt text |
| `--max-tokens` | `128` | Maximum tokens to generate |

The prompt is automatically wrapped in Cohere2's chat template:
```
<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
```

---

## Dependencies

| Crate | Purpose |
|---|---|
| `half` | BF16 scalar conversions |
| `memmap2` | Zero-copy memory-mapped safetensors |
| `rayon` | Data-parallel iteration |
| `safetensors` | Safetensors format parsing |
| `serde` / `serde_json` | Config and index JSON deserialization |
| `tokenizers` | HuggingFace tokenizer (loads `tokenizer.json`) |

---

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for planned features including Flash Attention v2, paged KV cache, batched prefill, and fp8 weight support.
