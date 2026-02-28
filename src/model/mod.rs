/// Cohere2 model definition and weight loading.
///
/// This module implements the `Cohere2ForCausalLM` architecture as used by
/// the tiny-aya-global model. It handles:
///
/// - Parsing the HuggingFace `config.json`.
/// - Loading weights from sharded `.safetensors` files via the index JSON.
/// - Assembling all layers into a runnable model.
///
/// # Architecture highlights (Cohere2)
///
/// - **Parallel block**: attention and MLP run on the *same* layer-normed input
///   and their outputs are summed: `x = x + attn(ln(x)) + mlp(ln(x))`.
///   There is only one `input_layernorm` per layer (no post-attention norm).
///
/// - **Grouped-Query Attention**: 16 query heads, 4 key/value heads.
///
/// - **Hybrid sliding / full attention**: determined per-layer by the
///   `layer_types` array in config. Every 4th layer is full attention; the
///   rest use a sliding window of 4096.
///
/// - **Tied embeddings**: the LM head reuses `model.embed_tokens.weight`.
///
/// - **GPT-J style RoPE**: consecutive-pair interleaving, θ = 50 000.

pub mod config;
pub mod loading;
pub mod forward;
