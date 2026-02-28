/// Layer implementations for the inference engine.
///
/// This module contains all the fundamental building blocks needed to
/// assemble transformer models:
///
/// - [`tensor`]: A minimal f32 tensor type with basic linear algebra ops.
/// - [`linear`]: Dense linear projection (matrix-vector multiply).
/// - [`layernorm`]: Layer normalization.
/// - [`rope`]: Rotary Position Embeddings (GPT-J interleaved style).
/// - [`attention`]: Grouped-Query Attention with optional sliding window.
/// - [`mlp`]: Gated MLP (SwiGLU) feed-forward block.
/// - [`embedding`]: Token embedding lookup table.

pub mod tensor;
pub mod linear;
pub mod layernorm;
pub mod rope;
pub mod attention;
pub mod mlp;
pub mod embedding;
pub mod simd;
