/// vibe-cpu-llm: A low-latency CPU-based LLM inference engine.
///
/// This crate provides a minimal but functional implementation of transformer
/// model inference, targeting the Cohere2 architecture (used by Aya models).
///
/// # Crate structure
///
/// - [`layers`]: Fundamental building blocks (tensor, linear, attention, etc.)
/// - [`model`]: Cohere2 model definition, config parsing, and weight loading.

pub mod layers;
pub mod model;
