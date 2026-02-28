/// Model configuration, deserialized from the HuggingFace `config.json`.
///
/// Only the fields actually needed for inference are included; the rest are
/// silently ignored by serde.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Cohere2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f64,
    pub rope_theta: f64,
    pub rotary_pct: f64,
    pub sliding_window: usize,
    pub logit_scale: f64,
    pub hidden_act: String,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub use_embedding_sharing: bool,
    #[serde(default)]
    pub use_parallel_block: bool,
    #[serde(default)]
    pub use_gated_activation: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub pad_token_id: u32,
}

impl Cohere2Config {
    /// Load config from a JSON file path.
    pub fn from_file(path: &str) -> Self {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read config from {}: {}", path, e));
        serde_json::from_str(&data)
            .unwrap_or_else(|e| panic!("Failed to parse config JSON: {}", e))
    }
}
