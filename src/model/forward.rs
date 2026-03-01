/// Forward pass: assembling the full Cohere2 model from layers.
///
/// This file defines the high-level model structs and the autoregressive
/// forward function that runs one token at a time through all 36 layers.

use crate::layers::attention::{Attention, AttentionType, KvCache};
use crate::layers::embedding::Embedding;
use crate::layers::layernorm::LayerNorm;
use crate::layers::linear::Linear;
use crate::layers::mlp::GatedMlp;
use crate::layers::rope::RoPE;

use super::config::Cohere2Config;
use super::loading::ShardedWeights;

/// One transformer decoder layer (Cohere2 parallel block variant).
pub struct Cohere2Layer {
    pub input_layernorm: LayerNorm,
    pub self_attn: Attention,
    pub mlp: GatedMlp,
}

/// The full Cohere2ForCausalLM model.
pub struct Cohere2Model {
    pub config: Cohere2Config,
    pub embed_tokens: Embedding,
    pub layers: Vec<Cohere2Layer>,
    pub final_norm: LayerNorm,
    /// Per-layer KV caches.
    pub kv_caches: Vec<KvCache>,
}

impl Cohere2Model {
    /// Load the model from a directory containing config.json and safetensors.
    pub fn load(model_dir: &str) -> Self {
        eprintln!("Loading config...");
        let config_path = format!("{}/config.json", model_dir);
        let config = Cohere2Config::from_file(&config_path);

        eprintln!("Loading weights...");
        let weights = ShardedWeights::load(model_dir);

        eprintln!("Building embedding table...");
        let embed_weight = weights.tensor("model.embed_tokens.weight");
        let embed_tokens = Embedding::new(embed_weight);
        // Note: embed_weight is now Bf16Tensor — ~1 GB instead of ~2 GB

        eprintln!("Building {} layers...", config.num_hidden_layers);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut kv_caches = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            eprint!("  Layer {}...", i);

            let attn_type = match config.layer_types[i].as_str() {
                "full_attention" => AttentionType::Full,
                "sliding_attention" => AttentionType::SlidingWindow,
                other => panic!("Unknown layer type: {}", other),
            };

            let rope = RoPE::new(
                config.head_dim,
                config.rope_theta as f32,
                config.rotary_pct as f32,
            );

            let layer = Cohere2Layer {
                input_layernorm: LayerNorm::from_bf16(
                    weights.tensor_bf16_raw(&format!("{}.input_layernorm.weight", prefix)),
                    config.layer_norm_eps as f32,
                ),
                self_attn: Attention {
                    q_proj: Linear::new(weights.tensor(&format!(
                        "{}.self_attn.q_proj.weight",
                        prefix
                    ))),
                    k_proj: Linear::new(weights.tensor(&format!(
                        "{}.self_attn.k_proj.weight",
                        prefix
                    ))),
                    v_proj: Linear::new(weights.tensor(&format!(
                        "{}.self_attn.v_proj.weight",
                        prefix
                    ))),
                    o_proj: Linear::new(weights.tensor(&format!(
                        "{}.self_attn.o_proj.weight",
                        prefix
                    ))),
                    rope,
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    attn_type,
                    sliding_window: config.sliding_window,
                },
                mlp: GatedMlp {
                    gate_proj: Linear::new(weights.tensor(&format!(
                        "{}.mlp.gate_proj.weight",
                        prefix
                    ))),
                    up_proj: Linear::new(weights.tensor(&format!(
                        "{}.mlp.up_proj.weight",
                        prefix
                    ))),
                    down_proj: Linear::new(weights.tensor(&format!(
                        "{}.mlp.down_proj.weight",
                        prefix
                    ))),
                },
            };

            layers.push(layer);
            kv_caches.push(KvCache::new());
            eprintln!(" ok");
        }

        eprintln!("Building final norm...");
        let final_norm = LayerNorm::from_bf16(
            weights.tensor_bf16_raw("model.norm.weight"),
            config.layer_norm_eps as f32,
        );

        eprintln!("Model loaded.");
        Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            kv_caches,
        }
    }

    /// Run a single-token forward pass, returning logits over the vocabulary.
    ///
    /// `token_id`: the input token.
    /// `pos`: the absolute position index (0-based).
    ///
    /// Mutates the internal KV caches.
    pub fn forward(&mut self, token_id: u32, pos: usize) -> Vec<f32> {
        // 1. Embedding lookup
        let mut hidden = self.embed_tokens.forward(token_id);

        // 2. Run through all decoder layers
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];

            // Compute normed input for this layer (clones hidden for the norm)
            let normed = layer.input_layernorm.forward(&hidden);

            // Parallel block: attn and mlp both operate on the same normed input.
            // We split the borrows so rayon::join can run both concurrently.
            let cache = &mut self.kv_caches[i];
            let (attn_out, mlp_out) = rayon::join(
                || layer.self_attn.forward(&normed, pos, cache),
                || layer.mlp.forward(&normed),
            );

            // Fused residual accumulation: hidden += attn_out + mlp_out (single pass).
            // No separate allocation — avoids the two .add().add() chain.
            for j in 0..hidden.data.len() {
                hidden.data[j] += attn_out.data[j] + mlp_out.data[j];
            }
        }

        // 3. Final layer norm (in-place)
        self.final_norm.forward_inplace(&mut hidden);

        // 4. LM head (tied embedding weights): logits = embed_weight @ hidden
        let logits = self.embed_tokens.logits(&hidden);

        // 5. Apply logit_scale
        let scale = self.config.logit_scale as f32;
        if (scale - 1.0).abs() > 1e-6 {
            logits.data.iter().map(|&x| x * scale).collect()
        } else {
            logits.data
        }
    }

    /// Reset all KV caches (start a new sequence).
    pub fn reset_caches(&mut self) {
        for cache in &mut self.kv_caches {
            cache.keys.clear();
            cache.values.clear();
        }
    }
}
