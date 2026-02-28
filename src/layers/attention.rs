/// Grouped-Query Attention with optional sliding-window masking.
///
/// Cohere2 uses GQA with 16 query heads and 4 key/value heads (ratio 4:1).
/// Each query head group shares one KV head. The KV cache is stored per-layer
/// as a simple growing `Vec` of (K, V) pairs, one entry per past position.
///
/// Two attention modes are supported, selected per-layer by the model config:
/// - **Full causal attention**: standard causal mask over all positions.
/// - **Sliding window attention**: causal mask limited to the most recent
///   `window_size` positions.
///
/// # Assumptions
/// - Single-token (decode) forward pass: `q` is for one new position.
/// - KV cache is a `Vec<(Tensor, Tensor)>` where each entry holds
///   `(k_for_position, v_for_position)` as 1-D tensors of length
///   `num_kv_heads * head_dim`.
/// - No attention bias.
/// - No QK norm.

use super::linear::Linear;
use super::rope::RoPE;
use super::simd::{f32_dot_f32, f32_saxpy};
use super::tensor::Tensor;

use rayon::prelude::*;

/// Attention type for each layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    Full,
    SlidingWindow,
}

/// Per-layer KV cache: a list of (K, V) vectors, one per past position.
/// K has shape (num_kv_heads * head_dim,), likewise V.
pub struct KvCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rope: RoPE,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub attn_type: AttentionType,
    pub sliding_window: usize,
}

impl Attention {
    /// Run a single-token decode step.
    ///
    /// `x`: hidden state, shape (hidden_size,).
    /// `pos`: the absolute position index of this token.
    /// `cache`: the KV cache for this layer (mutated — new KV pair appended).
    ///
    /// Returns the attention output, shape (hidden_size,).
    pub fn forward(&self, x: &Tensor, pos: usize, cache: &mut KvCache) -> Tensor {
        let hidden_size = self.num_heads * self.head_dim;

        // Project to Q, K, V
        let q = self.q_proj.forward(x);   // (num_heads * head_dim,)
        let k = self.k_proj.forward(x);   // (num_kv_heads * head_dim,)
        let v = self.v_proj.forward(x);   // (num_kv_heads * head_dim,)

        // Apply RoPE to Q and K
        let q = self.rope.forward(&q, pos);
        let k = self.rope.forward(&k, pos);

        // Append to cache
        cache.keys.push(k);
        cache.values.push(v);

        // Determine the range of positions to attend to
        let seq_len = cache.len(); // includes the newly added position
        let start = match self.attn_type {
            AttentionType::Full => 0,
            AttentionType::SlidingWindow => {
                seq_len.saturating_sub(self.sliding_window)
            }
        };

        let num_groups = self.num_heads / self.num_kv_heads;

        // Compute attention per query head — parallelized across heads
        let head_outputs: Vec<Vec<f32>> = (0..self.num_heads)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / num_groups; // which KV head this query head uses

                // Extract this head's query: slice [h*head_dim .. (h+1)*head_dim]
                let q_start = h * self.head_dim;
                let q_end = q_start + self.head_dim;
                let q_head = &q.data[q_start..q_end];

                let kv_start = kv_h * self.head_dim;
                let kv_end = kv_start + self.head_dim;

                // Compute attention scores against all cached keys in [start..seq_len]
                let attend_len = seq_len - start;
                let scale = 1.0 / (self.head_dim as f32).sqrt();

                let mut scores = Vec::with_capacity(attend_len);
                for t in start..seq_len {
                    let k_data = &cache.keys[t].data[kv_start..kv_end];
                    let dot = f32_dot_f32(q_head, k_data);
                    scores.push(dot * scale);
                }

                // Softmax over scores
                let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for e in &mut exps {
                    *e /= sum;
                }

                // Weighted sum of values
                let mut out_head = vec![0.0f32; self.head_dim];
                for (idx, t) in (start..seq_len).enumerate() {
                    let v_data = &cache.values[t].data[kv_start..kv_end];
                    let w = exps[idx];
                    f32_saxpy(&mut out_head, w, v_data);
                }

                out_head
            })
            .collect();

        // Concatenate all head outputs → (hidden_size,)
        let mut attn_data = Vec::with_capacity(hidden_size);
        for head in &head_outputs {
            attn_data.extend_from_slice(head);
        }
        let attn_out = Tensor::new(attn_data, vec![hidden_size]);

        // Output projection
        self.o_proj.forward(&attn_out)
    }
}
