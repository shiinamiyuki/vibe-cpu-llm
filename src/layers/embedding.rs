/// Token embedding lookup table.
///
/// Stores the full embedding matrix of shape (vocab_size, hidden_size)
/// in bf16 to save memory (~1 GB instead of ~2 GB for 262K × 2048).
///
/// In Cohere2 the embedding table is shared with the LM head (tied weights),
/// so this same weight matrix is also used for the final logit projection.
///
/// # Assumptions
/// - Weights are stored in bf16 and converted to f32 on-the-fly during
///   embedding lookup and logit projection.

use super::tensor::{Bf16Tensor, Tensor};

pub struct Embedding {
    /// Shape: (vocab_size, hidden_size), stored in bf16.
    pub weight: Bf16Tensor,
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl Embedding {
    pub fn new(weight: Bf16Tensor) -> Self {
        assert_eq!(weight.ndim(), 2);
        let vocab_size = weight.shape[0];
        let hidden_size = weight.shape[1];
        Self {
            weight,
            vocab_size,
            hidden_size,
        }
    }

    /// Look up the embedding for a single token ID.
    /// Returns a 1-D f32 tensor of shape (hidden_size,).
    pub fn forward(&self, token_id: u32) -> Tensor {
        self.weight.row_f32(token_id as usize)
    }

    /// Compute logits for the full vocabulary by projecting a hidden state
    /// through the (tied) embedding matrix: logits = weight @ hidden.
    /// `hidden`: shape (hidden_size,) in f32.
    /// Returns shape (vocab_size,) in f32.
    pub fn logits(&self, hidden: &Tensor) -> Tensor {
        self.weight.matvec(hidden)
    }
}
