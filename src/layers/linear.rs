/// Dense linear layer: y = x @ W^T (no bias).
///
/// Weights are stored as a bf16 2-D tensor of shape (out_features, in_features),
/// matching the safetensors layout. The forward pass computes `W @ x` for a
/// 1-D input vector, converting bf16 → f32 on-the-fly in the inner loop.
///
/// # Assumptions
/// - No bias term (Cohere2 attention & MLP projections are bias-free).
/// - Weights stay in bf16 to halve memory usage (~6.7 GB for the full model).

use super::tensor::{Bf16Tensor, Tensor};

pub struct Linear {
    /// Weight matrix of shape (out_features, in_features), stored in bf16.
    pub weight: Bf16Tensor,
}

impl Linear {
    pub fn new(weight: Bf16Tensor) -> Self {
        assert_eq!(weight.ndim(), 2, "Linear weight must be 2-D");
        Self { weight }
    }

    /// Forward pass for a single vector input of shape (in_features,).
    /// Returns a vector of shape (out_features,).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // weight is (out, in) in bf16, x is (in,) in f32  →  result is (out,) in f32
        self.weight.matvec(x)
    }
}
