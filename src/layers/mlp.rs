/// Gated MLP (SwiGLU) feed-forward block.
///
/// Computes:
///   output = down_proj(silu(gate_proj(x)) * up_proj(x))
///
/// This is the standard SwiGLU variant used in LLaMA, Cohere2, and many
/// modern transformer architectures.
///
/// # Assumptions
/// - `gate_proj` and `up_proj` have shape (intermediate_size, hidden_size).
/// - `down_proj` has shape (hidden_size, intermediate_size).
/// - No bias terms.

use super::linear::Linear;
use super::tensor::{Bf16Tensor, Tensor};

pub struct GatedMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl GatedMlp {
    /// Forward pass for a single hidden vector of shape (hidden_size,).
    /// Returns a vector of shape (hidden_size,).
    ///
    /// Uses `Bf16Tensor::fused_gate_up_matvec` to compute
    /// `silu(gate_proj(x)) * up_proj(x)` in a single parallel pass,
    /// avoiding two separate intermediate allocations and an element-wise mul.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = Bf16Tensor::fused_gate_up_matvec(
            &self.gate_proj.weight,
            &self.up_proj.weight,
            x,
        );
        self.down_proj.forward(&hidden)
    }
}
