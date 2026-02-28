/// Layer Normalization.
///
/// Cohere2 uses standard Layer Normalization (not RMSNorm):
///   y = (x - mean) / sqrt(var + eps) * weight
///
/// There is only a `weight` (gamma) parameter and no `bias` (beta).
///
/// # Assumptions
/// - Input is a 1-D vector (single token hidden state).
/// - `eps` = 1e-5 as specified in the model config.
/// - The weight vector is small (hidden_size = 2048), so we store it as f32.
///   It is converted from bf16 at load time via the helper constructor.

use half::bf16;
use super::tensor::Tensor;

pub struct LayerNorm {
    /// Weight (gamma) stored in f32 — only 2048 elements per layer, negligible.
    pub weight: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        assert_eq!(weight.ndim(), 1, "LayerNorm weight must be 1-D");
        Self { weight, eps }
    }

    /// Construct from raw bf16 u16-bits data, converting to f32.
    /// LayerNorm weights are tiny so f32 storage is fine.
    pub fn from_bf16(data: Vec<u16>, eps: f32) -> Self {
        let f32_data: Vec<f32> = data.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let n = f32_data.len();
        Self::new(Tensor::new(f32_data, vec![n]), eps)
    }

    /// Normalize a 1-D input vector.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let n = x.shape[0] as f32;

        let mean: f32 = x.data.iter().sum::<f32>() / n;
        let var: f32 = x.data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + self.eps).sqrt();

        let data: Vec<f32> = x
            .data
            .iter()
            .zip(self.weight.data.iter())
            .map(|(&xi, &wi)| (xi - mean) * inv_std * wi)
            .collect();

        Tensor::new(data, x.shape.clone())
    }
}
