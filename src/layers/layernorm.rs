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

    /// Normalize a 1-D input vector, returning a new allocation.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let out = x.data.clone();
        let result = self.norm_slice(out);
        Tensor::new(result, x.shape.clone())
    }

    /// Normalize `x` in-place: no allocation for the output.
    /// Overwrites `x.data` with the normalized values.
    pub fn forward_inplace(&self, x: &mut Tensor) {
        assert_eq!(x.ndim(), 1);
        let normed = self.norm_slice(std::mem::take(&mut x.data));
        x.data = normed;
    }

    /// Fused residual-add + layer-norm.
    ///
    /// Performs in a single allocation:
    ///   1. `residual[i] += d1[i] + d2[i]`  (in-place, no temp tensor)
    ///   2. Normalizes `residual` and returns the normed copy.
    ///
    /// This replaces the three-step pattern in the decoder loop:
    ///   `hidden = hidden.add(&a).add(&b)` then `layernorm.forward(&hidden)`
    /// with a single pass that updates `hidden` in place and returns the normed
    /// version for the next layer's input.
    ///
    /// `residual` is updated in-place (becomes `x + d1 + d2`).
    /// Returns the layer-normed copy.
    pub fn add_and_norm(&self, residual: &mut Tensor, d1: &Tensor, d2: &Tensor) -> Tensor {
        assert_eq!(residual.ndim(), 1);
        debug_assert_eq!(residual.shape[0], d1.shape[0]);
        debug_assert_eq!(residual.shape[0], d2.shape[0]);

        // Step 1: accumulate residual in-place (single pass, no temp allocation)
        for i in 0..residual.data.len() {
            residual.data[i] += d1.data[i] + d2.data[i];
        }

        // Step 2: normalize — clones into a new Vec, then normalizes it
        let normed = self.norm_slice(residual.data.clone());
        Tensor::new(normed, residual.shape.clone())
    }

    /// Core normalization: computes mean, variance, applies weight.
    /// Takes ownership of `data`, normalizes in place, returns it.
    #[inline]
    fn norm_slice(&self, mut data: Vec<f32>) -> Vec<f32> {
        let n = data.len() as f32;
        let mean: f32 = data.iter().sum::<f32>() / n;
        let var: f32 = data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + self.eps).sqrt();
        for (x, &w) in data.iter_mut().zip(self.weight.data.iter()) {
            *x = (*x - mean) * inv_std * w;
        }
        data
    }
}
