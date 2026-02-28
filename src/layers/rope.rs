/// Rotary Position Embeddings — GPT-J interleaved style.
///
/// In the GPT-J convention the rotation is applied to consecutive pairs:
///   (x[0], x[1]), (x[2], x[3]), …, (x[d-2], x[d-1])
///
/// For each pair index `i` at position `pos`:
///   θ_i = pos / (θ_base ^ (2i / d))
///   (x[2i], x[2i+1]) → (x[2i]·cos(θ_i) − x[2i+1]·sin(θ_i),
///                         x[2i]·sin(θ_i) + x[2i+1]·cos(θ_i))
///
/// # Parameters
/// - `head_dim`: dimension of each attention head (128 for Cohere2).
/// - `theta_base`: RoPE base frequency (50 000 for Cohere2).
/// - `rotary_pct`: fraction of head_dim to rotate (1.0 → full rotation).
///
/// # Assumptions
/// - Input is a 1-D tensor whose length is `num_heads * head_dim`.
/// - The same position offset is applied uniformly (single-token decode step).

use super::tensor::Tensor;

pub struct RoPE {
    pub head_dim: usize,
    pub theta_base: f32,
    /// Number of dimensions to rotate per head. For `rotary_pct = 1.0` this
    /// equals `head_dim`.
    pub rotary_dim: usize,
}

impl RoPE {
    pub fn new(head_dim: usize, theta_base: f32, rotary_pct: f32) -> Self {
        let rotary_dim = ((head_dim as f32) * rotary_pct) as usize;
        // Must be even
        assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
        Self {
            head_dim,
            theta_base,
            rotary_dim,
        }
    }

    /// Apply RoPE to a vector of shape (num_heads * head_dim,) at a given
    /// position index.
    pub fn forward(&self, x: &Tensor, pos: usize) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let total = x.shape[0];
        assert_eq!(total % self.head_dim, 0);
        let num_heads = total / self.head_dim;

        let mut out = x.data.clone();

        for h in 0..num_heads {
            let base = h * self.head_dim;
            for i in 0..self.rotary_dim / 2 {
                let freq =
                    1.0 / (self.theta_base.powf(2.0 * i as f32 / self.rotary_dim as f32));
                let angle = pos as f32 * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let idx0 = base + 2 * i;
                let idx1 = base + 2 * i + 1;
                let x0 = x.data[idx0];
                let x1 = x.data[idx1];
                out[idx0] = x0 * cos_a - x1 * sin_a;
                out[idx1] = x0 * sin_a + x1 * cos_a;
            }
        }

        Tensor::new(out, x.shape.clone())
    }
}
