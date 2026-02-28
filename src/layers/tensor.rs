use half::bf16;
use rayon::prelude::*;

use super::simd::bf16_dot_f32;

/// A minimal, row-major, f32 tensor used for activations and intermediate values.
///
/// This is intentionally simple — no autograd, no strides, no broadcasting.
/// All data is stored as a flat `Vec<f32>` in row-major (C) order.
/// The tensor can be 1-D or 2-D; higher ranks are represented by reshaping.
///
/// # Assumptions
/// - All operations eagerly allocate new storage (no in-place mutation except
///   explicit `+=` style helpers).
/// - Performance is *not* a priority in this first implementation. Matmul is
///   a naive triple loop.
/// - This type is for *activations* (small, transient). Model weights are
///   stored in [`Bf16Tensor`] to halve memory usage.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// A row-major tensor that stores data in bf16 (bfloat16) to save memory.
///
/// Used for **model weights only** — the large matrices that dominate memory.
/// Arithmetic is performed by converting bf16 → f32 on-the-fly inside
/// `matvec` / `row` calls, so activations remain in full f32 precision.
///
/// Internally stores raw `u16` bits (identical to `half::bf16` layout) to
/// avoid depending on the `half` crate's `Vec` support.
#[derive(Debug, Clone)]
pub struct Bf16Tensor {
    /// Raw bf16 data stored as u16 bits, row-major order.
    pub data: Vec<u16>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with the given shape and data.
    /// Panics if `data.len()` does not match the product of `shape`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Tensor::new: data length {} does not match shape {:?} (numel={})",
            data.len(),
            shape,
            numel,
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: vec![0.0; numel],
            shape,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Return the number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape in-place. Panics if the new shape has a different total number
    /// of elements.
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let numel: usize = new_shape.iter().product();
        assert_eq!(numel, self.numel(), "reshape: incompatible shapes");
        self.shape = new_shape;
    }

    /// Return a reshaped *clone*.
    pub fn reshaped(&self, new_shape: Vec<usize>) -> Self {
        let mut t = self.clone();
        t.reshape(new_shape);
        t
    }

    /// Matrix multiply two 2-D tensors: (M, K) × (K, N) → (M, N).
    ///
    /// This is a naive O(MKN) implementation — correctness first.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2-D");
        assert_eq!(other.ndim(), 2, "matmul: rhs must be 2-D");
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(other.shape[0], k, "matmul: inner dims mismatch");
        let n = other.shape[1];

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        Tensor::new(out, vec![m, n])
    }

    /// Matrix-vector multiply: (M, K) × (K,) → (M,).
    /// Treats a 1-D tensor as a column vector.
    pub fn matvec(&self, vec: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matvec: matrix must be 2-D");
        assert_eq!(vec.ndim(), 1, "matvec: vec must be 1-D");
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(vec.shape[0], k, "matvec: dimension mismatch");

        let mut out = vec![0.0f32; m];
        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..k {
                sum += self.data[i * k + j] * vec.data[j];
            }
            out[i] = sum;
        }
        Tensor::new(out, vec![m])
    }

    /// Element-wise addition. Shapes must match exactly.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "add: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Element-wise multiply (Hadamard product). Shapes must match exactly.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "mul: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Scalar multiply.
    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Apply SiLU (Swish) activation element-wise: x * sigmoid(x).
    pub fn silu(&self) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Softmax over the last dimension.
    /// Assumes the tensor is 1-D or the softmax is over the full flat data
    /// when called on a 1-D slice.
    pub fn softmax(&self) -> Tensor {
        assert_eq!(self.ndim(), 1, "softmax: expecting 1-D tensor");
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let data: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Return a row slice from a 2-D tensor as a 1-D tensor.
    pub fn row(&self, i: usize) -> Tensor {
        assert_eq!(self.ndim(), 2);
        let cols = self.shape[1];
        let start = i * cols;
        Tensor::new(self.data[start..start + cols].to_vec(), vec![cols])
    }

    /// Transpose a 2-D tensor.
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2);
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }
        Tensor::new(data, vec![cols, rows])
    }

    /// Dot product of two 1-D tensors.
    pub fn dot(&self, other: &Tensor) -> f32 {
        assert_eq!(self.ndim(), 1);
        assert_eq!(other.ndim(), 1);
        assert_eq!(self.shape[0], other.shape[0]);
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Concatenate a list of 1-D tensors into a single 1-D tensor.
    pub fn cat(tensors: &[Tensor]) -> Tensor {
        let total: usize = tensors.iter().map(|t| {
            assert_eq!(t.ndim(), 1, "cat: all tensors must be 1-D");
            t.shape[0]
        }).sum();
        let mut data = Vec::with_capacity(total);
        for t in tensors {
            data.extend_from_slice(&t.data);
        }
        Tensor::new(data, vec![total])
    }

    /// Slice a 1-D tensor: returns elements [start..end).
    pub fn slice(&self, start: usize, end: usize) -> Tensor {
        assert_eq!(self.ndim(), 1);
        Tensor::new(self.data[start..end].to_vec(), vec![end - start])
    }
}

// ---------------------------------------------------------------------------
// Bf16Tensor — bf16 weight storage with on-the-fly f32 conversion
// ---------------------------------------------------------------------------

impl Bf16Tensor {
    /// Create from raw u16 bf16 bits and a shape.
    pub fn new(data: Vec<u16>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Bf16Tensor::new: data length {} does not match shape {:?} (numel={})",
            data.len(),
            shape,
            numel,
        );
        Self { data, shape }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Matrix-vector multiply: (M, K) × (K,) → (M,).
    /// The matrix (self) is bf16; the vector and output are f32.
    /// Conversion from bf16 → f32 happens element-by-element in the inner loop.
    ///
    /// Rows are processed in parallel via rayon.
    pub fn matvec(&self, vec: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "Bf16Tensor::matvec: matrix must be 2-D");
        assert_eq!(vec.ndim(), 1, "Bf16Tensor::matvec: vec must be 1-D");
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(vec.shape[0], k, "Bf16Tensor::matvec: dimension mismatch");

        let vec_data = &vec.data;
        let out: Vec<f32> = self
            .data
            .par_chunks_exact(k)
            .map(|row| bf16_dot_f32(row, vec_data))
            .collect();

        debug_assert_eq!(out.len(), m);
        Tensor::new(out, vec![m])
    }

    /// Return a single row from a 2-D bf16 tensor, converted to f32 1-D Tensor.
    pub fn row_f32(&self, i: usize) -> Tensor {
        assert_eq!(self.ndim(), 2);
        let cols = self.shape[1];
        let start = i * cols;
        let data: Vec<f32> = self.data[start..start + cols]
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect();
        Tensor::new(data, vec![cols])
    }
}
