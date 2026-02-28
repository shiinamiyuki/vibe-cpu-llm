/// SIMD-accelerated bf16·f32 dot product kernels.
///
/// This module provides platform-specific SIMD implementations for the
/// critical inner loop: computing the dot product of a bf16 weight row
/// with an f32 activation vector. This is the dominant cost in every
/// linear projection (Q/K/V/O projections, MLP gate/up/down, logit head).
///
/// # Supported platforms
///
/// - **aarch64 (NEON)**: bf16→f32 via `vshll_n_u16` (16-bit left-shift),
///   fused multiply-add via `vfmaq_f32`. Processes 32 elements per loop
///   iteration using 4 independent accumulators to hide FMA latency.
///
/// - **x86_64 (AVX2 + F16C)**: bf16→f32 via 16-bit left-shift with
///   `vpslld`/`vpmovzxwd`, fused multiply-add via `vfmadd231ps` (FMA3).
///   Processes 32 elements per loop iteration (4×8 accumulators).
///   Falls back to SSE2 scalar if AVX2+FMA are not available at runtime.
///
/// - **Fallback**: Scalar loop using `half::bf16::to_f32()` for any other
///   architecture.
///
/// # Safety
/// All SIMD paths use `unsafe` intrinsics but are wrapped in safe public
/// functions. Pointer validity is guaranteed by the caller passing valid
/// slices of the correct length.

// -----------------------------------------------------------------------
// Public API: dispatches to the best available implementation
// -----------------------------------------------------------------------

/// Compute the dot product of a bf16 weight row and an f32 vector.
///
/// `weights` and `vec` must have the same length `k`.
/// Returns the sum of `bf16_to_f32(weights[i]) * vec[i]` for i in 0..k.
#[inline]
pub fn bf16_dot_f32(weights: &[u16], vec: &[f32]) -> f32 {
    debug_assert_eq!(weights.len(), vec.len());

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64.
        return unsafe { neon::bf16_dot_f32_neon(weights, vec) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Runtime feature detection for AVX2 + FMA.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::bf16_dot_f32_avx2(weights, vec) };
        }
        // Fall through to scalar
    }

    #[allow(unreachable_code)]
    scalar_bf16_dot_f32(weights, vec)
}

/// Compute the dot product of two f32 slices.
///
/// `a` and `b` must have the same length.
/// Used for QK attention score computation (both slices are f32 activations).
#[inline]
pub fn f32_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::f32_dot_f32_neon(a, b) };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::f32_dot_f32_avx2(a, b) };
        }
    }

    #[allow(unreachable_code)]
    scalar_f32_dot_f32(a, b)
}

/// SAXPY: `out[i] += alpha * x[i]` for all i.
///
/// Used for the weighted value accumulation in attention:
/// `out_head += softmax_weight * v_data`.
#[inline]
pub fn f32_saxpy(out: &mut [f32], alpha: f32, x: &[f32]) {
    debug_assert_eq!(out.len(), x.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::f32_saxpy_neon(out, alpha, x) };
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { avx2::f32_saxpy_avx2(out, alpha, x) };
            return;
        }
    }

    #[allow(unreachable_code)]
    scalar_f32_saxpy(out, alpha, x);
}

/// Scalar fallback: works on any platform.
#[inline]
fn scalar_bf16_dot_f32(weights: &[u16], vec: &[f32]) -> f32 {
    use half::bf16;
    let mut sum = 0.0f32;
    for i in 0..weights.len() {
        let w = bf16::from_bits(weights[i]).to_f32();
        sum += w * vec[i];
    }
    sum
}

/// Scalar f32·f32 dot product fallback.
#[inline]
fn scalar_f32_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar SAXPY fallback: `out[i] += alpha * x[i]`.
#[inline]
fn scalar_f32_saxpy(out: &mut [f32], alpha: f32, x: &[f32]) {
    for i in 0..out.len() {
        out[i] += alpha * x[i];
    }
}

// -----------------------------------------------------------------------
// aarch64 NEON implementation
// -----------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Convert 4 bf16 values (as u16) to 4 f32 values via left-shift.
    ///
    /// bf16 has the same exponent and sign layout as f32 — the mantissa is
    /// simply truncated. So `(bits as u32) << 16` produces a valid f32.
    #[inline(always)]
    unsafe fn bf16x4_to_f32x4(v: uint16x4_t) -> float32x4_t {
        unsafe {
            // Zero-extend u16 → u32, then shift left by 16.
            let wide: uint32x4_t = vmovl_u16(v);
            let shifted: uint32x4_t = vshlq_n_u32(wide, 16);
            vreinterpretq_f32_u32(shifted)
        }
    }

    /// NEON-accelerated bf16·f32 dot product.
    ///
    /// Processes 16 elements per iteration using 4 accumulators (4×4 floats)
    /// to exploit NEON's dual-issue FMA pipelines. Handles the tail with a
    /// scalar loop.
    ///
    /// # Safety
    /// Caller must ensure `weights.len() == vec.len()`.
    /// NEON is architecturally guaranteed on aarch64.
    #[inline]
    pub unsafe fn bf16_dot_f32_neon(weights: &[u16], vec: &[f32]) -> f32 {
        let k = weights.len();
        let w_ptr = weights.as_ptr();
        let v_ptr = vec.as_ptr();

        unsafe {
            // 4 independent f32x4 accumulators (16 partial sums total)
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let chunks = k / 16;
            let mut i = 0usize;

            for _ in 0..chunks {
                // Load 16 bf16 weights as four 4-wide u16 loads.
                let w0 = vld1_u16(w_ptr.add(i));
                let w1 = vld1_u16(w_ptr.add(i + 4));
                let w2 = vld1_u16(w_ptr.add(i + 8));
                let w3 = vld1_u16(w_ptr.add(i + 12));

                // Convert bf16 → f32
                let fw0 = bf16x4_to_f32x4(w0);
                let fw1 = bf16x4_to_f32x4(w1);
                let fw2 = bf16x4_to_f32x4(w2);
                let fw3 = bf16x4_to_f32x4(w3);

                // Load 16 f32 activation values
                let a0 = vld1q_f32(v_ptr.add(i));
                let a1 = vld1q_f32(v_ptr.add(i + 4));
                let a2 = vld1q_f32(v_ptr.add(i + 8));
                let a3 = vld1q_f32(v_ptr.add(i + 12));

                // Fused multiply-add: acc += w * a
                acc0 = vfmaq_f32(acc0, fw0, a0);
                acc1 = vfmaq_f32(acc1, fw1, a1);
                acc2 = vfmaq_f32(acc2, fw2, a2);
                acc3 = vfmaq_f32(acc3, fw3, a3);

                i += 16;
            }

            // Reduce 4 accumulators → single f32
            let sum01 = vaddq_f32(acc0, acc1);
            let sum23 = vaddq_f32(acc2, acc3);
            let sum_all = vaddq_f32(sum01, sum23);
            let mut result = vaddvq_f32(sum_all);

            // Scalar tail for remaining elements
            while i < k {
                let w = f32::from_bits((weights[i] as u32) << 16);
                result += w * vec[i];
                i += 1;
            }

            result
        }
    }

    /// NEON-accelerated f32·f32 dot product.
    ///
    /// Same structure as bf16 variant but both inputs are already f32.
    /// Processes 16 elements per iteration with 4 accumulators.
    #[inline]
    pub unsafe fn f32_dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let chunks = k / 16;
            let mut i = 0usize;

            for _ in 0..chunks {
                let a0 = vld1q_f32(a_ptr.add(i));
                let a1 = vld1q_f32(a_ptr.add(i + 4));
                let a2 = vld1q_f32(a_ptr.add(i + 8));
                let a3 = vld1q_f32(a_ptr.add(i + 12));

                let b0 = vld1q_f32(b_ptr.add(i));
                let b1 = vld1q_f32(b_ptr.add(i + 4));
                let b2 = vld1q_f32(b_ptr.add(i + 8));
                let b3 = vld1q_f32(b_ptr.add(i + 12));

                acc0 = vfmaq_f32(acc0, a0, b0);
                acc1 = vfmaq_f32(acc1, a1, b1);
                acc2 = vfmaq_f32(acc2, a2, b2);
                acc3 = vfmaq_f32(acc3, a3, b3);

                i += 16;
            }

            let sum01 = vaddq_f32(acc0, acc1);
            let sum23 = vaddq_f32(acc2, acc3);
            let sum_all = vaddq_f32(sum01, sum23);
            let mut result = vaddvq_f32(sum_all);

            while i < k {
                result += a[i] * b[i];
                i += 1;
            }

            result
        }
    }

    /// NEON-accelerated SAXPY: `out[i] += alpha * x[i]`.
    ///
    /// Broadcasts `alpha` into a NEON register and uses FMA.
    /// Processes 16 elements per iteration.
    #[inline]
    pub unsafe fn f32_saxpy_neon(out: &mut [f32], alpha: f32, x: &[f32]) {
        let k = out.len();
        let o_ptr = out.as_mut_ptr();
        let x_ptr = x.as_ptr();

        unsafe {
            let alpha_v = vdupq_n_f32(alpha);
            let chunks = k / 16;
            let mut i = 0usize;

            for _ in 0..chunks {
                let mut o0 = vld1q_f32(o_ptr.add(i));
                let mut o1 = vld1q_f32(o_ptr.add(i + 4));
                let mut o2 = vld1q_f32(o_ptr.add(i + 8));
                let mut o3 = vld1q_f32(o_ptr.add(i + 12));

                let x0 = vld1q_f32(x_ptr.add(i));
                let x1 = vld1q_f32(x_ptr.add(i + 4));
                let x2 = vld1q_f32(x_ptr.add(i + 8));
                let x3 = vld1q_f32(x_ptr.add(i + 12));

                o0 = vfmaq_f32(o0, alpha_v, x0);
                o1 = vfmaq_f32(o1, alpha_v, x1);
                o2 = vfmaq_f32(o2, alpha_v, x2);
                o3 = vfmaq_f32(o3, alpha_v, x3);

                vst1q_f32(o_ptr.add(i), o0);
                vst1q_f32(o_ptr.add(i + 4), o1);
                vst1q_f32(o_ptr.add(i + 8), o2);
                vst1q_f32(o_ptr.add(i + 12), o3);

                i += 16;
            }

            while i < k {
                *o_ptr.add(i) += alpha * x[i];
                i += 1;
            }
        }
    }
}

// -----------------------------------------------------------------------
// x86_64 AVX2 + FMA implementation
// -----------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2+FMA accelerated bf16·f32 dot product.
    ///
    /// Processes 32 elements per iteration using 4×8 f32 accumulators.
    /// bf16→f32 conversion: zero-extend u16→u32 with `vpmovzxwd`, then
    /// shift left 16 with `vpslld`.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available (checked at call site
    /// via `is_x86_feature_detected!`).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn bf16_dot_f32_avx2(weights: &[u16], vec: &[f32]) -> f32 {
        let k = weights.len();
        let w_ptr = weights.as_ptr();
        let v_ptr = vec.as_ptr();

        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let chunks = k / 32;
            let mut i = 0usize;

            for _ in 0..chunks {
                // Load 8 u16 bf16 weights, zero-extend to u32, shift left 16 → f32
                // Repeat 4 times for 32 elements
                let w0_i16 = _mm_loadu_si128(w_ptr.add(i) as *const __m128i);
                let w0_i32 = _mm256_cvtepu16_epi32(w0_i16);
                let w0_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(w0_i32, 16));

                let w1_i16 = _mm_loadu_si128(w_ptr.add(i + 8) as *const __m128i);
                let w1_i32 = _mm256_cvtepu16_epi32(w1_i16);
                let w1_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(w1_i32, 16));

                let w2_i16 = _mm_loadu_si128(w_ptr.add(i + 16) as *const __m128i);
                let w2_i32 = _mm256_cvtepu16_epi32(w2_i16);
                let w2_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(w2_i32, 16));

                let w3_i16 = _mm_loadu_si128(w_ptr.add(i + 24) as *const __m128i);
                let w3_i32 = _mm256_cvtepu16_epi32(w3_i16);
                let w3_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(w3_i32, 16));

                // Load 32 f32 activations
                let a0 = _mm256_loadu_ps(v_ptr.add(i));
                let a1 = _mm256_loadu_ps(v_ptr.add(i + 8));
                let a2 = _mm256_loadu_ps(v_ptr.add(i + 16));
                let a3 = _mm256_loadu_ps(v_ptr.add(i + 24));

                // FMA: acc += w * a
                acc0 = _mm256_fmadd_ps(w0_f32, a0, acc0);
                acc1 = _mm256_fmadd_ps(w1_f32, a1, acc1);
                acc2 = _mm256_fmadd_ps(w2_f32, a2, acc2);
                acc3 = _mm256_fmadd_ps(w3_f32, a3, acc3);

                i += 32;
            }

            // Reduce 4 accumulators → single f32
            let sum01 = _mm256_add_ps(acc0, acc1);
            let sum23 = _mm256_add_ps(acc2, acc3);
            let sum_all = _mm256_add_ps(sum01, sum23);

            // Horizontal sum of 8 f32 lanes
            let hi128 = _mm256_extractf128_ps(sum_all, 1);
            let lo128 = _mm256_castps256_ps128(sum_all);
            let sum128 = _mm_add_ps(lo128, hi128);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result_ss = _mm_add_ss(sums, shuf2);
            let mut result = _mm_cvtss_f32(result_ss);

            // Scalar tail
            while i < k {
                let w = f32::from_bits((weights[i] as u32) << 16);
                result += w * vec[i];
                i += 1;
            }

            result
        }
    }

    /// AVX2+FMA accelerated f32·f32 dot product.
    ///
    /// Processes 32 elements per iteration with 4×8 accumulators.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn f32_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let chunks = k / 32;
            let mut i = 0usize;

            for _ in 0..chunks {
                let a0 = _mm256_loadu_ps(a_ptr.add(i));
                let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                let a2 = _mm256_loadu_ps(a_ptr.add(i + 16));
                let a3 = _mm256_loadu_ps(a_ptr.add(i + 24));

                let b0 = _mm256_loadu_ps(b_ptr.add(i));
                let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                let b2 = _mm256_loadu_ps(b_ptr.add(i + 16));
                let b3 = _mm256_loadu_ps(b_ptr.add(i + 24));

                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                acc3 = _mm256_fmadd_ps(a3, b3, acc3);

                i += 32;
            }

            let sum01 = _mm256_add_ps(acc0, acc1);
            let sum23 = _mm256_add_ps(acc2, acc3);
            let sum_all = _mm256_add_ps(sum01, sum23);

            let hi128 = _mm256_extractf128_ps(sum_all, 1);
            let lo128 = _mm256_castps256_ps128(sum_all);
            let sum128 = _mm_add_ps(lo128, hi128);
            let shuf = _mm_movehdup_ps(sum128);
            let sums = _mm_add_ps(sum128, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            let result_ss = _mm_add_ss(sums, shuf2);
            let mut result = _mm_cvtss_f32(result_ss);

            while i < k {
                result += a[i] * b[i];
                i += 1;
            }

            result
        }
    }

    /// AVX2+FMA accelerated SAXPY: `out[i] += alpha * x[i]`.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn f32_saxpy_avx2(out: &mut [f32], alpha: f32, x: &[f32]) {
        let k = out.len();
        let o_ptr = out.as_mut_ptr();
        let x_ptr = x.as_ptr();

        unsafe {
            let alpha_v = _mm256_set1_ps(alpha);
            let chunks = k / 32;
            let mut i = 0usize;

            for _ in 0..chunks {
                let mut o0 = _mm256_loadu_ps(o_ptr.add(i));
                let mut o1 = _mm256_loadu_ps(o_ptr.add(i + 8));
                let mut o2 = _mm256_loadu_ps(o_ptr.add(i + 16));
                let mut o3 = _mm256_loadu_ps(o_ptr.add(i + 24));

                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                o0 = _mm256_fmadd_ps(alpha_v, x0, o0);
                o1 = _mm256_fmadd_ps(alpha_v, x1, o1);
                o2 = _mm256_fmadd_ps(alpha_v, x2, o2);
                o3 = _mm256_fmadd_ps(alpha_v, x3, o3);

                _mm256_storeu_ps(o_ptr.add(i), o0);
                _mm256_storeu_ps(o_ptr.add(i + 8), o1);
                _mm256_storeu_ps(o_ptr.add(i + 16), o2);
                _mm256_storeu_ps(o_ptr.add(i + 24), o3);

                i += 32;
            }

            while i < k {
                *o_ptr.add(i) += alpha * x[i];
                i += 1;
            }
        }
    }
}
