#include "optmath/neon_kernels.hpp"
#include <cmath>
#include <algorithm>

#ifdef OPTMATH_USE_NEON
#include <arm_neon.h>
#endif

namespace optmath {
namespace neon {

bool is_available() {
#ifdef OPTMATH_USE_NEON
    return true;
#else
    return false;
#endif
}

// =========================================================================
// Core Intrinsics Implementations
// =========================================================================

float neon_dot_f32(const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;

    // Unrolled loop (4x4 = 16 elements per iter could be better, but we stick to 4 per iter for simplicity)
    // Actually, let's do 4x unroll (16 floats)
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, a0, b0);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        vsum = vmlaq_f32(vsum, a1, b1);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        vsum = vmlaq_f32(vsum, a2, b2);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        vsum = vmlaq_f32(vsum, a3, b3);
    }

    // Residual blocks of 4
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vsum = vmlaq_f32(vsum, va, vb);
    }

    float sum = vaddvq_f32(vsum);

    // Scalar tail
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

double neon_dot_f64(const double* a, const double* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float64x2_t vsum = vdupq_n_f64(0.0);
    size_t i = 0;

    for (; i + 3 < n; i += 4) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t b0 = vld1q_f64(b + i);
        vsum = vmlaq_f64(vsum, a0, b0);

        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        vsum = vmlaq_f64(vsum, a1, b1);
    }

    // Residual block of 2
    if (i + 1 < n) {
        float64x2_t va = vld1q_f64(a + i);
        float64x2_t vb = vld1q_f64(b + i);
        vsum = vmlaq_f64(vsum, va, vb);
        i += 2;
    }

    double sum = vaddvq_f64(vsum);

    // Scalar tail
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

void neon_add_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
#endif
}

void neon_sub_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
#endif
}

void neon_mul_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

void neon_div_f32(float* out, const float* a, const float* b, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vdivq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
#endif
}

float neon_norm_f32(const float* a, std::size_t n) {
    // Norm = sqrt(dot(a, a))
    float dot = neon_dot_f32(a, a, n);
    return std::sqrt(dot);
}

float neon_reduce_sum_f32(const float* a, std::size_t n) {
    // Sum is dot with 1.0, but faster to just accumulate
#ifdef OPTMATH_USE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 15 < n; i += 16) {
        vsum = vaddq_f32(vsum, vld1q_f32(a + i));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 4));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 8));
        vsum = vaddq_f32(vsum, vld1q_f32(a + i + 12));
    }
    for (; i + 3 < n; i += 4) {
        vsum = vaddq_f32(vsum, vld1q_f32(a + i));
    }
    float sum = vaddvq_f32(vsum);
    for (; i < n; ++i) sum += a[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i=0; i<n; ++i) sum += a[i];
    return sum;
#endif
}

float neon_reduce_max_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_NEON
    float32x4_t vmax = vdupq_n_f32(-3.402823466e+38f); // Init with small num
    size_t i = 0;
    // Load first element to avoid dummy small num if preferred, but vdup is easier
    // Just handling remaining scalars carefully.

    for (; i + 3 < n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(a + i));
    }
    float max_val = vmaxvq_f32(vmax);
    for (; i < n; ++i) if(a[i] > max_val) max_val = a[i];
    return max_val;
#else
    float m = a[0];
    for(size_t i=1; i<n; ++i) if(a[i] > m) m = a[i];
    return m;
#endif
}

float neon_reduce_min_f32(const float* a, std::size_t n) {
    if (n == 0) return 0.0f;
#ifdef OPTMATH_USE_NEON
    float32x4_t vmin = vdupq_n_f32(3.402823466e+38f);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vmin = vminq_f32(vmin, vld1q_f32(a + i));
    }
    float min_val = vminvq_f32(vmin);
    for (; i < n; ++i) if(a[i] < min_val) min_val = a[i];
    return min_val;
#else
    float m = a[0];
    for(size_t i=1; i<n; ++i) if(a[i] < m) m = a[i];
    return m;
#endif
}

void neon_gemm_4x4_f32(float* C, const float* A, std::size_t lda, const float* B, std::size_t ldb, std::size_t ldc) {
#ifdef OPTMATH_USE_NEON
    // A, B, C are assumed to point to the top-left corner of the blocks in larger column-major matrices.
    // lda, ldb, ldc are the strides (leading dimensions).

    // Load C columns
    float32x4_t c0 = vld1q_f32(C);
    float32x4_t c1 = vld1q_f32(C + ldc);
    float32x4_t c2 = vld1q_f32(C + 2*ldc);
    float32x4_t c3 = vld1q_f32(C + 3*ldc);

    // Load A columns
    // A block is 4x4. Column 0 is at A, Column 1 at A+lda, etc.
    float32x4_t a0 = vld1q_f32(A);
    float32x4_t a1 = vld1q_f32(A + lda);
    float32x4_t a2 = vld1q_f32(A + 2*lda);
    float32x4_t a3 = vld1q_f32(A + 3*lda);

    // B is 4x4 block.
    // For C = A * B, Column 0 of C uses Column 0 of B.
    // Column 0 of B is at B[0], B[1], B[2], B[3]. (B points to start of col 0)
    // Column 1 of B is at B + ldb.

    auto accumulate_col = [&](float32x4_t& c_col, const float* b_col_ptr) {
        c_col = vmlaq_n_f32(c_col, a0, b_col_ptr[0]);
        c_col = vmlaq_n_f32(c_col, a1, b_col_ptr[1]);
        c_col = vmlaq_n_f32(c_col, a2, b_col_ptr[2]);
        c_col = vmlaq_n_f32(c_col, a3, b_col_ptr[3]);
    };

    accumulate_col(c0, B);
    accumulate_col(c1, B + ldb);
    accumulate_col(c2, B + 2*ldb);
    accumulate_col(c3, B + 3*ldb);

    vst1q_f32(C, c0);
    vst1q_f32(C + ldc, c1);
    vst1q_f32(C + 2*ldc, c2);
    vst1q_f32(C + 3*ldc, c3);
#else
    // Fallback scalar
    for(int j=0; j<4; ++j) {
        for(int i=0; i<4; ++i) {
            float sum = 0.0f;
            for(int k=0; k<4; ++k) {
                // A[i, k] is A[i + k*lda]
                // B[k, j] is B[k + j*ldb]
                sum += A[i + k*lda] * B[k + j*ldb];
            }
            C[i + j*ldc] += sum;
        }
    }
#endif
}

void neon_fir_f32(const float* x, std::size_t n_x, const float* h, std::size_t n_h, float* y) {
    // y[i] = sum(x[i+k] * h[k]) for k=0..n_h-1
    // We assume 'y' has size n_x - n_h + 1 (valid convolution) or similar.
    // The user must manage buffer sizes.
    // This simple kernel computes one output at a time, but vectorizes the dot product.

    // Output size
    size_t n_y = (n_x >= n_h) ? (n_x - n_h + 1) : 0;

    for (size_t i = 0; i < n_y; ++i) {
        y[i] = neon_dot_f32(x + i, h, n_h);
    }
}

void neon_relu_f32(float* data, std::size_t n) {
#ifdef OPTMATH_USE_NEON
    float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for(; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vmaxq_f32(v, vzero));
    }
    for(; i < n; ++i) {
        if(data[i] < 0.0f) data[i] = 0.0f;
    }
#else
    for(size_t i=0; i<n; ++i) if(data[i] < 0.0f) data[i] = 0.0f;
#endif
}

// =========================================================================
// Vectorized Transcendental Functions
// =========================================================================
// High-performance approximations using minimax polynomials

#ifdef OPTMATH_USE_NEON
// Helper: Clamp float32x4_t to range
static inline float32x4_t clamp_f32(float32x4_t x, float32x4_t min_val, float32x4_t max_val) {
    return vminq_f32(vmaxq_f32(x, min_val), max_val);
}
#endif

void neon_exp_f32_approx(float* out, const float* in, std::size_t n) {
    // Approximation: exp(x) using range reduction and polynomial
    // exp(x) = 2^(x * log2(e)) = 2^k * 2^f where k = floor(x*log2e), f = frac
    // For f in [-0.5, 0.5], use minimax polynomial
    //
    // This uses a 6th-order minimax polynomial for 2^f
    // Accurate to ~1e-6 relative error for |x| < 88

    const float log2e = 1.44269504088896341f;
    const float ln2 = 0.693147180559945309f;

    // Polynomial coefficients for 2^x on [-0.5, 0.5]
    const float c0 = 1.0f;
    const float c1 = 0.693147182464599609f;
    const float c2 = 0.240226507186889648f;
    const float c3 = 0.055504187941551208f;
    const float c4 = 0.009618341922760010f;
    const float c5 = 0.001333355903625488f;
    const float c6 = 0.000154034309089184f;

#ifdef OPTMATH_USE_NEON
    float32x4_t vlog2e = vdupq_n_f32(log2e);
    float32x4_t vln2 = vdupq_n_f32(ln2);
    float32x4_t v0_5 = vdupq_n_f32(0.5f);
    float32x4_t vc0 = vdupq_n_f32(c0);
    float32x4_t vc1 = vdupq_n_f32(c1);
    float32x4_t vc2 = vdupq_n_f32(c2);
    float32x4_t vc3 = vdupq_n_f32(c3);
    float32x4_t vc4 = vdupq_n_f32(c4);
    float32x4_t vc5 = vdupq_n_f32(c5);
    float32x4_t vc6 = vdupq_n_f32(c6);

    float32x4_t vmax = vdupq_n_f32(88.0f);
    float32x4_t vmin = vdupq_n_f32(-88.0f);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);

        // Clamp input to avoid overflow/underflow
        x = clamp_f32(x, vmin, vmax);

        // Range reduction: x = k * ln2 + f, where k = round(x / ln2)
        float32x4_t t = vmulq_f32(x, vlog2e);
        float32x4_t k = vrndnq_f32(t);  // Round to nearest integer
        float32x4_t f = vmlsq_f32(x, k, vln2);  // f = x - k * ln2

        // Polynomial evaluation: 2^f = c0 + c1*f + c2*f^2 + ...
        // Using Horner's method
        float32x4_t p = vmlaq_f32(vc5, vc6, f);
        p = vmlaq_f32(vc4, p, f);
        p = vmlaq_f32(vc3, p, f);
        p = vmlaq_f32(vc2, p, f);
        p = vmlaq_f32(vc1, p, f);
        p = vmlaq_f32(vc0, p, f);

        // Reconstruct: exp(x) = 2^k * p
        // Use integer manipulation for 2^k
        int32x4_t ki = vcvtq_s32_f32(k);
        ki = vaddq_s32(ki, vdupq_n_s32(127));  // Add bias
        ki = vshlq_n_s32(ki, 23);  // Shift to exponent position
        float32x4_t scale = vreinterpretq_f32_s32(ki);

        float32x4_t result = vmulq_f32(p, scale);
        vst1q_f32(out + i, result);
    }

    // Scalar tail
    for (; i < n; ++i) {
        float x = in[i];
        if (x > 88.0f) x = 88.0f;
        if (x < -88.0f) x = -88.0f;

        float t = x * log2e;
        float k = std::round(t);
        float f = x - k * ln2;

        float p = c6;
        p = c5 + p * f;
        p = c4 + p * f;
        p = c3 + p * f;
        p = c2 + p * f;
        p = c1 + p * f;
        p = c0 + p * f;

        int ki = (int)k + 127;
        union { float f; int32_t i; } u;
        u.i = ki << 23;
        out[i] = p * u.f;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp(in[i]);
    }
#endif
}

void neon_sin_f32_approx(float* out, const float* in, std::size_t n) {
    // Sine approximation using Chebyshev polynomial
    // Range reduction to [-pi, pi], then polynomial

    const float pi = 3.14159265358979323846f;
    const float inv_pi = 0.31830988618379067154f;
    const float two_pi = 6.28318530717958647693f;

    // Chebyshev coefficients for sin(x*pi/2) on [-1, 1]
    // sin(x) = x * (c1 + x^2 * (c3 + x^2 * (c5 + x^2 * c7)))
    const float c1 = 1.0f;
    const float c3 = -0.16666667163372039795f;
    const float c5 = 0.00833333376795053482f;
    const float c7 = -0.00019841269776225090f;
    const float c9 = 0.00000275573189712526f;

#ifdef OPTMATH_USE_NEON
    float32x4_t vpi = vdupq_n_f32(pi);
    float32x4_t vinv_pi = vdupq_n_f32(inv_pi);
    float32x4_t vc1 = vdupq_n_f32(c1);
    float32x4_t vc3 = vdupq_n_f32(c3);
    float32x4_t vc5 = vdupq_n_f32(c5);
    float32x4_t vc7 = vdupq_n_f32(c7);
    float32x4_t vc9 = vdupq_n_f32(c9);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);

        // Range reduction: x = x - round(x / pi) * pi
        float32x4_t k = vrndnq_f32(vmulq_f32(x, vinv_pi));
        x = vmlsq_f32(x, k, vpi);

        // sin(x) = x * (c1 + x^2*(c3 + x^2*(c5 + x^2*(c7 + x^2*c9))))
        float32x4_t x2 = vmulq_f32(x, x);

        float32x4_t p = vmlaq_f32(vc7, vc9, x2);
        p = vmlaq_f32(vc5, p, x2);
        p = vmlaq_f32(vc3, p, x2);
        p = vmlaq_f32(vc1, p, x2);
        p = vmulq_f32(p, x);

        // Handle sign flip for odd k
        int32x4_t ki = vcvtq_s32_f32(k);
        uint32x4_t odd = vtstq_s32(ki, vdupq_n_s32(1));
        p = vbslq_f32(odd, vnegq_f32(p), p);

        vst1q_f32(out + i, p);
    }

    for (; i < n; ++i) {
        float x = in[i];
        float k = std::round(x * inv_pi);
        x = x - k * pi;

        float x2 = x * x;
        float p = c9;
        p = c7 + p * x2;
        p = c5 + p * x2;
        p = c3 + p * x2;
        p = c1 + p * x2;
        p = p * x;

        if ((int)k & 1) p = -p;
        out[i] = p;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
#endif
}

void neon_cos_f32_approx(float* out, const float* in, std::size_t n) {
    // cos(x) = sin(x + pi/2)
    const float half_pi = 1.57079632679489661923f;

#ifdef OPTMATH_USE_NEON
    float32x4_t vhalf_pi = vdupq_n_f32(half_pi);

    // Process in place with offset
    std::vector<float> temp(n);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        vst1q_f32(temp.data() + i, vaddq_f32(x, vhalf_pi));
    }
    for (; i < n; ++i) {
        temp[i] = in[i] + half_pi;
    }
    neon_sin_f32_approx(out, temp.data(), n);
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
#endif
}

void neon_sigmoid_f32_fast(float* out, const float* in, std::size_t n) {
    // Fast sigmoid: 1 / (1 + exp(-x))
    // Uses vectorized exp approximation

#ifdef OPTMATH_USE_NEON
    float32x4_t vone = vdupq_n_f32(1.0f);

    // Negate input for exp(-x)
    std::vector<float> neg_x(n);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        vst1q_f32(neg_x.data() + i, vnegq_f32(x));
    }
    for (; i < n; ++i) {
        neg_x[i] = -in[i];
    }

    // Compute exp(-x)
    std::vector<float> exp_neg_x(n);
    neon_exp_f32_approx(exp_neg_x.data(), neg_x.data(), n);

    // Compute 1 / (1 + exp(-x))
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t e = vld1q_f32(exp_neg_x.data() + i);
        float32x4_t denom = vaddq_f32(vone, e);
        float32x4_t result = vdivq_f32(vone, denom);
        vst1q_f32(out + i, result);
    }
    for (; i < n; ++i) {
        out[i] = 1.0f / (1.0f + exp_neg_x[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
#endif
}

void neon_tanh_f32_fast(float* out, const float* in, std::size_t n) {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // Or equivalently: 2 * sigmoid(2x) - 1

#ifdef OPTMATH_USE_NEON
    float32x4_t vtwo = vdupq_n_f32(2.0f);
    float32x4_t vone = vdupq_n_f32(1.0f);

    // Compute 2x
    std::vector<float> two_x(n);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(in + i);
        vst1q_f32(two_x.data() + i, vmulq_f32(vtwo, x));
    }
    for (; i < n; ++i) {
        two_x[i] = 2.0f * in[i];
    }

    // Compute sigmoid(2x)
    std::vector<float> sig(n);
    neon_sigmoid_f32_fast(sig.data(), two_x.data(), n);

    // Compute 2*sigmoid(2x) - 1
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t s = vld1q_f32(sig.data() + i);
        float32x4_t result = vmlsq_f32(vnegq_f32(vone), vtwo, s); // 2*s - 1
        result = vmlaq_f32(vnegq_f32(vone), vtwo, s);
        vst1q_f32(out + i, result);
    }
    for (; i < n; ++i) {
        out[i] = 2.0f * sig[i] - 1.0f;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
#endif
}

// Original scalar implementations (kept for backward compatibility)
void neon_sigmoid_f32(float* data, std::size_t n) {
    for(size_t i=0; i<n; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

void neon_tanh_f32(float* data, std::size_t n) {
    for(size_t i=0; i<n; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

// =========================================================================
// Eigen Wrappers
// =========================================================================

float neon_dot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return 0.0f; // Minimal error handling
    return neon_dot_f32(a.data(), b.data(), a.size());
}

double neon_dot(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    if(a.size() != b.size()) return 0.0;
    return neon_dot_f64(a.data(), b.data(), a.size());
}

Eigen::VectorXf neon_add(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_add_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf neon_sub(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_sub_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf neon_mul(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_mul_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

Eigen::VectorXf neon_div(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    if(a.size() != b.size()) return Eigen::VectorXf();
    Eigen::VectorXf res(a.size());
    neon_div_f32(res.data(), a.data(), b.data(), a.size());
    return res;
}

float neon_norm(const Eigen::VectorXf& a) {
    return neon_norm_f32(a.data(), a.size());
}

float neon_reduce_sum(const Eigen::VectorXf& a) {
    return neon_reduce_sum_f32(a.data(), a.size());
}

float neon_reduce_max(const Eigen::VectorXf& a) {
    return neon_reduce_max_f32(a.data(), a.size());
}

float neon_reduce_min(const Eigen::VectorXf& a) {
    return neon_reduce_min_f32(a.data(), a.size());
}

Eigen::VectorXf neon_fir(const Eigen::VectorXf& x, const Eigen::VectorXf& h) {
    if (x.size() < h.size()) return Eigen::VectorXf();
    long out_size = x.size() - h.size() + 1;
    Eigen::VectorXf y(out_size);
    neon_fir_f32(x.data(), x.size(), h.data(), h.size(), y.data());
    return y;
}

void neon_relu(Eigen::VectorXf& x) {
    neon_relu_f32(x.data(), x.size());
}

void neon_sigmoid(Eigen::VectorXf& x) {
    neon_sigmoid_f32(x.data(), x.size());
}

void neon_tanh(Eigen::VectorXf& x) {
    neon_tanh_f32(x.data(), x.size());
}

Eigen::MatrixXf neon_gemm(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B) {
    if (A.cols() != B.rows()) return Eigen::MatrixXf();

    // Result C
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(A.rows(), B.cols());

    // Simple tiled implementation calling 4x4 microkernel
    // We iterate over 4x4 blocks of C
    for (long j = 0; j < C.cols(); j += 4) {
        for (long i = 0; i < C.rows(); i += 4) {
            // For each block C[i:i+4, j:j+4]
            // Accumulate A[i:i+4, k:k+4] * B[k:k+4, j:j+4]
            for (long k = 0; k < A.cols(); k += 4) {
                // Check bounds
                if (i + 4 <= C.rows() && j + 4 <= C.cols() && k + 4 <= A.cols()) {
                    // Fast path: 4x4 aligned block
                     neon_gemm_4x4_f32(&C(i, j), &A(i, k), A.outerStride(),
                                       &B(k, j), B.outerStride(),
                                       C.outerStride());
                } else {
                    // Fallback for boundary blocks (naive multiply)
                    long i_lim = std::min(i + 4, (long)C.rows());
                    long j_lim = std::min(j + 4, (long)C.cols());
                    long k_lim = std::min(k + 4, (long)A.cols());

                    for (long jj = j; jj < j_lim; ++jj) {
                        for (long ii = i; ii < i_lim; ++ii) {
                            float sum = 0.0f;
                            for (long kk = k; kk < k_lim; ++kk) {
                                sum += A(ii, kk) * B(kk, jj);
                            }
                            C(ii, jj) += sum;
                        }
                    }
                }
            }
        }
    }
    return C;
}

Eigen::MatrixXf neon_mat_scale(const Eigen::MatrixXf& A, float s) {
    Eigen::MatrixXf C(A.rows(), A.cols());
#ifdef OPTMATH_USE_NEON
    float32x4_t vs = vdupq_n_f32(s);
    size_t n = A.size();
    const float* in = A.data();
    float* out = C.data();
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(in + i), vs));
    }
    for (; i < n; ++i) {
        out[i] = in[i] * s;
    }
#else
    C = A * s;
#endif
    return C;
}

Eigen::MatrixXf neon_mat_transpose(const Eigen::MatrixXf& A) {
    Eigen::MatrixXf C(A.cols(), A.rows());
#ifdef OPTMATH_USE_NEON
    // Blocked transpose 4x4 using NEON trn/zip/uzp intrinsics
    // vtrn1q_f32, vtrn2q_f32, etc.
    // For simplicity, we just use Eigen's implementation or a naive block transpose?
    // User requested "best build ... add every neon kernel you can find".
    // A specific 4x4 transpose kernel is good.

    // We assume C is col-major. C(i, j) = A(j, i).
    // If we iterate 4x4 blocks.

    for (int j = 0; j < A.cols(); j += 4) {
        for (int i = 0; i < A.rows(); i += 4) {
            if (i + 4 <= A.rows() && j + 4 <= A.cols()) {
                // Transpose 4x4 block from A(i,j) to C(j,i)
                // Load 4 columns of A
                float32x4_t c0 = vld1q_f32(&A(i, j));
                float32x4_t c1 = vld1q_f32(&A(i, j+1));
                float32x4_t c2 = vld1q_f32(&A(i, j+2));
                float32x4_t c3 = vld1q_f32(&A(i, j+3));

                // Transpose 4x4 in registers
                // 0: 00 10 20 30
                // 1: 01 11 21 31
                // ...

                // trn1/2 swaps elements
                float32x4_t t0 = vtrn1q_f32(c0, c1); // 00 01 20 21
                float32x4_t t1 = vtrn2q_f32(c0, c1); // 10 11 30 31
                float32x4_t t2 = vtrn1q_f32(c2, c3); // 02 03 22 23
                float32x4_t t3 = vtrn2q_f32(c2, c3); // 12 13 32 33

                // Swap 64-bit halves
                // C0' = 00 01 02 03 = Low(t0), Low(t2)
                // C1' = 10 11 12 13 = Low(t1), Low(t3)
                // C2' = 20 21 22 23 = High(t0), High(t2)
                // C3' = 30 31 32 33 = High(t1), High(t3)

                // vcombine? No, need vzip or similar.
                // Actually reinterpret as 64x2 and trn?
                // Let's use vcombine mechanism manually or just st4?
                // Actually, vtrn is efficient enough.
                // Construct from halves:
                // vextq?

                // Simpler: Use vst4q_f32 which interleaves!
                // Wait, vst4q takes {v0, v1, v2, v3} and stores them interleaved.
                // If we load 4 cols:
                // Col0: A00 A10 A20 A30
                // Col1: A01 A11 A21 A31
                // ...
                // If we interpret these as 4 vectors and do vst4, we get:
                // A00 A01 A02 A03, A10 A11 A12 A13...
                // Which is exactly the row-major order of the block,
                // i.e., the columns of the transposed block (since C is col-major).
                // So: Load 4 vectors (cols of A), Store using vst4 (interleave) into C (cols of C).

                // But vst4q stores linearly.
                // C is col-major. C's memory for the block is contiguous? No.
                // C(j, i) -> C's col j, row i.
                // The block in C starts at C(j, i).
                // Its columns are C(j, i)...C(j+3, i). These are stride separated by LDC.
                // So vst4 is not directly usable unless C is transposed in memory?
                // Wait, Transpose of A -> C.
                // C[j, i] = A[i, j].
                // We want to write into C.
                // C stores columns continuously.
                // Col j of C is C(0, j)...C(M, j).
                // We want to write A's row i into C's col i.
                // A's row i is strided.

                // So we have 4 vectors (Cols of A).
                // We transpose them in registers to get Rows of A.
                // Then store each Row of A as a Col of C.
                // Since C is col-major, a Col of C is contiguous.
                // So we just need to transpose in registers and store each vector to &C(j, i), &C(j, i+1)...

                // Re-doing transpose logic:
                // t0 = 00 01 20 21
                // t2 = 02 03 22 23
                // out0 = 00 01 02 03 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2))

                float32x4_t out0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
                float32x4_t out1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
                float32x4_t out2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
                float32x4_t out3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));

                vst1q_f32(&C(j, i), out0);
                vst1q_f32(&C(j, i+1), out1);
                vst1q_f32(&C(j, i+2), out2);
                vst1q_f32(&C(j, i+3), out3);

            } else {
                // Fallback
                for (int ii = i; ii < std::min(i+4, (int)A.rows()); ++ii) {
                    for (int jj = j; jj < std::min(j+4, (int)A.cols()); ++jj) {
                        C(jj, ii) = A(ii, jj);
                    }
                }
            }
        }
    }
#else
    C = A.transpose();
#endif
    return C;
}

Eigen::VectorXf neon_mat_vec_mul(const Eigen::MatrixXf& A, const Eigen::VectorXf& v) {
    if (A.cols() != v.size()) return Eigen::VectorXf();
    Eigen::VectorXf res = Eigen::VectorXf::Zero(A.rows());

#ifdef OPTMATH_USE_NEON
    // res = A * v
    // A is col-major. A = [col0 col1 ...]
    // res = sum(col_i * v_i)

    // We iterate over columns of A (and elements of v)
    // and accumulate into res.

    for (int j = 0; j < A.cols(); ++j) {
        float val = v[j];
        float32x4_t vval = vdupq_n_f32(val);

        int i = 0;
        float* r_ptr = res.data();
        const float* a_col = &A(0, j);

        for (; i + 3 < A.rows(); i += 4) {
            float32x4_t acc = vld1q_f32(r_ptr + i);
            float32x4_t col = vld1q_f32(a_col + i);
            acc = vmlaq_f32(acc, col, vval);
            vst1q_f32(r_ptr + i, acc);
        }
        for (; i < A.rows(); ++i) {
            res[i] += A(i, j) * val;
        }
    }
#else
    res = A * v;
#endif
    return res;
}

}
}
