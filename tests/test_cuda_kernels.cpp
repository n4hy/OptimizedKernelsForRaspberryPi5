/**
 * @file test_cuda_kernels.cpp
 * @brief Comprehensive tests for CUDA backend operations
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <complex>
#include <cmath>
#include <vector>
#include <random>

#ifdef OPTMATH_USE_CUDA
#include "optmath/cuda_backend.hpp"
#endif

constexpr float TOLERANCE = 1e-4f;
constexpr double TOLERANCE_D = 1e-10;

// ============================================================================
// Test Fixture
// ============================================================================

class CudaKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef OPTMATH_USE_CUDA
        if (!optmath::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        // Check if device is supported by toolkit before initializing
        if (!optmath::cuda::is_device_supported()) {
            GTEST_SKIP() << "GPU architecture not supported by CUDA toolkit (Blackwell requires CUDA 13.x+)";
        }
        if (!optmath::cuda::CudaContext::get().init()) {
            GTEST_SKIP() << "CUDA context initialization failed";
        }
#else
        GTEST_SKIP() << "CUDA not enabled in build";
#endif
    }

    void TearDown() override {
#ifdef OPTMATH_USE_CUDA
        if (optmath::cuda::is_available()) {
            optmath::cuda::synchronize();
        }
#endif
    }

    // Random number generator
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};
};

// ============================================================================
// Vector Operation Tests
// ============================================================================

#ifdef OPTMATH_USE_CUDA

TEST_F(CudaKernelTest, VectorAdd) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_add(a, b);
    Eigen::VectorXf expected = a + b;

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), expected(i), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, VectorMul) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_mul(a, b);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), a(i) * b(i), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, VectorScale) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    float scalar = 3.14159f;

    Eigen::VectorXf result = optmath::cuda::cuda_scale(a, scalar);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), a(i) * scalar, TOLERANCE);
    }
}

TEST_F(CudaKernelTest, VectorDot) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_dot(a, b);
    float expected = a.dot(b);

    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-3f);
}

TEST_F(CudaKernelTest, VectorSum) {
    const int n = 4096;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_sum(a);
    float expected = a.sum();

    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-3f);
}

TEST_F(CudaKernelTest, VectorMax) {
    const int n = 4096;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_max(a);
    float expected = a.maxCoeff();

    EXPECT_NEAR(result, expected, TOLERANCE);
}

TEST_F(CudaKernelTest, VectorMin) {
    const int n = 4096;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n);

    float result = optmath::cuda::cuda_min(a);
    float expected = a.minCoeff();

    EXPECT_NEAR(result, expected, TOLERANCE);
}

// ============================================================================
// Transcendental Function Tests
// ============================================================================

TEST_F(CudaKernelTest, VectorExp) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 0.5f; // Limit range

    Eigen::VectorXf result = optmath::cuda::cuda_exp(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::exp(a(i)), std::abs(std::exp(a(i))) * 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorLog) {
    const int n = 1024;
    Eigen::VectorXf a = (Eigen::VectorXf::Random(n).array().abs() + 0.1f).matrix(); // Positive values

    Eigen::VectorXf result = optmath::cuda::cuda_log(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::log(a(i)), 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorSin) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 6.28f;

    Eigen::VectorXf result = optmath::cuda::cuda_sin(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::sin(a(i)), 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorCos) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 6.28f;

    Eigen::VectorXf result = optmath::cuda::cuda_cos(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::cos(a(i)), 1e-3f);
    }
}

TEST_F(CudaKernelTest, VectorSqrt) {
    const int n = 1024;
    Eigen::VectorXf a = (Eigen::VectorXf::Random(n).array().abs() + 0.1f).matrix();

    Eigen::VectorXf result = optmath::cuda::cuda_sqrt(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::sqrt(a(i)), 1e-4f);
    }
}

// ============================================================================
// Activation Function Tests
// ============================================================================

TEST_F(CudaKernelTest, Sigmoid) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 10.0f;

    Eigen::VectorXf result = optmath::cuda::cuda_sigmoid(a);

    for (int i = 0; i < n; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-a(i)));
        EXPECT_NEAR(result(i), expected, 1e-4f);
    }
}

TEST_F(CudaKernelTest, Tanh) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 5.0f;

    Eigen::VectorXf result = optmath::cuda::cuda_tanh(a);

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(result(i), std::tanh(a(i)), 1e-4f);
    }
}

TEST_F(CudaKernelTest, ReLU) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 10.0f;

    Eigen::VectorXf result = optmath::cuda::cuda_relu(a);

    for (int i = 0; i < n; ++i) {
        float expected = std::max(0.0f, a(i));
        EXPECT_NEAR(result(i), expected, TOLERANCE);
    }
}

TEST_F(CudaKernelTest, LeakyReLU) {
    const int n = 1024;
    Eigen::VectorXf a = Eigen::VectorXf::Random(n) * 10.0f;
    float alpha = 0.1f;

    Eigen::VectorXf result = optmath::cuda::cuda_leaky_relu(a, alpha);

    for (int i = 0; i < n; ++i) {
        float expected = a(i) >= 0 ? a(i) : alpha * a(i);
        EXPECT_NEAR(result(i), expected, TOLERANCE);
    }
}

// ============================================================================
// Matrix Operation Tests
// ============================================================================

TEST_F(CudaKernelTest, MatrixGEMM) {
    const int m = 128, k = 64, n = 256;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(m, k);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(k, n);

    Eigen::MatrixXf result = optmath::cuda::cuda_gemm(a, b);
    Eigen::MatrixXf expected = a * b;

    // Note: CudaContext::init() enables TF32 on Ampere+, so this runs with an
    // 11-bit significand (TF32 is 19 bits TOTAL: 1 sign + 8 exponent + 10
    // explicit mantissa -- not a 19-bit mantissa). Measured ~2e-4 relative on
    // an RTX 5090; with k=64 accumulations on top, use 1% relative + 5e-3 abs.
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(result(i,j), expected(i,j), std::abs(expected(i,j)) * 1e-2f + 5e-3f);
        }
    }
}

// --- Double / complex GEMM ---------------------------------------------------
//
// These four entry points had no coverage at all. Dimensions are deliberately
// non-square and not multiples of any tile size (97 x 53 x 71) so a kernel that
// transposed an operand, or used the wrong leading dimension, cannot still
// agree with Eigen by symmetry.
//
// Tolerances are set from measurement, not guessed: on an RTX 5090 these run at
// 3.6e-15 (Dgemm), 8.2e-15 (Zgemm) and 1.1e-15 (Dgemv) max absolute error.
// The bounds below leave ~2 decades of headroom while staying ~11 decades
// tighter than TF32 would permit, so a silent drop to single precision or to
// tensor-core math on the FP64 paths fails here rather than passing quietly.

namespace {
constexpr int GEMM_M = 97, GEMM_K = 53, GEMM_N = 71;
}  // namespace

TEST_F(CudaKernelTest, MatrixGEMM_F64) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(GEMM_M, GEMM_K);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(GEMM_K, GEMM_N);

    Eigen::MatrixXd result = optmath::cuda::cuda_gemm(a, b);
    Eigen::MatrixXd expected = a * b;

    ASSERT_EQ(result.rows(), GEMM_M);
    ASSERT_EQ(result.cols(), GEMM_N);
    for (int i = 0; i < GEMM_M; ++i) {
        for (int j = 0; j < GEMM_N; ++j) {
            EXPECT_NEAR(result(i, j), expected(i, j),
                        std::abs(expected(i, j)) * 1e-12 + 1e-12)
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(CudaKernelTest, MatrixGEMV_F64) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(GEMM_M, GEMM_K);
    Eigen::VectorXd x = Eigen::VectorXd::Random(GEMM_K);

    Eigen::VectorXd result = optmath::cuda::cuda_gemv(a, x);
    Eigen::VectorXd expected = a * x;

    ASSERT_EQ(result.size(), GEMM_M);
    for (int i = 0; i < GEMM_M; ++i) {
        EXPECT_NEAR(result(i), expected(i),
                    std::abs(expected(i)) * 1e-12 + 1e-12) << "at " << i;
    }
}

TEST_F(CudaKernelTest, MatrixGEMM_ComplexF32) {
    Eigen::MatrixXcf a = Eigen::MatrixXcf::Random(GEMM_M, GEMM_K);
    Eigen::MatrixXcf b = Eigen::MatrixXcf::Random(GEMM_K, GEMM_N);

    Eigen::MatrixXcf result = optmath::cuda::cuda_gemm(a, b);
    Eigen::MatrixXcf expected = a * b;

    ASSERT_EQ(result.rows(), GEMM_M);
    ASSERT_EQ(result.cols(), GEMM_N);
    // cublasCgemm is a SINGLE-precision path, so it inherits the context's TF32
    // math mode just as cuda_gemm(MatrixXf) does: measured 2.6e-4 relative here
    // against 2.6e-7 under CUBLAS_DEFAULT_MATH. Tolerance matches MatrixGEMM.
    for (int i = 0; i < GEMM_M; ++i) {
        for (int j = 0; j < GEMM_N; ++j) {
            EXPECT_NEAR(result(i, j).real(), expected(i, j).real(),
                        std::abs(expected(i, j)) * 1e-2f + 5e-3f)
                << "real at (" << i << ", " << j << ")";
            EXPECT_NEAR(result(i, j).imag(), expected(i, j).imag(),
                        std::abs(expected(i, j)) * 1e-2f + 5e-3f)
                << "imag at (" << i << ", " << j << ")";
        }
    }
}

TEST_F(CudaKernelTest, MatrixGEMM_ComplexF64) {
    Eigen::MatrixXcd a = Eigen::MatrixXcd::Random(GEMM_M, GEMM_K);
    Eigen::MatrixXcd b = Eigen::MatrixXcd::Random(GEMM_K, GEMM_N);

    Eigen::MatrixXcd result = optmath::cuda::cuda_gemm(a, b);
    Eigen::MatrixXcd expected = a * b;

    ASSERT_EQ(result.rows(), GEMM_M);
    ASSERT_EQ(result.cols(), GEMM_N);
    for (int i = 0; i < GEMM_M; ++i) {
        for (int j = 0; j < GEMM_N; ++j) {
            EXPECT_NEAR(result(i, j).real(), expected(i, j).real(),
                        std::abs(expected(i, j)) * 1e-12 + 1e-12)
                << "real at (" << i << ", " << j << ")";
            EXPECT_NEAR(result(i, j).imag(), expected(i, j).imag(),
                        std::abs(expected(i, j)) * 1e-12 + 1e-12)
                << "imag at (" << i << ", " << j << ")";
        }
    }
}

// A complex GEMM that dropped the cross terms (ac - bd / ad + bc) would still
// match on operands that happen to be real or imaginary only. Multiply two
// purely imaginary matrices: every entry of the product must be real and
// negative-definite in sign relative to the real-operand product.
TEST_F(CudaKernelTest, MatrixGEMM_ComplexF64_CrossTerms) {
    Eigen::MatrixXd ar = Eigen::MatrixXd::Random(GEMM_M, GEMM_K);
    Eigen::MatrixXd br = Eigen::MatrixXd::Random(GEMM_K, GEMM_N);
    Eigen::MatrixXcd a = ar.cast<std::complex<double>>() * std::complex<double>(0, 1);
    Eigen::MatrixXcd b = br.cast<std::complex<double>>() * std::complex<double>(0, 1);

    Eigen::MatrixXcd result = optmath::cuda::cuda_gemm(a, b);
    Eigen::MatrixXd expected_real = -(ar * br);  // (i*A)(i*B) = -A*B

    for (int i = 0; i < GEMM_M; ++i) {
        for (int j = 0; j < GEMM_N; ++j) {
            EXPECT_NEAR(result(i, j).real(), expected_real(i, j),
                        std::abs(expected_real(i, j)) * 1e-12 + 1e-12)
                << "real at (" << i << ", " << j << ")";
            EXPECT_NEAR(result(i, j).imag(), 0.0, 1e-12)
                << "imag at (" << i << ", " << j << ")";
        }
    }
}

// A dimension mismatch must return empty rather than reading past the operand:
// the H2D copy sizes are computed from A.cols(), so an unguarded call copies
// sizeof(T)*K*N bytes out of a buffer holding only B.rows()*N.
TEST_F(CudaKernelTest, MatrixGEMMDimensionMismatchReturnsEmpty) {
    Eigen::MatrixXf af = Eigen::MatrixXf::Random(GEMM_M, GEMM_K);
    Eigen::MatrixXd ad = Eigen::MatrixXd::Random(GEMM_M, GEMM_K);
    Eigen::MatrixXcf ac = Eigen::MatrixXcf::Random(GEMM_M, GEMM_K);
    Eigen::MatrixXcd az = Eigen::MatrixXcd::Random(GEMM_M, GEMM_K);

    EXPECT_EQ(optmath::cuda::cuda_gemm(
                  af, Eigen::MatrixXf::Random(GEMM_K + 3, GEMM_N)).size(), 0);
    EXPECT_EQ(optmath::cuda::cuda_gemm(
                  ad, Eigen::MatrixXd::Random(GEMM_K + 3, GEMM_N)).size(), 0);
    EXPECT_EQ(optmath::cuda::cuda_gemm(
                  ac, Eigen::MatrixXcf::Random(GEMM_K + 3, GEMM_N)).size(), 0);
    EXPECT_EQ(optmath::cuda::cuda_gemm(
                  az, Eigen::MatrixXcd::Random(GEMM_K + 3, GEMM_N)).size(), 0);

    EXPECT_EQ(optmath::cuda::cuda_gemv(
                  af, Eigen::VectorXf::Random(GEMM_K + 3)).size(), 0);
    EXPECT_EQ(optmath::cuda::cuda_gemv(
                  ad, Eigen::VectorXd::Random(GEMM_K + 3)).size(), 0);
}

// The transA/transB flags select the leading dimensions inside
// cuda_mat_mul_f64 (lda = transA ? K : M, ldb = transB ? N : K). Getting either
// wrong still produces a correctly-shaped matrix full of plausible numbers, so
// only an explicit check against Eigen catches it. All four flag combinations
// are exercised; the operands are non-square so a swapped lda/ldb cannot happen
// to be valid. This is also the only direct test of the raw-pointer API --
// everything else goes through the Eigen wrappers.
TEST_F(CudaKernelTest, MatMulF64RawTransposeFlags) {
    struct Case { bool transA, transB; };
    const Case cases[] = {{false,false},{true,false},{false,true},{true,true}};

    for (const Case& c : cases) {
        // op(A) is M x K and op(B) is K x N, so the stored operands are
        // transposed relative to that when the corresponding flag is set.
        const int Ar = c.transA ? GEMM_K : GEMM_M;
        const int Ac = c.transA ? GEMM_M : GEMM_K;
        const int Br = c.transB ? GEMM_N : GEMM_K;
        const int Bc = c.transB ? GEMM_K : GEMM_N;

        Eigen::MatrixXd A = Eigen::MatrixXd::Random(Ar, Ac);
        Eigen::MatrixXd B = Eigen::MatrixXd::Random(Br, Bc);
        Eigen::MatrixXd opA = c.transA ? Eigen::MatrixXd(A.transpose()) : A;
        Eigen::MatrixXd opB = c.transB ? Eigen::MatrixXd(B.transpose()) : B;
        Eigen::MatrixXd expected = opA * opB;

        double *dA = nullptr, *dB = nullptr, *dC = nullptr;
        ASSERT_EQ(cudaMalloc(&dA, sizeof(double) * Ar * Ac), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dB, sizeof(double) * Br * Bc), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dC, sizeof(double) * GEMM_M * GEMM_N), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dA, A.data(), sizeof(double) * Ar * Ac,
                             cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dB, B.data(), sizeof(double) * Br * Bc,
                             cudaMemcpyHostToDevice), cudaSuccess);

        const bool ok = optmath::cuda::cuda_mat_mul_f64(
            dC, dA, dB, GEMM_M, GEMM_N, GEMM_K, c.transA, c.transB);
        EXPECT_TRUE(ok) << "transA=" << c.transA << " transB=" << c.transB;

        Eigen::MatrixXd result(GEMM_M, GEMM_N);
        if (ok) {
            ASSERT_EQ(cudaMemcpy(result.data(), dC,
                                 sizeof(double) * GEMM_M * GEMM_N,
                                 cudaMemcpyDeviceToHost), cudaSuccess);
            for (int i = 0; i < GEMM_M; ++i) {
                for (int j = 0; j < GEMM_N; ++j) {
                    ASSERT_NEAR(result(i, j), expected(i, j),
                                std::abs(expected(i, j)) * 1e-12 + 1e-12)
                        << "transA=" << c.transA << " transB=" << c.transB
                        << " at (" << i << ", " << j << ")";
                }
            }
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}

TEST_F(CudaKernelTest, MatrixGEMV) {
    const int m = 256, n = 128;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(m, n);
    Eigen::VectorXf x = Eigen::VectorXf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_gemv(a, x);
    Eigen::VectorXf expected = a * x;

    for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(result(i), expected(i), std::abs(expected(i)) * 1e-3f + 1e-4f);
    }
}

TEST_F(CudaKernelTest, MatrixTranspose) {
    const int m = 128, n = 64;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(m, n);

    Eigen::MatrixXf result = optmath::cuda::cuda_transpose(a);

    EXPECT_EQ(result.rows(), n);
    EXPECT_EQ(result.cols(), m);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(result(j, i), a(i, j), TOLERANCE);
        }
    }
}

// ============================================================================
// Complex Number Tests
// ============================================================================

TEST_F(CudaKernelTest, ComplexMul) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf result = optmath::cuda::cuda_complex_mul(a, b);

    for (int i = 0; i < n; ++i) {
        std::complex<float> expected = a(i) * b(i);
        EXPECT_NEAR(result(i).real(), expected.real(), TOLERANCE);
        EXPECT_NEAR(result(i).imag(), expected.imag(), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, ComplexConjMul) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf result = optmath::cuda::cuda_complex_conj_mul(a, b);

    for (int i = 0; i < n; ++i) {
        std::complex<float> expected = a(i) * std::conj(b(i));
        EXPECT_NEAR(result(i).real(), expected.real(), TOLERANCE);
        EXPECT_NEAR(result(i).imag(), expected.imag(), TOLERANCE);
    }
}

TEST_F(CudaKernelTest, ComplexMagnitude) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_complex_abs(a);

    for (int i = 0; i < n; ++i) {
        float expected = std::abs(a(i));
        EXPECT_NEAR(result(i), expected, TOLERANCE);
    }
}

TEST_F(CudaKernelTest, ComplexPhase) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);

    Eigen::VectorXf result = optmath::cuda::cuda_complex_arg(a);

    for (int i = 0; i < n; ++i) {
        float expected = std::arg(a(i));
        EXPECT_NEAR(result(i), expected, 1e-3f);
    }
}

TEST_F(CudaKernelTest, ComplexDot) {
    const int n = 1024;
    Eigen::VectorXcf a = Eigen::VectorXcf::Random(n);
    Eigen::VectorXcf b = Eigen::VectorXcf::Random(n);

    std::complex<float> result = optmath::cuda::cuda_complex_dot(a, b);
    std::complex<float> expected = a.dot(b);

    EXPECT_NEAR(result.real(), expected.real(), std::abs(expected.real()) * 1e-3f + 1e-4f);
    EXPECT_NEAR(result.imag(), expected.imag(), std::abs(expected.imag()) * 1e-3f + 1e-4f);
}

// ============================================================================
// FFT Tests
// ============================================================================

TEST_F(CudaKernelTest, FFTForwardInverse) {
    const int n = 1024;
    Eigen::VectorXcf input = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf fft_result = optmath::cuda::cuda_fft(input);
    Eigen::VectorXcf ifft_result = optmath::cuda::cuda_ifft(fft_result);

    // IFFT should recover the original signal
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(ifft_result(i).real(), input(i).real(), 1e-3f);
        EXPECT_NEAR(ifft_result(i).imag(), input(i).imag(), 1e-3f);
    }
}

TEST_F(CudaKernelTest, FFTParseval) {
    const int n = 1024;
    Eigen::VectorXcf input = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf fft_result = optmath::cuda::cuda_fft(input);

    // Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
    float time_energy = input.squaredNorm();
    float freq_energy = fft_result.squaredNorm() / n;

    EXPECT_NEAR(time_energy, freq_energy, time_energy * 1e-3f);
}

// ============================================================================
// Convolution Tests
// ============================================================================

TEST_F(CudaKernelTest, Convolution1D) {
    const int n = 256;
    const int k = 16;
    Eigen::VectorXf signal = Eigen::VectorXf::Random(n);
    Eigen::VectorXf kernel = Eigen::VectorXf::Random(k);

    Eigen::VectorXf result = optmath::cuda::cuda_convolve_1d(signal, kernel);

    // Manual convolution check for a few points
    int mid = n / 2;
    float expected = 0;
    for (int j = 0; j < k && (mid - j) >= 0; ++j) {
        expected += signal(mid - j) * kernel(j);
    }

    // Allow some tolerance due to boundary handling differences
    EXPECT_NEAR(result(mid), expected, std::abs(expected) * 0.1f + 1e-3f);
}

// ============================================================================
// Cholesky Decomposition Tests (cuSOLVER)
// ============================================================================

TEST_F(CudaKernelTest, CholeskySmall) {
    // Create a small symmetric positive definite matrix
    // A = B * B^T + diagonal for positive definiteness
    const int n = 4;
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf A = B * B.transpose() + n * Eigen::MatrixXf::Identity(n, n);

    Eigen::MatrixXf L = optmath::cuda::cuda_cholesky(A);

    // Verify L is lower triangular
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            EXPECT_NEAR(L(i, j), 0.0f, TOLERANCE);
        }
    }

    // Verify L * L^T = A
    Eigen::MatrixXf reconstructed = L * L.transpose();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            EXPECT_NEAR(reconstructed(i, j), A(i, j), TOLERANCE * 10);
        }
    }
}

TEST_F(CudaKernelTest, CholeskyMedium) {
    // Medium-sized matrix
    const int n = 64;
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf A = B * B.transpose() + n * Eigen::MatrixXf::Identity(n, n);

    Eigen::MatrixXf L = optmath::cuda::cuda_cholesky(A);

    // Verify reconstruction
    Eigen::MatrixXf reconstructed = L * L.transpose();
    float max_error = (reconstructed - A).cwiseAbs().maxCoeff();
    EXPECT_LT(max_error, 1e-3f);
}

TEST_F(CudaKernelTest, CholeskyLarge) {
    // Large matrix to exercise GPU parallelism
    const int n = 512;
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf A = B * B.transpose() + n * Eigen::MatrixXf::Identity(n, n);

    Eigen::MatrixXf L = optmath::cuda::cuda_cholesky(A);

    // Verify dimensions
    EXPECT_EQ(L.rows(), n);
    EXPECT_EQ(L.cols(), n);

    // Spot check reconstruction
    Eigen::MatrixXf reconstructed = L * L.transpose();
    EXPECT_NEAR(reconstructed(0, 0), A(0, 0), 1e-2f);
    EXPECT_NEAR(reconstructed(n/2, n/2), A(n/2, n/2), 1e-2f);
    EXPECT_NEAR(reconstructed(n-1, n-1), A(n-1, n-1), 1e-2f);
}

TEST_F(CudaKernelTest, CholeskySolve) {
    const int n = 32;
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf A = B * B.transpose() + n * Eigen::MatrixXf::Identity(n, n);
    Eigen::VectorXf b = Eigen::VectorXf::Random(n);

    // Compute Cholesky factor
    Eigen::MatrixXf L = optmath::cuda::cuda_cholesky(A);

    // Solve Ax = b using the Cholesky factor
    Eigen::VectorXf x = optmath::cuda::cuda_cholesky_solve(L, b);

    // Verify A * x = b
    Eigen::VectorXf Ax = A * x;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(Ax(i), b(i), TOLERANCE * 100);
    }
}

TEST_F(CudaKernelTest, CholeskyInverse) {
    const int n = 16;
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf A = B * B.transpose() + n * Eigen::MatrixXf::Identity(n, n);

    // Compute Cholesky factor
    Eigen::MatrixXf L = optmath::cuda::cuda_cholesky(A);

    // Compute inverse
    Eigen::MatrixXf Ainv = optmath::cuda::cuda_cholesky_inverse(L);

    // Verify A * A^{-1} = I
    Eigen::MatrixXf product = A * Ainv;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(product(i, j), expected, 1e-3f);
        }
    }
}

TEST_F(CudaKernelTest, CholeskyDoublePrecision) {
    const int n = 32;
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd A = B * B.transpose() + n * Eigen::MatrixXd::Identity(n, n);

    Eigen::MatrixXd L = optmath::cuda::cuda_cholesky_f64(A);

    // Verify reconstruction with double precision
    Eigen::MatrixXd reconstructed = L * L.transpose();
    double max_error = (reconstructed - A).cwiseAbs().maxCoeff();
    EXPECT_LT(max_error, 1e-10);
}

TEST_F(CudaKernelTest, CholeskyNotPositiveDefinite) {
    // Create a matrix that is NOT positive definite
    const int n = 4;
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n, n);
    A(0, 0) = -1.0f;  // Negative diagonal makes it not positive definite

    Eigen::MatrixXf L = optmath::cuda::cuda_cholesky(A);

    // Should return zero matrix for non-positive-definite input
    EXPECT_NEAR(L.norm(), 0.0f, TOLERANCE);
}

// ============================================================================
// Large Scale Performance Tests
// ============================================================================

TEST_F(CudaKernelTest, LargeMatrixMultiply) {
    const int size = 1024;
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(size, size);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(size, size);

    Eigen::MatrixXf result = optmath::cuda::cuda_gemm(a, b);

    // Just verify it completes without error
    EXPECT_EQ(result.rows(), size);
    EXPECT_EQ(result.cols(), size);

    // Spot check a few elements
    Eigen::MatrixXf expected = a * b;
    EXPECT_NEAR(result(0, 0), expected(0, 0), std::abs(expected(0, 0)) * 1e-2f + 1e-2f);
    EXPECT_NEAR(result(size-1, size-1), expected(size-1, size-1),
                std::abs(expected(size-1, size-1)) * 1e-2f + 1e-2f);
}

TEST_F(CudaKernelTest, LargeFFT) {
    const int n = 65536;  // 64k point FFT
    Eigen::VectorXcf input = Eigen::VectorXcf::Random(n);

    Eigen::VectorXcf result = optmath::cuda::cuda_fft(input);

    EXPECT_EQ(result.size(), n);

    // Verify inverse recovers original
    Eigen::VectorXcf recovered = optmath::cuda::cuda_ifft(result);
    EXPECT_NEAR(recovered(0).real(), input(0).real(), 1e-2f);
}

#endif // OPTMATH_USE_CUDA

// Main
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
